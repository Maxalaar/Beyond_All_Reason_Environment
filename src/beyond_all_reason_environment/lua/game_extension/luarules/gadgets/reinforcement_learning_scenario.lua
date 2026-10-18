-- Synced side of the scenarios (see scenario.py): places the units of a scenario and sets their states.
--
-- Two ways to start a scenario:
--
-- 1. Combat environment (combat_environment.py): the BridgeAI (C Skirmish AI, on the first agent team) sends its
--    requests via gadget:RecvSkirmishAIMessage. The BridgeAI calls COMMAND_CALL_LUA_RULES, which invokes this
--    handler synchronously, in the same frame — no server-lead delay.
--      "rl_reset|<episode>|<team>,<unit>,<x>,<z>;...|<t0_fire>,<t0_move>,<t1_fire>,<t1_move>,..."
--         Returns: "<unit_id>,<unit_id>,..." (spawn order).
--      "rl_actions|<snap radius>|<unit>,<command>,<x>,<z>;..."
--         Orders of the agents of every agent team (a Skirmish AI can only order its own team's units, so the
--         gadget gives them all). "attack" targets the enemy of the unit's allyteam nearest to (x, z) within the
--         snap radius, otherwise it is ignored. Returns "".
--      "rl_commands|<first index>"
--         Current order of every unit, "<unit>,<command>,<target x>,<target z>;..." (empty target: none), from
--         the first index on; responses are capped near 9 kB, so the answer starts with the next index to ask
--         for, or -1 when complete: "<next index>|<entries>".
--
-- 2. A match without Python (batch.py): the mod option reinforcement_learning_scenario holds the same spawns and
--    states, "<team>,<unit>,<x>,<z>/...|<t0_fire>,<t0_move>,...", applied once at SETUP_FRAME. Spawns are separated
--    by "/" there because ";" ends a value in the start script.
--
-- Each reset spawns the new units (x, z as fractions of the map) first, then removes every unit that was there
-- before (commanders at the start of a match) and the wrecks of the previous episode: a team is never left without
-- units, so the game's end condition (deathmode killall in matches) only fires when a side is really eliminated.
-- The states array has 2*N values for N allyteams: fire and move per team index, -1 keeps the unit's default.
--
-- Scripted controllers (controllers.py) run here too, in matches and in the environment alike: the mod option
-- reinforcement_learning_controllers lists "<team>,<script>,<parameter>,.../...", and SCRIPTS below runs each
-- script for its team at every game frame.

if not gadgetHandler:IsSyncedCode() then
	return
end

function gadget:GetInfo()
	return {
		name    = "Reinforcement Learning Scenario",
		desc    = "Spawns and resets the units of a reinforcement learning scenario",
		layer   = 0,
		enabled = true,
	}
end

local modOptions = Spring.GetModOptions()

-- Commanders are created at GameStart (game_initial_spawn.lua); the scenario replaces them just after.
local SETUP_FRAME = 1

local currentEpisode  = -1
local episodeFeatures = {}  -- wrecks created during the current episode

local function split(text, separator)
	local parts = {}
	for part in string.gmatch(text, "([^" .. separator .. "]+)") do
		parts[#parts + 1] = part
	end
	return parts
end

local function mapPoint(x, z)
	local wx = math.max(0, math.min(1, x)) * Game.mapSizeX
	local wz = math.max(0, math.min(1, z)) * Game.mapSizeZ
	return wx, Spring.GetGroundHeight(wx, wz), wz
end

local function setStates(unitID, fireState, moveState)
	if fireState >= 0 then
		Spring.GiveOrderToUnit(unitID, CMD.FIRE_STATE, { fireState }, 0)
	end
	if moveState >= 0 then
		Spring.GiveOrderToUnit(unitID, CMD.MOVE_STATE, { moveState }, 0)
	end
end

-- Spawns the units of spawnsText, then removes every unit and wreck of before; returns the new unit IDs.
local function placeScenario(spawnsText, spawnSeparator, statesText)
	-- states: 2*N values — fire and move state for team 0, 1, …, N-1
	local states = split(statesText or "", ",")

	local previousUnits = Spring.GetAllUnits()
	for featureID in pairs(episodeFeatures) do
		if Spring.ValidFeatureID(featureID) then
			Spring.DestroyFeature(featureID)
		end
	end
	episodeFeatures = {}

	local unitIDs = {}
	for _, spawn in ipairs(split(spawnsText or "", spawnSeparator)) do
		local values  = split(spawn, ",")
		local teamID  = tonumber(values[1])
		local name    = values[2]
		local x, y, z = mapPoint(tonumber(values[3]), tonumber(values[4]))
		-- Team 0 faces east (toward increasing x), all others face west.
		-- Units face the centre of the map: east on its western half, west on its eastern half.
		local facing  = (x < Game.mapSizeX / 2) and "east" or "west"
		local uid     = Spring.CreateUnit(name, x, y, z, facing, teamID)
		if uid then
			-- states array is 1-indexed in Lua; team t uses indices 2*t+1 and 2*t+2.
			local fireState = tonumber(states[teamID * 2 + 1]) or -1
			local moveState = tonumber(states[teamID * 2 + 2]) or -1
			setStates(uid, fireState, moveState)
			unitIDs[#unitIDs + 1] = tostring(uid)
		else
			Spring.Echo("[RL Scenario] could not create " .. tostring(name))
			unitIDs[#unitIDs + 1] = "-1"
		end
	end

	for _, unitID in ipairs(previousUnits) do
		if Spring.ValidUnitID(unitID) then
			Spring.DestroyUnit(unitID, false, true)  -- reclaimed: no wreck
		end
	end
	return unitIDs
end

local function reset(message)
	local fields = split(message, "|")
	local unitIDs = placeScenario(fields[2], ";", fields[3])
	currentEpisode = tonumber(fields[1])
	-- Return the spawned unit IDs to the BridgeAI (synchronous response).
	return table.concat(unitIDs, ",")
end

-- AttackNearest: every intervalFrames, each unit attacks the nearest enemy it sees, or moves towards the nearest
-- enemy while fighting (attack-move) when it sees none.
local function attackNearest(teamID, frame, intervalFrames)
	if frame % intervalFrames ~= 0 then
		return
	end
	for _, unitID in ipairs(Spring.GetTeamUnits(teamID)) do
		if not Spring.GetUnitIsDead(unitID) then
			local enemyID = Spring.GetUnitNearestEnemy(unitID, 1.0e9, true)
			if enemyID then
				local commandID, _, _, targetID = Spring.GetUnitCurrentCommand(unitID)
				if commandID ~= CMD.ATTACK or targetID ~= enemyID then
					Spring.GiveOrderToUnit(unitID, CMD.ATTACK, { enemyID }, 0)
				end
			else
				enemyID = Spring.GetUnitNearestEnemy(unitID, 1.0e9, false)
				if enemyID then
					local x, y, z = Spring.GetUnitPosition(enemyID)
					Spring.GiveOrderToUnit(unitID, CMD.FIGHT, { x, y, z }, 0)
				end
			end
		end
	end
end

-- Script name (controllers.Controller.gadget_script) -> function(teamID, frame, parameters...).
local SCRIPTS = {
	attack_nearest = attackNearest,
}

-- { teamID, function, parameters } for every scripted controller of the match.
local scriptedTeams = {}
for _, entry in ipairs(split(modOptions.reinforcement_learning_controllers or "", "/")) do
	local values = split(entry, ",")
	local script = SCRIPTS[values[2]]
	if script then
		local parameters = {}
		for index = 3, #values do
			parameters[#parameters + 1] = tonumber(values[index])
		end
		scriptedTeams[#scriptedTeams + 1] = { teamID = tonumber(values[1]), script = script, parameters = parameters }
	else
		Spring.Echo("[RL Scenario] unknown controller script " .. tostring(values[2]))
	end
end

function gadget:GameFrame(frame)
	local scenario = modOptions.reinforcement_learning_scenario
	if frame == SETUP_FRAME and scenario and scenario ~= "" then
		local fields = split(scenario, "|")
		placeScenario(fields[1], "/", fields[2])
		currentEpisode = 0
	end
	if frame > SETUP_FRAME then
		for _, scripted in ipairs(scriptedTeams) do
			scripted.script(scripted.teamID, frame, unpack(scripted.parameters))
		end
	end
end

local COMMAND_NAMES = {
	[CMD.MOVE]   = "move",
	[CMD.FIGHT]  = "fight",
	[CMD.ATTACK] = "attack",
	[CMD.STOP]   = "stop",
}
local MAXIMUM_RESPONSE_LENGTH = 9000  -- the BridgeAI's response buffer holds 10240 bytes (MAX_RESPONSE_SIZE)

local function nearestEnemyWithin(unitID, wx, wz, radius)
	local best, bestDistance = nil, math.huge
	local teamID = Spring.GetUnitTeam(unitID)
	for _, otherID in ipairs(Spring.GetUnitsInCylinder(wx, wz, radius)) do
		if not Spring.AreTeamsAllied(teamID, Spring.GetUnitTeam(otherID)) and not Spring.GetUnitIsDead(otherID) then
			local ox, _, oz = Spring.GetUnitPosition(otherID)
			local distance = (ox - wx) ^ 2 + (oz - wz) ^ 2
			if distance < bestDistance then
				best, bestDistance = otherID, distance
			end
		end
	end
	return best
end

local function applyActions(message)
	local fields = split(message, "|")
	local snapRadius = tonumber(fields[1]) or 60
	for _, order in ipairs(split(fields[2] or "", ";")) do
		local values = split(order, ",")
		local unitID, command = tonumber(values[1]), values[2]
		if unitID and Spring.ValidUnitID(unitID) and not Spring.GetUnitIsDead(unitID) then
			local wx, wy, wz = mapPoint(tonumber(values[3]) or 0, tonumber(values[4]) or 0)
			if command == "stop" then
				Spring.GiveOrderToUnit(unitID, CMD.STOP, {}, 0)
			elseif command == "move" then
				Spring.GiveOrderToUnit(unitID, CMD.MOVE, { wx, wy, wz }, 0)
			elseif command == "fight" then
				Spring.GiveOrderToUnit(unitID, CMD.FIGHT, { wx, wy, wz }, 0)
			elseif command == "attack" then
				local targetID = nearestEnemyWithin(unitID, wx, wz, snapRadius)
				if targetID then
					Spring.GiveOrderToUnit(unitID, CMD.ATTACK, { targetID }, 0)
				end
			end
		end
	end
	return ""
end

local function currentCommands(message)
	local first = tonumber(message) or 0
	local units = Spring.GetAllUnits()
	local entries, length = {}, 0
	for index = first + 1, #units do
		local unitID = units[index]
		local commandID, _, _, p1, p2, p3 = Spring.GetUnitCurrentCommand(unitID)
		local name, tx, tz = "none", "", ""
		if commandID then
			name = COMMAND_NAMES[commandID] or "other"
			if p3 then
				tx, tz = string.format("%.6f", p1 / Game.mapSizeX), string.format("%.6f", p3 / Game.mapSizeZ)
			elseif p1 and Spring.ValidUnitID(p1) then
				local ux, _, uz = Spring.GetUnitPosition(p1)
				tx, tz = string.format("%.6f", ux / Game.mapSizeX), string.format("%.6f", uz / Game.mapSizeZ)
			end
		end
		local entry = unitID .. "," .. name .. "," .. tx .. "," .. tz
		if length + #entry + 1 > MAXIMUM_RESPONSE_LENGTH then
			return (index - 1) .. "|" .. table.concat(entries, ";")
		end
		entries[#entries + 1] = entry
		length = length + #entry + 1
	end
	return "-1|" .. table.concat(entries, ";")
end

function gadget:RecvSkirmishAIMessage(aiTeam, message)
	if message:sub(1, 9) == "rl_reset|" then
		return reset(message:sub(10))
	elseif message:sub(1, 11) == "rl_actions|" then
		return applyActions(message:sub(12))
	elseif message:sub(1, 12) == "rl_commands|" then
		return currentCommands(message:sub(13))
	end
end

function gadget:FeatureCreated(featureID, allyTeamID)
	if currentEpisode >= 0 then
		episodeFeatures[featureID] = true
	end
end

function gadget:Initialize()
	if modOptions.reinforcement_learning_full_vision == "1" then
		for _, allyTeamID in ipairs(Spring.GetAllyTeamList()) do
			Spring.SetGlobalLos(allyTeamID, true)
		end
	end
end
