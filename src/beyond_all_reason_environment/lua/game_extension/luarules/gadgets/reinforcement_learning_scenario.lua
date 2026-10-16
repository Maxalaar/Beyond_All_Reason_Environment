-- Synced side of the reinforcement learning micro environment (see micro_environment.py).
--
-- Receives reset requests from the BridgeAI (C Skirmish AI, team 0) via
-- gadget:RecvSkirmishAIMessage. The BridgeAI calls COMMAND_CALL_LUA_RULES which invokes
-- this handler synchronously, in the same frame — no server-lead delay.
--
-- Message format from the BridgeAI:
--   "rl_reset|<episode>|<team>,<unit>,<x>,<z>;...|<agent fire>,<agent move>,<opp fire>,<opp move>"
--      removes every unit of the scenario teams and the wrecks of the previous episode,
--      spawns the new units (x, z as fractions of the map), sets fire and move states.
--      Returns: "<unit_id>,<unit_id>,..." (spawned order, agents first).

if not gadgetHandler:IsSyncedCode() then
	return
end

function gadget:GetInfo()
	return {
		name    = "Reinforcement Learning Scenario",
		desc    = "Spawns and resets units for the reinforcement learning micro environment",
		layer   = 0,
		enabled = true,
	}
end

local SCENARIO_TEAMS = { 0, 1 }
local modOptions     = Spring.GetModOptions()

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
	Spring.GiveOrderToUnit(unitID, CMD.FIRE_STATE, { fireState }, 0)
	Spring.GiveOrderToUnit(unitID, CMD.MOVE_STATE, { moveState }, 0)
end

local function reset(message)
	local fields  = split(message, "|")
	local episode = tonumber(fields[1])
	local states  = split(fields[3] or "2,1,0,0", ",")

	for _, teamID in ipairs(SCENARIO_TEAMS) do
		for _, unitID in ipairs(Spring.GetTeamUnits(teamID)) do
			Spring.DestroyUnit(unitID, false, true)  -- reclaimed: no wreck
		end
	end
	for featureID in pairs(episodeFeatures) do
		if Spring.ValidFeatureID(featureID) then
			Spring.DestroyFeature(featureID)
		end
	end
	episodeFeatures = {}

	local unitIDs = {}
	for _, spawn in ipairs(split(fields[2] or "", ";")) do
		local values  = split(spawn, ",")
		local teamID  = tonumber(values[1])
		local name    = values[2]
		local x, y, z = mapPoint(tonumber(values[3]), tonumber(values[4]))
		local facing  = (teamID == 0) and "east" or "west"
		local uid     = Spring.CreateUnit(name, x, y, z, facing, teamID)
		if uid then
			if teamID == 0 then
				setStates(uid, tonumber(states[1]), tonumber(states[2]))
			else
				setStates(uid, tonumber(states[3]), tonumber(states[4]))
			end
			unitIDs[#unitIDs + 1] = tostring(uid)
		else
			Spring.Echo("[RL Scenario] could not create " .. tostring(name))
			unitIDs[#unitIDs + 1] = "-1"
		end
	end

	currentEpisode = episode
	-- Return the spawned unit IDs to the BridgeAI (synchronous response).
	return table.concat(unitIDs, ",")
end

function gadget:RecvSkirmishAIMessage(aiTeam, message)
	if message:sub(1, 9) == "rl_reset|" then
		return reset(message:sub(10))
	end
end

function gadget:FeatureCreated(featureID, allyTeamID)
	if currentEpisode >= 0 then
		episodeFeatures[featureID] = true
	end
end

function gadget:Initialize()
	if modOptions.reinforcement_learning_full_vision == "1" then
		for _, teamID in ipairs(SCENARIO_TEAMS) do
			local allyTeamID = select(6, Spring.GetTeamInfo(teamID, false))
			Spring.SetGlobalLos(allyTeamID, true)
		end
	end
end
