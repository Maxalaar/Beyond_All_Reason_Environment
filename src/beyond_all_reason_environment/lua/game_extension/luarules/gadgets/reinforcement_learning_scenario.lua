-- Synced side of the reinforcement learning micro environment (see micro_environment.py).
--
-- Receives messages from the bridge widget (Spring.SendLuaRulesMsg), which relays Python:
--   "reinforcement_learning_reset|<episode>|<team>,<unit name>,<x>,<z>;...|<agent fire>,<agent move>,<opponent fire>,<opponent move>"
--      removes every unit of the scenario teams and the wrecks of the previous episode, then spawns the units
--      (x, z are fractions of the map) and sets their fire and move states.
--   "reinforcement_learning_actions|<episode>|<decision frame>|<unit id>,<command>,<x>,<z>;..."
--      given to the units at <decision frame> + action delay, or as soon as they arrive if later.
-- Unsynced code can only reach the simulation through the network, and while the bridge waits for Python the
-- server keeps simulating ahead of the local client: usually less than 30 frames, more when the simulation is
-- fast. The fixed delay makes the timing exact in the usual case; the bridge waits for
-- reinforcement_learning_applied_decision before observing, so an observation always follows the actions of the
-- previous decision.
-- Publishes for the widget (game rules parameters, readable by unsynced code):
--   reinforcement_learning_episode (set last, once the units of the episode exist),
--   reinforcement_learning_units ("<unit id>,<unit id>,..." in spawn order),
--   reinforcement_learning_applied_decision (decision frame of the last actions given to the units),
--   reinforcement_learning_late_actions (decisions whose actions arrived after their apply frame),
--   reinforcement_learning_maximum_lead (largest delay, in frames, between a decision and the arrival of its actions).

if not gadgetHandler:IsSyncedCode() then
	return
end

function gadget:GetInfo()
	return {
		name = "Reinforcement Learning Scenario",
		desc = "Spawns, resets and commands the units of the reinforcement learning micro environment",
		layer = 0,
		enabled = true,
	}
end

local MESSAGE_RESET = "reinforcement_learning_reset|"
local MESSAGE_ACTIONS = "reinforcement_learning_actions|"
local SCENARIO_TEAMS = { 0, 1 }
local modOptions = Spring.GetModOptions()
local ACTION_DELAY = tonumber(modOptions.reinforcement_learning_action_delay) or 32
local ATTACK_SNAP_RADIUS = tonumber(modOptions.reinforcement_learning_attack_snap_radius) or 60

local currentEpisode = -1
local pendingDecisions = {} -- apply frame -> { decisionFrame = ..., actions = { { unitID, command, x, z }, ... } }
local episodeFeatures = {} -- wrecks created during the current episode
local lateActions = 0
local maximumLead = 0

local function split(text, separator)
	local parts = {}
	for part in string.gmatch(text, "([^" .. separator .. "]+)") do
		parts[#parts + 1] = part
	end
	return parts
end

local function mapPoint(x, z)
	local worldX = math.max(0, math.min(1, x)) * Game.mapSizeX
	local worldZ = math.max(0, math.min(1, z)) * Game.mapSizeZ
	return worldX, Spring.GetGroundHeight(worldX, worldZ), worldZ
end

local function setStates(unitID, fireState, moveState)
	Spring.GiveOrderToUnit(unitID, CMD.FIRE_STATE, { fireState }, 0)
	Spring.GiveOrderToUnit(unitID, CMD.MOVE_STATE, { moveState }, 0)
end

local function nearestEnemy(unitID, x, z, radius)
	local teamID = Spring.GetUnitTeam(unitID)
	local best, bestDistance = nil, math.huge
	for _, candidateID in ipairs(Spring.GetUnitsInCylinder(x, z, radius)) do
		local candidateTeam = Spring.GetUnitTeam(candidateID)
		if candidateTeam ~= teamID and candidateTeam ~= Spring.GetGaiaTeamID() then
			local candidateX, _, candidateZ = Spring.GetUnitPosition(candidateID)
			local distance = (candidateX - x) ^ 2 + (candidateZ - z) ^ 2
			if distance < bestDistance then
				best, bestDistance = candidateID, distance
			end
		end
	end
	return best
end

local function applyAction(unitID, command, fractionX, fractionZ)
	if not Spring.ValidUnitID(unitID) or Spring.GetUnitIsDead(unitID) then
		return
	end
	if command == "continue" then
		return
	elseif command == "stop" then
		Spring.GiveOrderToUnit(unitID, CMD.STOP, {}, 0)
		return
	end
	local x, y, z = mapPoint(fractionX, fractionZ)
	if command == "move" then
		Spring.GiveOrderToUnit(unitID, CMD.MOVE, { x, y, z }, 0)
	elseif command == "fight" then
		Spring.GiveOrderToUnit(unitID, CMD.FIGHT, { x, y, z }, 0)
	elseif command == "attack" then
		local targetID = nearestEnemy(unitID, x, z, ATTACK_SNAP_RADIUS)
		if targetID then
			Spring.GiveOrderToUnit(unitID, CMD.ATTACK, { targetID }, 0)
		end
	end
end

local function applyDecision(decision)
	for _, action in ipairs(decision.actions) do
		applyAction(action[1], action[2], action[3], action[4])
	end
	Spring.SetGameRulesParam("reinforcement_learning_applied_decision", decision.decisionFrame)
end

local function reset(message)
	local fields = split(message:sub(#MESSAGE_RESET + 1), "|")
	local episode = tonumber(fields[1])
	local states = split(fields[3] or "2,1,0,0", ",")

	for _, teamID in ipairs(SCENARIO_TEAMS) do
		for _, unitID in ipairs(Spring.GetTeamUnits(teamID)) do
			Spring.DestroyUnit(unitID, false, true) -- reclaimed: leaves no wreck
		end
	end
	for featureID in pairs(episodeFeatures) do
		if Spring.ValidFeatureID(featureID) then
			Spring.DestroyFeature(featureID)
		end
	end
	episodeFeatures = {}
	pendingDecisions = {}

	local unitIDs = {}
	for _, spawn in ipairs(split(fields[2], ";")) do
		local values = split(spawn, ",")
		local teamID, unitName = tonumber(values[1]), values[2]
		local x, y, z = mapPoint(tonumber(values[3]), tonumber(values[4]))
		local facing = teamID == 0 and "east" or "west"
		local unitID = Spring.CreateUnit(unitName, x, y, z, facing, teamID)
		if unitID then
			if teamID == 0 then
				setStates(unitID, tonumber(states[1]), tonumber(states[2]))
			else
				setStates(unitID, tonumber(states[3]), tonumber(states[4]))
			end
			unitIDs[#unitIDs + 1] = tostring(unitID)
		else
			Spring.Echo("[Reinforcement Learning Scenario] could not create " .. unitName)
			unitIDs[#unitIDs + 1] = "-1"
		end
	end

	currentEpisode = episode
	Spring.SetGameRulesParam("reinforcement_learning_units", table.concat(unitIDs, ","))
	Spring.SetGameRulesParam("reinforcement_learning_episode", episode)
end

local function receiveActions(message)
	local fields = split(message:sub(#MESSAGE_ACTIONS + 1), "|")
	if tonumber(fields[1]) ~= currentEpisode then
		return -- actions of a previous episode
	end
	local decision = { decisionFrame = tonumber(fields[2]), actions = {} }
	for _, action in ipairs(split(fields[3] or "", ";")) do
		local values = split(action, ",")
		decision.actions[#decision.actions + 1] = { tonumber(values[1]), values[2], tonumber(values[3]), tonumber(values[4]) }
	end

	-- Messages are handled between frames: the last simulated frame is the current one.
	local frame = Spring.GetGameFrame()
	local lead = frame - decision.decisionFrame
	if lead > maximumLead then
		maximumLead = lead
		Spring.SetGameRulesParam("reinforcement_learning_maximum_lead", maximumLead)
	end
	local applyFrame = decision.decisionFrame + ACTION_DELAY
	if applyFrame > frame then
		pendingDecisions[applyFrame] = decision
	else
		lateActions = lateActions + 1
		Spring.SetGameRulesParam("reinforcement_learning_late_actions", lateActions)
		applyDecision(decision) -- too late for its frame: given right away, before the next frame
	end
end

function gadget:RecvLuaMsg(message, playerID)
	if message:sub(1, #MESSAGE_ACTIONS) == MESSAGE_ACTIONS then
		receiveActions(message)
	elseif message:sub(1, #MESSAGE_RESET) == MESSAGE_RESET then
		reset(message)
	end
end

-- Runs before the widgets' GameFrame of the same frame, so the bridge observes the orders just given.
function gadget:GameFrame(frame)
	local decision = pendingDecisions[frame]
	if decision then
		pendingDecisions[frame] = nil
		applyDecision(decision)
	end
end

function gadget:FeatureCreated(featureID, allyTeamID)
	if currentEpisode >= 0 then
		episodeFeatures[featureID] = true
	end
end

function gadget:Initialize()
	Spring.SetGameRulesParam("reinforcement_learning_episode", -1)
	Spring.SetGameRulesParam("reinforcement_learning_applied_decision", -1)
	Spring.SetGameRulesParam("reinforcement_learning_late_actions", 0)
	Spring.SetGameRulesParam("reinforcement_learning_maximum_lead", 0)
	-- Full vision: units can only be attacked when seen, and observations show every unit anyway.
	if modOptions.reinforcement_learning_full_vision == "1" then
		for _, teamID in ipairs(SCENARIO_TEAMS) do
			local allyTeamID = select(6, Spring.GetTeamInfo(teamID, false))
			Spring.SetGlobalLos(allyTeamID, true)
		end
	end
end
