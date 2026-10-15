-- Unsynced side of the reinforcement learning micro environment (see micro_environment.py).
--
-- Connects to the Python environment (TCP, one JSON object per line) and, at each decision frame, sends the state
-- of the scenario units and waits for the answer: the simulation is blocked meanwhile, which makes step()
-- synchronous. Answers are relayed to the synced gadget (reinforcement_learning_scenario.lua):
--   {"type": "reset", "episode": n, "spawns": [{"team", "unit", "x", "z"}...], "states": [4 numbers]}
--   {"type": "actions", "actions": [{"unit": id, "command": "move", "x": 0.5, "z": 0.5}...]}
--   {"type": "quit"}
-- Game files can use LuaSocket; Beyond All Reason removes it from user widgets, hence this game extension.

function widget:GetInfo()
	return {
		name = "Reinforcement Learning Bridge",
		desc = "Exchanges observations and actions with the Python environment",
		layer = 0,
		enabled = true,
	}
end

local modOptions = Spring.GetModOptions()
local DECISION_INTERVAL = tonumber(modOptions.reinforcement_learning_decision_interval) or 32
-- Before this frame Beyond All Reason is still spawning commanders and setting the game up.
local FIRST_CONTACT_FRAME = 30
local SCENARIO_TEAMS = { 0, 1 }

local COMMAND_NAMES = {
	[CMD.MOVE] = "move",
	[CMD.FIGHT] = "fight",
	[CMD.ATTACK] = "attack",
	[CMD.STOP] = "stop",
}

local client
local state = "connecting" -- "connecting", "waiting" (for an episode to start), "running", "quitting"
local requestedEpisode = -1
local episodeStartFrame = 0
local previousDecisionFrame = -1
local nextDecisionFrame = -1

-- Absent parameters give no value at all (not nil): the parentheses turn that into nil.
local function rulesParameter(name, default)
	local value = (Spring.GetGameRulesParam(name))
	if value == nil then
		return default
	end
	return value
end

local function quit(reason)
	Spring.Echo("[Reinforcement Learning Bridge] quitting: " .. tostring(reason))
	state = "quitting"
	Spring.SendCommands("quitforce")
end

local function send(object)
	local ok, error = client:send(Json.encode(object) .. "\n")
	if not ok then
		quit(error)
	end
end

local function unitTypes(names)
	local types = {}
	for name in pairs(names) do
		local unitDef = UnitDefNames[name]
		if unitDef then
			types[name] = {
				speed = unitDef.speed or 0, -- game units per second
				weapon_range = unitDef.maxWeaponRange or 0,
				sight = unitDef.sightDistance or 0,
				maximum_health = unitDef.health or 0,
			}
		end
	end
	return types
end

local function commandOf(unitID)
	local commands = Spring.GetUnitCommands(unitID, 1)
	local command = commands and commands[1]
	if not command then
		return "none", nil, nil
	end
	local name = COMMAND_NAMES[command.id] or "other"
	local parameters = command.params or {}
	if #parameters == 1 then -- a unit target
		local x, _, z = Spring.GetUnitPosition(parameters[1])
		if x then
			return name, x / Game.mapSizeX, z / Game.mapSizeZ
		end
		return name, nil, nil
	elseif #parameters >= 3 then
		return name, parameters[1] / Game.mapSizeX, parameters[3] / Game.mapSizeZ
	end
	return name, nil, nil
end

local function observation(frame)
	local units, names = {}, {}
	for _, teamID in ipairs(SCENARIO_TEAMS) do
		for _, unitID in ipairs(Spring.GetTeamUnits(teamID)) do
			-- Destroyed units stay listed until the next frame (for example right after a reset).
			if not Spring.GetUnitIsDead(unitID) then
			local x, _, z = Spring.GetUnitPosition(unitID)
			local velocityX, _, velocityZ = Spring.GetUnitVelocity(unitID)
			local health, maximumHealth = Spring.GetUnitHealth(unitID)
			local name = UnitDefs[Spring.GetUnitDefID(unitID)].name
			local command, targetX, targetZ = commandOf(unitID)
			names[name] = true
			units[#units + 1] = {
				id = unitID,
				team = teamID,
				unit = name,
				x = x / Game.mapSizeX,
				z = z / Game.mapSizeZ,
				-- game units per frame
				velocity_x = velocityX,
				velocity_z = velocityZ,
				health = health,
				maximum_health = maximumHealth,
				command = command,
				target_x = targetX,
				target_z = targetZ,
			}
			end
		end
	end
	return {
		type = "observation",
		episode = requestedEpisode,
		frame = frame,
		episode_frame = frame - episodeStartFrame,
		-- DECISION_INTERVAL unless the previous actions reached the simulation late (see the gadget).
		frames_since_previous_decision = previousDecisionFrame >= 0 and frame - previousDecisionFrame or 0,
		spawned_units = rulesParameter("reinforcement_learning_units", ""),
		late_actions = rulesParameter("reinforcement_learning_late_actions", 0),
		maximum_lead = rulesParameter("reinforcement_learning_maximum_lead", 0),
		units = units,
		unit_types = unitTypes(names),
	}
end

-- Waits for Python's answer (the simulation is blocked meanwhile) and relays it to the gadget.
local function receiveAndRelay(decisionFrame)
	local line, error = client:receive("*l")
	if not line then
		quit(error)
		return
	end
	local message = Json.decode(line)
	if message.type == "actions" then
		local parts = {}
		for _, action in ipairs(message.actions or {}) do
			parts[#parts + 1] = string.format("%d,%s,%.6f,%.6f", action.unit, action.command, action.x or 0, action.z or 0)
		end
		Spring.SendLuaRulesMsg(string.format(
			"reinforcement_learning_actions|%d|%d|%s", requestedEpisode, decisionFrame, table.concat(parts, ";")
		))
		previousDecisionFrame = decisionFrame
		nextDecisionFrame = decisionFrame + DECISION_INTERVAL
		state = "running"
	elseif message.type == "reset" then
		local spawns = {}
		for _, spawn in ipairs(message.spawns) do
			spawns[#spawns + 1] = string.format("%d,%s,%.6f,%.6f", spawn.team, spawn.unit, spawn.x, spawn.z)
		end
		requestedEpisode = message.episode
		previousDecisionFrame = -1
		Spring.SendLuaRulesMsg(string.format(
			"reinforcement_learning_reset|%d|%s|%s", message.episode, table.concat(spawns, ";"), table.concat(message.states, ",")
		))
		state = "waiting"
	elseif message.type == "quit" then
		quit("asked by Python")
	else
		quit("unknown message " .. tostring(message.type))
	end
end

function widget:Initialize()
	client = socket.tcp()
	client:settimeout(300)
	local ok, error = client:connect("127.0.0.1", tonumber(modOptions.reinforcement_learning_port))
	if not ok then
		Spring.Echo("[Reinforcement Learning Bridge] cannot connect to Python: " .. tostring(error))
		client = nil
	end
end

function widget:GameFrame(frame)
	if not client or state == "quitting" then
		return
	end
	if state == "connecting" then
		if frame >= FIRST_CONTACT_FRAME then
			send({ type = "ready", frame = frame, map = { size_x = Game.mapSizeX, size_z = Game.mapSizeZ, name = Game.mapName } })
			receiveAndRelay(frame)
		end
	elseif state == "waiting" then
		if rulesParameter("reinforcement_learning_episode", -1) == requestedEpisode then
			-- The gadget spawned the units of the episode just before this frame.
			episodeStartFrame = frame
			send(observation(frame))
			receiveAndRelay(frame)
		end
	elseif
		state == "running"
		and frame >= nextDecisionFrame
		-- Observe only once the actions of the previous decision are given, so the units show them.
		and rulesParameter("reinforcement_learning_applied_decision", -1) == previousDecisionFrame
	then
		send(observation(frame))
		receiveAndRelay(frame)
	end
end

function widget:Shutdown()
	if client then
		client:close()
	end
end
