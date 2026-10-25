-- Settings of a graphical game (engine_configuration.GraphicsConfiguration): camera keys, camera rotation with the
-- right mouse button, closing at the end, a camera over the whole map at the start, and no ground scar of an episode
-- of the reinforcement learning environment left in the next. Copied into LuaUI/Widgets of the run folder by match.py only when there
-- is a window; match.py writes the six values below.

function widget:GetInfo()
	return {
		name = "Reinforcement Learning Graphics Settings",
		desc = "Camera keys and closing the window at the end of the game, for the reinforcement learning harness",
		-- First in the input order (lowest layer), so the right button reaches this widget before the game's ones.
		layer = -1000000,
		enabled = true,
	}
end

-- Forward, left, back and right; "" keeps the game's keys.
local CAMERA_KEYS = ""
-- Seconds between the end of the game and closing the window; negative leaves it open.
local CLOSE_DELAY_SECONDS = -1
-- Holding the right mouse button and moving the mouse rotates the camera (spectators only: they give no orders).
local RIGHT_BUTTON_ROTATION = false
-- Radians per pixel of mouse movement.
local ROTATION_SPEED = 0.003
-- 1: the view follows the mouse (right turns right, up tilts up); -1 inverts that axis.
local HORIZONTAL_ROTATION_SIGN = 1
local VERTICAL_ROTATION_SIGN = 1

local RIGHT_BUTTON = 3
-- The Spring camera's own limits on its tilt (SpringController.cpp): from almost horizontal to straight down.
local MINIMUM_TILT, MAXIMUM_TILT = math.pi * 0.51, math.pi * 0.99

local CAMERA_ACTIONS = { "moveforward", "moveleft", "moveback", "moveright" }

-- The camera at the start: looking at the centre of the map from the south, tilted (between MINIMUM_TILT and
-- MAXIMUM_TILT), from a distance that shows the whole map: DISTANCE_PER_MAP_SIZE times its larger side.
local START_TILT = math.pi * 0.8
local DISTANCE_PER_MAP_SIZE = 1.3

local gameOverTimer = nil
local missionInfoClosed = false
local cameraPlacements = 0
-- Episode of the reinforcement learning environment (game rules parameter of the scenario gadget, at each reset).
local episode = nil

-- The key alone moves the camera, Shift+key moves it faster (movefast is bound to Any+shift by the engine). Other
-- modifiers keep their game bindings; rotation stays the game's, with the mouse.
local function bindCameraKeys()
	if #CAMERA_KEYS ~= 4 then
		return
	end
	for index, action in ipairs(CAMERA_ACTIONS) do
		local key = CAMERA_KEYS:sub(index, index)
		for _, keyset in ipairs({ key, "Shift+" .. key }) do
			Spring.SendCommands("unbindkeyset " .. keyset, "bind " .. keyset .. " " .. action)
		end
	end
end

-- BAR opens its mission briefing for games with scenariooptions (set to avoid commanders, game.py), and hides the
-- rest of the interface while it is open: close it, there is no mission.
local function closeMissionInfo()
	if WG.missioninfo and WG.missioninfo.toggle then
		WG.missioninfo.toggle(false)
		missionInfoClosed = true
	end
end

local function placeCamera()
	if Spring.GetCameraState().name ~= "spring" then
		Spring.SendCommands("viewspring")
	end
	local state = Spring.GetCameraState()
	state.px, state.pz = Game.mapSizeX / 2, Game.mapSizeZ / 2
	state.py = Spring.GetGroundHeight(state.px, state.pz)
	state.dist = DISTANCE_PER_MAP_SIZE * math.max(Game.mapSizeX, Game.mapSizeZ)
	state.rx, state.ry = START_TILT, 0
	Spring.SetCameraState(state, 0)
end

-- Ground scars and tracks are drawn by the game's Decals GL4 widget, which keeps them for minutes: removed at each
-- new episode.
local function removeGroundDecals()
	if not (WG.decalsgl4 and WG.decalsgl4.GetVBOData) then
		return
	end
	local decals = {}
	for _, vbo in ipairs((WG.decalsgl4.GetVBOData())) do
		for instanceID in pairs(vbo.instanceIDtoIndex) do
			decals[#decals + 1] = instanceID
		end
	end
	for _, instanceID in ipairs(decals) do
		WG.decalsgl4.RemoveDecalGL4(instanceID)
	end
end

function widget:GameFrame(frame)
	-- A few frames in, after the game's own camera widgets, which move it at the start; again at frame 30, as it was
	-- once found moved at frame 20 (2026-10-25, cause not found).
	if cameraPlacements < 2 and frame >= (cameraPlacements == 0 and 2 or 30) then
		placeCamera()
		cameraPlacements = cameraPlacements + 1
	end
	local current = Spring.GetGameRulesParam("reinforcement_learning_episode")
	if current ~= episode then
		if episode ~= nil then
			removeGroundDecals()
		end
		episode = current
	end
end

function widget:Initialize()
	bindCameraKeys()
	closeMissionInfo()
	-- Only the Spring camera rotates freely around the point it looks at (the default TA camera looks straight down).
	if RIGHT_BUTTON_ROTATION and Spring.GetCameraState().name ~= "spring" then
		Spring.SendCommands("viewspring")
	end
end

function widget:MousePress(x, y, button)
	-- Returning true takes the following mouse moves until the release, and no order is given.
	return RIGHT_BUTTON_ROTATION and button == RIGHT_BUTTON and Spring.GetSpectatingState()
end

function widget:MouseMove(x, y, dx, dy, button)
	local state = Spring.GetCameraState()
	if state.name ~= "spring" then
		return
	end
	-- Horizontal movement turns around the vertical axis, vertical movement tilts; both around the point looked at.
	-- By default the view follows the mouse: moving it right turns the view right, moving it up tilts the view up.
	state.ry = state.ry + HORIZONTAL_ROTATION_SIGN * dx * ROTATION_SPEED
	local tilt = state.rx + VERTICAL_ROTATION_SIGN * dy * ROTATION_SPEED
	state.rx = math.max(MINIMUM_TILT, math.min(MAXIMUM_TILT, tilt))
	Spring.SetCameraState(state, 0)
end

function widget:MouseRelease(x, y, button)
	return true
end

-- The game's key binding profile may load after this widget: bind again once the game runs.
function widget:GameStart()
	bindCameraKeys()
	closeMissionInfo()
end

function widget:GameOver()
	if CLOSE_DELAY_SECONDS >= 0 then
		gameOverTimer = Spring.GetTimer()
	end
end

function widget:Update()
	-- This widget initialises before the game's Mission Info panel: close it once it exists.
	if not missionInfoClosed then
		closeMissionInfo()
	end
	if gameOverTimer and Spring.DiffTimers(Spring.GetTimer(), gameOverTimer) >= CLOSE_DELAY_SECONDS then
		gameOverTimer = nil
		Spring.SendCommands("quitforce")
	end
end
