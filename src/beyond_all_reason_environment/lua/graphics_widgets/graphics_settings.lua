-- Settings of a graphical game (engine_configuration.GraphicsConfiguration): camera keys, camera rotation with the
-- right mouse button, and closing at the end. Copied into LuaUI/Widgets of the run folder by match.py only when there
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

local gameOverTimer = nil
local missionInfoClosed = false

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
