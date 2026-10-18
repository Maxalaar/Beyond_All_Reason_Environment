-- Settings of a graphical game (engine_configuration.GraphicsConfiguration): camera keys and closing at the end.
-- Copied into LuaUI/Widgets of the run folder by match.py only when there is a window; match.py writes the two
-- values below.

function widget:GetInfo()
	return {
		name = "Reinforcement Learning Graphics Settings",
		desc = "Camera keys and closing the window at the end of the game, for the reinforcement learning harness",
		layer = 1000,  -- after the game's widgets, so its key bindings are loaded first
		enabled = true,
	}
end

-- Forward, left, back and right; "" keeps the game's keys.
local CAMERA_KEYS = ""
-- Seconds between the end of the game and closing the window; negative leaves it open.
local CLOSE_DELAY_SECONDS = -1

local CAMERA_ACTIONS = { "moveforward", "moveleft", "moveback", "moveright" }

local gameOverTimer = nil

-- "Any+" also matches with modifiers held: with Alt (moverotate), the Spring camera rotates and tilts instead.
local function bindCameraKeys()
	if #CAMERA_KEYS ~= 4 then
		return
	end
	for index, action in ipairs(CAMERA_ACTIONS) do
		local key = CAMERA_KEYS:sub(index, index)
		Spring.SendCommands("unbindkeyset Any+" .. key, "bind Any+" .. key .. " " .. action)
	end
end

function widget:Initialize()
	bindCameraKeys()
end

-- The game's key binding profile may load after this widget: bind again once the game runs.
function widget:GameStart()
	bindCameraKeys()
end

function widget:GameOver()
	if CLOSE_DELAY_SECONDS >= 0 then
		gameOverTimer = Spring.GetTimer()
	end
end

function widget:Update()
	if gameOverTimer and Spring.DiffTimers(Spring.GetTimer(), gameOverTimer) >= CLOSE_DELAY_SECONDS then
		gameOverTimer = nil
		Spring.SendCommands("quitforce")
	end
end
