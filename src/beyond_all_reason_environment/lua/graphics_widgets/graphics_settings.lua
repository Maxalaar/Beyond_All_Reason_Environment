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
	end
end

function widget:Initialize()
	bindCameraKeys()
	closeMissionInfo()
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
	if gameOverTimer and Spring.DiffTimers(Spring.GetTimer(), gameOverTimer) >= CLOSE_DELAY_SECONDS then
		gameOverTimer = nil
		Spring.SendCommands("quitforce")
	end
end
