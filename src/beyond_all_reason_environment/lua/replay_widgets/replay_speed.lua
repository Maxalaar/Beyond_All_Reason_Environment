-- Sets the playback speed of a replay. Copied by replay.py, which replaces the value of SPEED.
-- Needed because a recorded match replays its own "setspeed 1000" debug command at frame 1
-- (see debugcommands in start_script.py); this widget overrides it right after.

function widget:GetInfo()
	return {
		name = "Replay Speed",
		desc = "Sets the replay speed chosen in scripts/watch_replay.py",
		layer = 0,
		enabled = true,
	}
end

local SPEED = 1

local applied = false

function widget:GameFrame(frame)
	if not applied and frame >= 2 then
		Spring.SendCommands("setspeed " .. SPEED)
		applied = true
	end
end
