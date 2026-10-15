-- Writes the outcome of the match to LuaUI/Config/match_result.json when the game ends,
-- then quits right away when running without graphics.
-- Copied into LuaUI/Widgets of each run folder by match.py; user widgets are allowed for spectators,
-- but may only write inside LuaUI/Config or LuaUI/Widgets (see checkUserWritePath in barwidgets.lua).

function widget:GetInfo()
	return {
		name = "Match Result Reporter",
		desc = "Writes the winner to match_result.json for the reinforcement learning harness",
		layer = 0,
		enabled = true,
	}
end

local RESULT_PATH = "LuaUI/Config/match_result.json"

function widget:GameOver(winningAllyTeams)
	local winners = {}
	for _, allyTeamID in ipairs(winningAllyTeams) do
		winners[#winners + 1] = tostring(allyTeamID)
	end
	local file = io.open(RESULT_PATH, "w")
	if file then
		file:write(string.format(
			'{"winning_ally_teams": [%s], "frame": %d}\n',
			table.concat(winners, ", "),
			Spring.GetGameFrame()
		))
		file:close()
	end
	Spring.Echo("[Match Result Reporter] game over at frame " .. Spring.GetGameFrame())
	-- Without graphics, skip the game's autoquit countdown; with graphics, let the viewer see the end screen.
	if Platform.isHeadless then
		Spring.SendCommands("quitforce")
	end
end
