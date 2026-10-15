-- Writes the outcome and the team statistics history to LuaUI/Config/match_result.json:
-- when the game ends (then quits right away when running without graphics),
-- or when the engine shuts down without a winner (maximum game time reached).
-- Copied into LuaUI/Widgets of each run folder by match.py; user widgets are allowed for spectators,
-- but may only write inside LuaUI/Config or LuaUI/Widgets (see checkUserWritePath in barwidgets.lua).

function widget:GetInfo()
	return {
		name = "Match Result Reporter",
		desc = "Writes the winner and team statistics for the reinforcement learning harness",
		layer = 0,
		enabled = true,
	}
end

local RESULT_PATH = "LuaUI/Config/match_result.json"

local resultWritten = false

-- Full engine statistics history of each team: one entry every 16 seconds of game time, with
-- cumulative values (metalProduced, damageDealt, unitsKilled...), plus the units alive at the end.
local function teamStatistics()
	local statistics = {}
	local gaiaTeamID = Spring.GetGaiaTeamID()
	for _, teamID in ipairs(Spring.GetTeamList()) do
		if teamID ~= gaiaTeamID then
			local count = Spring.GetTeamStatsHistory(teamID) or 0
			statistics[tostring(teamID)] = {
				history = count > 0 and Spring.GetTeamStatsHistory(teamID, 1, count) or {},
				unitsAlive = Spring.GetTeamUnitCount(teamID),
			}
		end
	end
	return statistics
end

local function writeResult(gameOver, winningAllyTeams)
	local file = io.open(RESULT_PATH, "w")
	if file then
		file:write(Json.encode({
			game_over = gameOver,
			winning_ally_teams = winningAllyTeams,
			frame = Spring.GetGameFrame(),
			teams = teamStatistics(),
		}))
		file:close()
	end
	resultWritten = true
end

function widget:GameOver(winningAllyTeams)
	writeResult(true, winningAllyTeams)
	Spring.Echo("[Match Result Reporter] game over at frame " .. Spring.GetGameFrame())
	-- Without graphics, skip the game's autoquit countdown; with graphics, let the viewer see the end screen.
	if Platform.isHeadless then
		Spring.SendCommands("quitforce")
	end
end

function widget:Shutdown()
	if not resultWritten and Spring.GetGameFrame() > 0 then
		writeResult(false, {})
	end
end
