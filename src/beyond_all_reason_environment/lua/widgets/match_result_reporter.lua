-- Writes the outcome and final team statistics to LuaUI/Config/match_result.json:
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

-- Last entry of the engine statistics history (metalProduced, damageDealt, unitsKilled...).
-- The engine records an entry every 16 seconds of game time, so values can lag slightly behind.
local function teamStatistics()
	local statistics = {}
	local gaiaTeamID = Spring.GetGaiaTeamID()
	for _, teamID in ipairs(Spring.GetTeamList()) do
		if teamID ~= gaiaTeamID then
			local values = {}
			local count = Spring.GetTeamStatsHistory(teamID)
			if count and count > 0 then
				local history = Spring.GetTeamStatsHistory(teamID, count, count)
				for key, value in pairs(history[1]) do
					values[key] = value
				end
			end
			values.unitsAlive = Spring.GetTeamUnitCount(teamID)
			statistics[tostring(teamID)] = values
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
