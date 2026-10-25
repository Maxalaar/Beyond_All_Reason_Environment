-- Unsynced: the colours of the scenario's teams (scenario.TeamConfiguration.color), to tell the sides apart in the
-- game window. The mod option reinforcement_learning_team_colors lists "<team>,<red>,<green>,<blue>/..." (0 to 255),
-- for the teams that have a colour; the others keep the game's own.
--
-- Beyond All Reason's AutoColorPicker (game_autocolors.lua) gives the colours when the game loads, from palettes
-- that can be shuffled; this gadget sets them again at its first update, after AutoColorPicker.

if gadgetHandler:IsSyncedCode() then
	return
end

function gadget:GetInfo()
	return {
		name    = "Reinforcement Learning Team Colors",
		desc    = "The colours of the scenario's teams",
		layer   = 0,
		enabled = true,
	}
end

local teamColors = {} -- team -> { red, green, blue } in [0, 1]

local function setTeamColors()
	for teamID, color in pairs(teamColors) do
		Spring.SetTeamColor(teamID, color[1], color[2], color[3])
	end
end

function gadget:Initialize()
	local option = Spring.GetModOptions().reinforcement_learning_team_colors or ""
	for entry in string.gmatch(option, "([^/]+)") do
		local team, red, green, blue = string.match(entry, "^(%d+),(%d+),(%d+),(%d+)$")
		if team then
			teamColors[tonumber(team)] = { tonumber(red) / 255, tonumber(green) / 255, tonumber(blue) / 255 }
		end
	end
	if next(teamColors) == nil then
		gadgetHandler:RemoveGadget(self)
	end
end

function gadget:Update()
	setTeamColors()
	gadgetHandler:RemoveCallIn("Update")
end

-- Becoming a spectator makes AutoColorPicker apply its colours again.
function gadget:PlayerChanged()
	setTeamColors()
end
