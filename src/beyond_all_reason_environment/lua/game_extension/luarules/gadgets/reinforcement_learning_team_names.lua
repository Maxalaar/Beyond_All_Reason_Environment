-- Synced: the names of the teams in the game window, the label of their controller (controllers.py: Agent,
-- AttackNearest...). The mod option reinforcement_learning_team_names lists "<team>,<name>/...".
--
-- Beyond All Reason's AI namer (ai_namer.lua, layer 999) gives every artificial intelligence team a random name, in
-- the game rules parameter "ainame_<team>" that the interface shows (followed by "(AI)"); this gadget, on a later
-- layer, writes its names over them in the same call-in.

if not gadgetHandler:IsSyncedCode() then
	return
end

function gadget:GetInfo()
	return {
		name    = "Reinforcement Learning Team Names",
		desc    = "The names of the teams: their controllers",
		layer   = 1000,
		enabled = true,
	}
end

local teamNames = {} -- team -> name

function gadget:Initialize()
	local option = Spring.GetModOptions().reinforcement_learning_team_names or ""
	for entry in string.gmatch(option, "([^/]+)") do
		local team, name = string.match(entry, "^(%d+),(.+)$")
		if team then
			teamNames[tonumber(team)] = name
		end
	end
	if next(teamNames) == nil then
		gadgetHandler:RemoveGadget(self)
	end
end

function gadget:GameID()
	for teamID, name in pairs(teamNames) do
		Spring.SetGameRulesParam("ainame_" .. teamID, name)
	end
	gadgetHandler:RemoveGadget(self)
end
