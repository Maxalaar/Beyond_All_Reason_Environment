"""Headless test harness and reinforcement learning environment for Beyond All Reason.

ScenarioConfiguration (the situation) -> GameConfiguration (+ who plays each team) -> RunConfiguration (games without
Python, scripts/run.py) or ReinforcementLearningEnvironmentConfiguration (ReinforcementLearningEnvironment, for the
agents).
"""

from pathlib import Path

from .controllers import (
    Agent,
    AttackNearest,
    BuiltInArtificialIntelligence,
    Controller,
    DoNothing,
    HoldAndFire,
    Passive,
)
from .engine_configuration import EngineConfiguration, GraphicsConfiguration
from .game import GameConfiguration
from .match import MatchResult, run_match
from .reinforcement_learning_environment_configuration import (
    ActionConfiguration,
    ObservationConfiguration,
    ReinforcementLearningEnvironmentConfiguration,
    RewardWeights,
)
from .run import RunConfiguration
from .scenario import ScenarioConfiguration, TeamConfiguration
from .start_script import ArtificialIntelligencePlayer, MatchConfiguration


def results_browser_path() -> Path:
    """The Streamlit script of the web interface (not imported: Streamlit runs it as a page)."""
    return Path(__file__).parent / "results_browser.py"


__all__ = [
    "ActionConfiguration",
    "Agent",
    "ArtificialIntelligencePlayer",
    "AttackNearest",
    "BuiltInArtificialIntelligence",
    "Controller",
    "DoNothing",
    "EngineConfiguration",
    "GameConfiguration",
    "GraphicsConfiguration",
    "HoldAndFire",
    "MatchConfiguration",
    "MatchResult",
    "ObservationConfiguration",
    "Passive",
    "ReinforcementLearningEnvironmentConfiguration",
    "RewardWeights",
    "RunConfiguration",
    "ScenarioConfiguration",
    "TeamConfiguration",
    "results_browser_path",
    "run_match",
]
