"""Pawn curriculum on the 1v1 arena (flat 1024×1024 elmos): each stage until 90 % victories, then the next one from
its weights.

    0. one_versus_one_ppo.py                  1 agent against 1 Passive Pawn
    1. five_versus_five_passive_ppo.py        5 agents against 5 Passive Pawns
    2. five_versus_five_hold_and_fire_ppo.py  5 agents against 5 HoldAndFire Pawns (they shoot back)

Usage (from this repository root; the same command resumes the curriculum where it stopped, --new starts a new run):
    python ../Maxalaar_Reinforcement_Learning/scripts/curriculum_training.py \
        --curriculum_configuration_file configurations/reinforcement_learning/pawn_curriculum.py
"""

from maxalaar_reinforcement_learning.configuration.curriculum_configuration import CurriculumConfiguration

pawn_curriculum = CurriculumConfiguration(
    curriculum_name="pawn_curriculum",
    stages=[
        "configurations/reinforcement_learning/one_versus_one_ppo.py",
        "configurations/reinforcement_learning/five_versus_five_passive_ppo.py",
        "configurations/reinforcement_learning/five_versus_five_hold_and_fire_ppo.py",
    ],
)
