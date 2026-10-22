"""The training configurations of configurations/reinforcement_learning/ load (no engine, no training)."""

from pathlib import Path

import pytest

pytest.importorskip("maxalaar_reinforcement_learning")

from maxalaar_reinforcement_learning.configuration.curriculum_configuration import CurriculumConfiguration  # noqa: E402
from maxalaar_reinforcement_learning.configuration.experimentation_configuration import (  # noqa: E402
    ExperimentationConfiguration,
)
from maxalaar_reinforcement_learning.training.build_algorithm_configuration import (  # noqa: E402
    build_algorithm_configuration,
)
from maxalaar_reinforcement_learning.training.curriculum_training import check_stages  # noqa: E402
from maxalaar_reinforcement_learning.utilities.save.get_configuration_class import get_configuration_class  # noqa: E402

CONFIGURATIONS = Path(__file__).resolve().parents[1] / "configurations" / "reinforcement_learning"


@pytest.mark.parametrize("path", sorted(CONFIGURATIONS.glob("*.py")), ids=lambda path: path.stem)
def test_configurations_load(path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(CONFIGURATIONS.parents[1])  # stage paths of a curriculum are relative to the repository root
    configuration = get_configuration_class(str(path))
    if isinstance(configuration, CurriculumConfiguration):
        check_stages(configuration, [get_configuration_class(str(stage)) for stage in configuration.stages])
    else:
        assert isinstance(configuration, ExperimentationConfiguration)
        build_algorithm_configuration(configuration)[1].validate()
