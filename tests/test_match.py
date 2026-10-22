"""Starting the engine: a process left behind by a killed parent (no engine needed)."""

import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from beyond_all_reason_environment.match import dying_with_parent


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie (finished, not yet reaped by init) is not running anymore.
    with open(f"/proc/{pid}/stat") as stat:
        return stat.read().split(")")[-1].split()[0] != "Z"


@pytest.mark.skipif(shutil.which("setpriv") is None, reason="setpriv (util-linux) is not installed")
def test_a_command_dies_with_its_parent_even_killed():
    # The parent starts a long child as start_engine starts the engine, prints its pid, then waits.
    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(f"""
                import subprocess, sys, time
                from beyond_all_reason_environment.match import dying_with_parent
                child = subprocess.Popen(dying_with_parent([{sys.executable!r}, "-c", "import time; time.sleep(60)"]))
                print(child.pid, flush=True)
                time.sleep(60)
            """),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert parent.stdout is not None
    child = int(parent.stdout.readline())
    assert alive(child)

    parent.send_signal(signal.SIGKILL)  # as Ray kills an env runner when the memory is full
    parent.wait()
    deadline = time.time() + 5
    while alive(child) and time.time() < deadline:
        time.sleep(0.1)
    assert not alive(child), "the child outlived its killed parent"


def test_without_setpriv_the_command_runs_as_is(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert dying_with_parent(["engine", "script.txt"]) == ["engine", "script.txt"]
