"""Abstract base class for all Beyond All Reason environments.

Handles engine process lifecycle, TCP connection to BridgeAI, and the PettingZoo parallel API contract.
Concrete subclasses implement reset(), step(), observation_space() and action_space().
"""

import contextlib
import json
import shutil
import socket
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .environment_configuration import EnvironmentConfiguration

CONNECTION_TIMEOUT_SECONDS = 300


class Environment(ABC):
    """One engine process; episodes are reset inside the running match, without reloading the game."""

    def __init__(self, configuration: EnvironmentConfiguration) -> None:
        self.configuration = configuration
        self.possible_agents: list[str] = []
        self.agents: list[str] = []
        self.run_directory: Path | None = None

        self._process: subprocess.Popen | None = None
        self._server: socket.socket | None = None
        self._connection: socket.socket | None = None
        self._reader = None

    @abstractmethod
    def observation_space(self, agent: str) -> Any: ...

    @abstractmethod
    def action_space(self, agent: str) -> Any: ...

    @abstractmethod
    def reset(self, seed: int | None = None, options: dict | None = None) -> Any: ...

    @abstractmethod
    def step(self, actions: dict) -> Any: ...

    def _send(self, message: dict) -> None:
        assert self._connection is not None
        self._connection.sendall((json.dumps(message, separators=(",", ":")) + "\n").encode())

    def _receive(self) -> dict:
        assert self._reader is not None
        line = self._reader.readline()
        if not line:
            log = self.run_directory / "engine_output.log" if self.run_directory else "the engine log"
            raise RuntimeError(f"The engine closed the connection, see {log}")
        return json.loads(line)

    def close(self) -> None:
        if self._connection is not None:
            with contextlib.suppress(OSError):
                self._send({"type": "quit"})
        if self._process is not None:
            try:
                self._process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        for resource in (self._reader, self._connection, self._server):
            if resource is not None:
                resource.close()
        self._process = self._connection = self._server = self._reader = None
        self.agents = []
        if self.run_directory is not None and not self.configuration.keep_run_directory:
            shutil.rmtree(self.run_directory, ignore_errors=True)

    def __enter__(self) -> "Environment":
        return self

    def __exit__(self, *exception) -> None:
        self.close()
