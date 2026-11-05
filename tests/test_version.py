"""The version the package reports is the version it was released as."""

import pathlib
import tomllib

import ulai_agents


def test_version_matches_the_release():
    pyproject = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"
    released = tomllib.loads(pyproject.read_text())["project"]["version"]
    assert ulai_agents.__version__ == released
