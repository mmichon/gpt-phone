"""The repo must contain everything the phone runs: a .gitignore rule once silently
kept phone/roles.py out of every commit while deploys (which copy the working
tree) kept working."""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def tracked():
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return set(out.splitlines())


def test_every_source_file_is_tracked():
    files = tracked()
    sources = [p.relative_to(ROOT).as_posix() for d in ("phone", "deploy") for p in (ROOT / d).rglob("*")
               if p.is_file() and "__pycache__" not in p.parts]
    missing = [s for s in sources if s not in files]
    assert not missing, f"not in git (check .gitignore): {missing}"


def test_private_files_are_not_tracked():
    files = tracked()
    assert not {"roles.py", "roles.yaml", ".env", "deploy/env"} & files
