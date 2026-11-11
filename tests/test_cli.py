"""CLI tests.

Two styles are used on purpose:

* In-process: call ``catkit.cli.main([...])`` directly. Fast, and pytest's
  ``capsys`` fixture captures what was printed.
* Subprocess: run the real commands (``catkit`` and ``python -m catkit``)
  exactly as a user would, and check exit codes and output.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from catkit import __version__
from catkit.cli import EXIT_OK, EXIT_USAGE, main

# ---------------------------------------------------------------- in-process


def test_env_returns_ok_and_reports_versions(capsys):
    assert main(["env"]) == EXIT_OK
    out = capsys.readouterr().out
    assert f"catkit {__version__}" in out
    assert "python 3.11" in out


def test_no_command_prints_help_to_stderr_and_returns_usage_error(capsys):
    assert main([]) == EXIT_USAGE
    captured = capsys.readouterr()
    assert "usage: catkit" in captured.err
    assert captured.out == ""


def test_version_flag_exits_zero(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip() == f"catkit {__version__}"


def test_help_flag_exits_zero_and_lists_subcommands(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "usage: catkit" in out
    assert "env" in out


@pytest.mark.parametrize("argv", [["no-such-command"], ["--no-such-option"], ["env", "extra"]])
def test_invalid_arguments_exit_with_usage_error(argv, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == EXIT_USAGE
    assert "error:" in capsys.readouterr().err


# ---------------------------------------------------------------- subprocess

# The console script lives next to the interpreter running the tests
# (e.g. .venv/bin/catkit). Using that exact path avoids picking up a
# different `catkit` that happens to be on PATH.
CONSOLE_SCRIPT = str(Path(sys.executable).parent / "catkit")

ENTRY_POINTS = {
    "console_script": [CONSOLE_SCRIPT],
    "python_m": [sys.executable, "-m", "catkit"],
}


@pytest.fixture(params=sorted(ENTRY_POINTS))
def entry_point(request):
    return ENTRY_POINTS[request.param]


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60)


def test_entry_point_version(entry_point):
    result = run(entry_point + ["--version"])
    assert result.returncode == 0
    assert result.stdout.strip() == f"catkit {__version__}"


def test_entry_point_help(entry_point):
    result = run(entry_point + ["--help"])
    assert result.returncode == 0
    assert "usage: catkit" in result.stdout


def test_entry_point_env(entry_point):
    result = run(entry_point + ["env"])
    assert result.returncode == 0
    assert f"catkit {__version__}" in result.stdout


def test_entry_point_no_command(entry_point):
    result = run(entry_point)
    assert result.returncode == EXIT_USAGE
    assert "usage: catkit" in result.stderr


def test_entry_point_invalid_command(entry_point):
    result = run(entry_point + ["no-such-command"])
    assert result.returncode == EXIT_USAGE
    assert "invalid choice" in result.stderr
