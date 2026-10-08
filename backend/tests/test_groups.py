"""Spec 28 AC-001: every test belongs to exactly one of the plan's groups, and each group runs on its own.

    pytest -m backend | rag | tools | security | mcp      one group
    pytest                                                everything except live tests
    pytest -m live                                        only the tests that call real models (cost money)
"""

from pathlib import Path

from conftest import GROUPS, TEST_GROUPS


# Implements: specs/28.md#AC-001
def test_every_test_file_is_assigned_to_a_group():
    files = {path.stem for path in Path(__file__).parent.glob("test_*.py")}

    assert files == set(TEST_GROUPS), "add the new test file to TEST_GROUPS in conftest.py"
    assert {g for g in TEST_GROUPS.values() if g} == set(GROUPS)


# Implements: specs/28.md#AC-001
def test_every_collected_test_has_exactly_one_group(request):
    for item in request.session.items:
        groups = {mark.name for mark in item.iter_markers()} & set(GROUPS)
        assert len(groups) == 1, f"{item.nodeid} has groups {groups or 'none'}"


# Implements: specs/28.md#AC-001, #AC-007
def test_a_normal_run_leaves_out_live_tests(request):
    assert "not live" in " ".join(request.config.getini("addopts"))


# Implements: specs/28.md#AC-001, #AC-007 (found in this phase: `-m rag` replaced the default and ran the live tests)
def test_running_one_group_never_includes_live_tests():
    import subprocess
    import sys

    run = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-m", "rag", "-p", "no:warnings"],
                         capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1], timeout=120)

    collected = [line for line in run.stdout.splitlines() if "::" in line]
    assert collected and not [line for line in collected if "test_live.py" in line]
