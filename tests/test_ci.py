"""The tool that runs again the CI jobs the machines failed (tools/ci_rerun.py, called by
.github/workflows/rerun.yml), on jobs shaped as GitHub's API listed real ones of this
project in November 2026, and the workflow files it goes with."""

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
WORKFLOWS = ROOT / ".github" / "workflows"


@pytest.fixture
def tool(monkeypatch):
    if not (TOOLS / "ci_rerun.py").exists():
        pytest.skip("tools/ is in the source checkout only")
    monkeypatch.syspath_prepend(str(TOOLS))
    import ci_rerun

    yield ci_rerun
    sys.modules.pop("ci_rerun", None)


def _job(name: str, conclusion: str, *steps: tuple[str, str, str | None]) -> dict:
    """A job as the API lists it, with its steps as (name, status, conclusion)."""
    return {
        "name": name,
        "status": "completed",
        "conclusion": conclusion,
        "steps": [{"name": n, "status": s, "conclusion": c} for n, s, c in steps],
    }


PACKAGES = "Fetch the system packages of the image's two stages"
#: The runner was lost while the tests ran (run 42744633807).
LOST_IN_THE_TESTS = _job(
    "Test (macos-26-intel, Python 3.10)",
    "failure",
    ("Set up job", "completed", "success"),
    (
        "Software Vulkan driver (Mesa's lavapipe), for the wave solver on a GPU",
        "completed",
        "skipped",
    ),
    ("Build and install", "completed", "success"),
    ("Test", "in_progress", None),
    ("Smoke-test the command line", "pending", None),
    ("Post Run actions/checkout@v6", "pending", None),
)
#: The runner was lost after the last step of the job (run 42697251725).
LOST_AT_THE_END = _job(
    "Editor (macos-26-intel)",
    "failure",
    ("Set up job", "completed", "success"),
    ("Test", "completed", "success"),
    ("Post Run actions/checkout@v6", "completed", "success"),
    ("Complete job", "in_progress", None),
)
#: The tests ran out of the step's time limit (run 42748185588); a failing test looks the same.
OUT_OF_TIME = _job(
    "Native code under ASan and UBSan",
    "failure",
    ("Set up job", "completed", "success"),
    ("Test", "completed", "failure"),
    ("Post Run actions/checkout@v6", "completed", "success"),
    ("Complete job", "completed", "success"),
)
#: Ubuntu's package servers failed the image's packages.
NO_PACKAGES = _job(
    "Container image",
    "failure",
    ("Set up job", "completed", "success"),
    (PACKAGES, "completed", "failure"),
    ("Build the image", "completed", "skipped"),
    ("Complete job", "completed", "success"),
)
#: Cancelled by a later push while its tests ran, and before it started (run 42176682150).
CANCELLED = _job(
    "Test (macos-latest, Python 3.14)",
    "cancelled",
    ("Set up job", "completed", "success"),
    ("Test", "completed", "cancelled"),
)
NEVER_STARTED = _job("Wheel (macos-latest)", "cancelled")
LOST_WHILE_PUBLISHING = _job(
    "Publish to PyPI",
    "failure",
    ("Set up job", "completed", "success"),
    ("Run pypa/gh-action-pypi-publish@release/v1", "in_progress", None),
)
PASSED = _job("Lint", "success", ("Set up job", "completed", "success"))
SKIPPED = _job("Publish to PyPI", "skipped")
EVERY_JOB = (
    LOST_IN_THE_TESTS,
    LOST_AT_THE_END,
    OUT_OF_TIME,
    NO_PACKAGES,
    CANCELLED,
    NEVER_STARTED,
    PASSED,
    SKIPPED,
)


def test_a_lost_runner_is_a_failed_job_with_a_step_that_never_ended(tool) -> None:
    lost = [job["name"] for job in EVERY_JOB if tool.lost_its_runner(job)]
    assert lost == [LOST_IN_THE_TESTS["name"], LOST_AT_THE_END["name"]]


def test_packages_not_fetched_is_a_failure_of_a_package_step_alone(tool) -> None:
    assert [job["name"] for job in EVERY_JOB if tool.packages_not_fetched(job)] == [
        NO_PACKAGES["name"]
    ]
    # With a second failed step that is not a package step, the job is one to look at.
    two = _job("Container image", "failure", (PACKAGES, "completed", "failure"))
    two["steps"].append({"name": "Upload", "status": "completed", "conclusion": "failure"})
    assert not tool.packages_not_fetched(two)


def _run(conclusion: str = "failure", attempt: int = 1) -> dict:
    return {
        "id": 42,
        "run_number": 783,
        "run_attempt": attempt,
        "status": "completed",
        "conclusion": conclusion,
        "event": "pull_request",
        "head_branch": "a-branch",
        "workflow_id": 7,
    }


@pytest.mark.parametrize(
    ("run", "jobs", "later", "again", "said"),
    [
        (_run(), [PASSED, LOST_IN_THE_TESTS, SKIPPED], 0, True, "runner lost in 'Test'"),
        (_run(), [LOST_AT_THE_END, NO_PACKAGES], 0, True, f"{PACKAGES!r} failed"),
        (_run(attempt=2), [PASSED, LOST_IN_THE_TESTS], 0, True, "attempt 2"),
        # The third attempt is the last.
        (_run(attempt=3), [PASSED, LOST_IN_THE_TESTS], 0, False, "attempt 3 of 3"),
        # A failure of the code beside it: nothing is run again.
        (_run(), [LOST_IN_THE_TESTS, OUT_OF_TIME], 0, False, "('Test' failed)"),
        (_run(), [LOST_IN_THE_TESTS, CANCELLED], 0, False, "(cancelled)"),
        (_run(), [LOST_IN_THE_TESTS, NEVER_STARTED], 0, False, "Wheel (macos-latest)"),
        # A later run of the branch would be cancelled by the re-run.
        (_run(), [PASSED, LOST_IN_THE_TESTS], 1, False, "a later run"),
        (_run("cancelled"), [LOST_IN_THE_TESTS, CANCELLED], 0, False, "is cancelled"),
        (_run("success"), [PASSED, SKIPPED], 0, False, "is success"),
        # Someone has started it again already: GitHub lists it as running, with no end.
        (_run() | {"status": "in_progress", "conclusion": None}, [PASSED], 0, False, "in_progress"),
        (_run(), [PASSED, SKIPPED], 0, False, "no job"),
        # A job that publishes is never run again from here, lost runner or not.
        (_run(), [PASSED, LOST_WHILE_PUBLISHING], 0, False, "Publish to PyPI (runner lost"),
    ],
)
def test_the_failed_jobs_are_run_again_only_when_the_machines_failed_them_all(
    tool, run, jobs, later, again, said
) -> None:
    decided, why = tool.verdict(run, jobs, later)
    assert decided is again
    assert said in why


@pytest.mark.parametrize(
    ("jobs", "newer", "again"),
    [([PASSED, LOST_IN_THE_TESTS], 0, True), ([PASSED, LOST_IN_THE_TESTS], 1, False)],
)
def test_the_tool_asks_github_and_runs_the_failed_jobs_again(
    tool, monkeypatch, capsys, tmp_path, jobs, newer, again
) -> None:
    run, asked, summary = _run(), [], tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    def gh(*arguments: str) -> str:
        asked.append(arguments)
        if arguments[0] == "run":
            return ""
        if arguments[-1].endswith("/actions/runs/42"):
            return json.dumps(run)
        if "/actions/runs/42/jobs" in arguments[-1]:
            # Two pages, as --paginate --slurp gives them.
            return json.dumps([{"jobs": jobs[:1]}, {"jobs": jobs[1:]}])
        # Later runs of the branch, whatever started them: CI's group is the ref alone.
        assert "repos/{owner}/{repo}/actions/workflows/7/runs" in arguments
        assert "branch=a-branch" in arguments
        assert not any(argument.startswith("event=") for argument in arguments)
        runs = [{"run_number": 783 + k} for k in range(-2, newer + 1)]
        return json.dumps({"workflow_runs": runs})

    monkeypatch.setattr(tool, "_gh", gh)
    assert tool.main(["42"]) == 0
    assert (("run", "rerun", "42", "--failed") in asked) is again
    said = capsys.readouterr().out
    assert ("is run again" if again else "is left as it is") in said
    assert summary.read_text(encoding="utf-8") == said


def _steps(workflow: dict):
    for job in workflow["jobs"].values():
        yield from job["steps"]


def _only_fetches_packages(step: dict) -> bool:
    commands = [command.strip() for command in step.get("run", "").split("&&")]
    return all(c.startswith(("sudo apt-get ", "docker build --target ")) for c in commands)


def test_the_steps_and_jobs_the_tool_names_are_those_of_ci(tool) -> None:
    """Every step of ci.yml that calls apt-get does nothing else, has a time limit (a
    step that stands still then fails, and is run again) and is in the tool's list by its
    name; every name of the list is such a step; and the jobs that publish are the
    tool's."""
    if not WORKFLOWS.exists():
        pytest.skip("the workflows are in the source checkout only")
    ci = yaml.safe_load((WORKFLOWS / "ci.yml").read_text(encoding="utf-8"))
    fetching = [
        step
        for step in _steps(ci)
        if "apt-get" in step.get("run", "") or "--target" in step.get("run", "")
    ]
    assert sorted(step["name"] for step in fetching) == sorted(tool.PACKAGE_STEPS)
    for step in fetching:
        assert _only_fetches_packages(step), step["name"]
        assert step.get("timeout-minutes", 0) > 0, step["name"]
    # The jobs that publish are those the workflow gives a permission of their own.
    publishing = [job["name"] for job in ci["jobs"].values() if "permissions" in job]
    assert sorted(publishing) == sorted(tool.PUBLISHING)
    # The stages the image's step builds are stages of the Dockerfile.
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    image = next(step for step in fetching if "docker build" in step["run"])
    for word in image["run"].split():
        if word not in ("docker", "build", "--target", ".", "&&"):
            assert f" AS {word}\n" in dockerfile, word


def test_the_rerun_workflow_follows_ci_and_calls_the_tool() -> None:
    if not WORKFLOWS.exists():
        pytest.skip("the workflows are in the source checkout only")
    ci = yaml.safe_load((WORKFLOWS / "ci.yml").read_text(encoding="utf-8"))
    rerun = yaml.safe_load((WORKFLOWS / "rerun.yml").read_text(encoding="utf-8"))
    # YAML 1.1 reads the key "on" as true.
    assert rerun[True]["workflow_run"] == {"workflows": [ci["name"]], "types": ["completed"]}
    assert rerun["permissions"] == {"actions": "write", "contents": "read"}
    (job,) = rerun["jobs"].values()
    assert "tools/ci_rerun.py" in job["steps"][-1]["run"]
    assert (TOOLS / "ci_rerun.py").exists()
