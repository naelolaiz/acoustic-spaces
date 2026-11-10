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
        "head_repository": {"id": 1},
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


RUN_PATH = "repos/{owner}/{repo}/actions/runs/42"
RERUN = ("run", "rerun", "42", "--failed")


def _github(tool, run: dict, jobs: list[dict], others: list[dict], refused: dict | None = None):
    """A stand-in for the tool's ``_gh`` and the list of what it was asked. It answers the
    four requests the tool makes, each only when made exactly as GitHub's command line
    tool takes it. With ``refused`` it refuses to run the jobs again, as GitHub does for
    a run that is running, and lists the run with those fields from then on."""
    asked: list[tuple[str, ...]] = []
    listed = dict(run)

    def gh(*arguments: str) -> str:
        asked.append(arguments)
        if arguments == RERUN:
            if refused is None:
                return ""
            listed.update(refused)
            raise tool.GhError("gh run rerun 42 --failed: run 42 cannot be rerun")
        if arguments == ("api", RUN_PATH):
            return json.dumps(listed)
        if arguments == (
            "api",
            "--paginate",
            "--slurp",
            f"{RUN_PATH}/jobs?filter=latest&per_page=100",
        ):
            # Two pages, as --paginate --slurp gives them.
            return json.dumps([{"jobs": jobs[:1]}, {"jobs": jobs[1:]}])
        # The later runs: those of the run's branch, whatever started them.
        branch = ("-f", f"branch={run['head_branch']}") if run["head_branch"] else ()
        path = "repos/{owner}/{repo}/actions/workflows/7/runs"
        assert arguments == ("api", "--method", "GET", path, "-F", "per_page=100", *branch)
        return json.dumps({"workflow_runs": others})

    return gh, asked


def _other(number: int, event: str = "pull_request", repository: int = 1) -> dict:
    """Another run of CI as the API lists it, with the head branch of ``_run()``."""
    return {
        "run_number": number,
        "event": event,
        "head_branch": "a-branch",
        "head_repository": {"id": repository},
    }


@pytest.mark.parametrize(
    ("others", "again"),
    [([_other(781), _other(782), _other(783)], True), ([_other(783), _other(784)], False)],
)
def test_the_tool_asks_github_and_runs_the_failed_jobs_again(
    tool, monkeypatch, capsys, tmp_path, others, again
) -> None:
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    gh, asked = _github(tool, _run(), [PASSED, LOST_IN_THE_TESTS], others)
    monkeypatch.setattr(tool, "_gh", gh)
    assert tool.main(["42"]) == 0
    assert (RERUN in asked) is again
    said = capsys.readouterr().out
    assert ("is run again" if again else "is left as it is") in said
    assert summary.read_text(encoding="utf-8") == said


PUSH = _run() | {"event": "push"}
FROM_A_FORK = _run() | {"head_repository": {"id": 2}}


@pytest.mark.parametrize(
    ("run", "others", "later"),
    [
        # Runs for the same pull request: the later ones count, however many.
        (_run(), [_other(782), _other(783), _other(784), _other(785)], 2),
        # A pull request is not the branch or the tag its head is called as, nor a pull
        # request from the branch of that name in another repository.
        (
            _run(),
            [_other(784, "push"), _other(785, "workflow_dispatch"), _other(786, repository=2)],
            0,
        ),
        (FROM_A_FORK, [_other(784, "push"), _other(785), _other(786, repository=2)], 1),
        # A branch is the same ref to a push and to a run started by hand, and not that
        # of a pull request from a branch of its name.
        (PUSH, [_other(784, "workflow_dispatch"), _other(785, "push"), _other(786)], 2),
        (PUSH, [_other(784, repository=2)], 0),
        # Without the branch's name every later run counts.
        (_run() | {"head_branch": None}, [_other(783), _other(784, "push", 2)], 1),
    ],
)
def test_a_later_run_counts_when_ci_would_cancel_it_for_the_rerun(
    tool, monkeypatch, run, others, later
) -> None:
    gh, _ = _github(tool, run, [], others)
    monkeypatch.setattr(tool, "_gh", gh)
    assert tool._later_runs(run) == later


@pytest.mark.parametrize(
    "now",
    [
        # Running again, by its status or by the number of its attempt, or ended again
        # already: someone started it since it was read.
        {"status": "queued", "conclusion": None, "run_attempt": 2},
        {"status": "in_progress", "conclusion": None},
        {"run_attempt": 2},
    ],
)
def test_a_run_someone_started_again_meanwhile_is_left_to_them(
    tool, monkeypatch, capsys, tmp_path, now
) -> None:
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    gh, asked = _github(tool, _run(), [PASSED, LOST_IN_THE_TESTS], [], refused=now)
    monkeypatch.setattr(tool, "_gh", gh)
    assert tool.main(["42"]) == 0
    assert asked[-2:] == [RERUN, ("api", RUN_PATH)]
    said = capsys.readouterr().out
    assert "is left as it is: someone started it again first" in said
    assert summary.read_text(encoding="utf-8") == said


def test_a_refusal_of_github_for_another_reason_is_an_error(tool, monkeypatch, tmp_path) -> None:
    """The run is as it was read, and GitHub would not run its jobs again: the step
    fails with what GitHub said, and nothing false is written."""
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    gh, _ = _github(tool, _run(), [PASSED, LOST_IN_THE_TESTS], [], refused={})
    monkeypatch.setattr(tool, "_gh", gh)
    with pytest.raises(tool.GhError, match="cannot be rerun"):
        tool.main(["42"])
    assert not summary.exists()


def test_an_error_of_the_command_line_tool_is_raised_with_what_it_said(tool, monkeypatch) -> None:
    def run(command, **_):
        assert command == ["gh", "api", "a/path"]
        return tool.subprocess.CompletedProcess(command, 1, stdout="", stderr="HTTP 403\n")

    monkeypatch.setattr(tool.subprocess, "run", run)
    with pytest.raises(tool.GhError, match="^gh api a/path: HTTP 403$"):
        tool._gh("api", "a/path")


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
