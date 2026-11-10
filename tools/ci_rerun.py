"""Run again the jobs of a CI run that the machines failed, not the code.

    python tools/ci_rerun.py RUN_ID      with gh on the path, GH_TOKEN and GH_REPO set

Two failures of a job say nothing about the code, and running the job again heals them:

- GitHub lost the job's runner. The job ends as failed with a step that never ended, the
  steps after it never started, and it has no log ("The hosted runner lost
  communication with the server"). In the sixty runs up to 10 November 2026 that
  happened to 6 of the 126 jobs on the Intel Mac image and to none of 869 on the others.
- A step that only fetches system packages failed or ran out of its time limit, because
  Ubuntu's package servers were out of step with each other or slow (``PACKAGE_STEPS``).

``.github/workflows/rerun.yml`` calls this after every CI run that failed. The failed
jobs are run again when all of these hold:

- every job that did not pass failed in one of those two ways. Any other failed job (a
  failing test, a test step out of its time limit) is a failure to look at, and so is a
  cancelled one, and so is a job that publishes (``PUBLISHING``): what it had published
  when it stopped is for a person to see;
- the run has had fewer than ``MAX_ATTEMPTS`` attempts, so a job that fails this way
  every time, perhaps by what it runs, stays red;
- no later run exists for the same branch. CI cancels what runs for a ref when another
  run starts for it, and a re-run counts as one: re-running an older run would cancel
  the newer one.

It needs only Python's standard library and GitHub's command line tool.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

#: Attempts a run gets in all: the first one and two re-runs.
MAX_ATTEMPTS = 3
#: The conclusions of a job that passed or had nothing to do.
PASSED = ("success", "skipped", "neutral")
#: The steps of .github/workflows/ci.yml that only fetch system packages, by their names.
#: tests/test_ci.py keeps this list and the workflow in step.
PACKAGE_STEPS = (
    "Software Vulkan driver (Mesa's lavapipe), for the wave solver on a GPU",
    "Install off-screen OpenGL",
    "Install the libraries Qt and its X11 platform load",
    "Fetch the system packages of the image's two stages",
)
#: The jobs of ci.yml that publish (the documentation, a release, the package on PyPI),
#: by their names: never run again from here.
PUBLISHING = (
    "Publish the documentation on GitHub Pages",
    "GitHub release",
    "Publish to PyPI",
)


def lost_its_runner(job: dict) -> bool:
    """Whether ``job``, as GitHub's API lists it, failed with a step that never ended and
    none that failed: the shape of a job whose runner stopped reporting. A step that
    fails or runs out of its time limit is completed with the conclusion "failure", and
    the steps after it are completed as skipped."""
    steps = job.get("steps") or []
    return (
        job.get("conclusion") == "failure"
        and any(step.get("status") != "completed" for step in steps)
        and not any(step.get("conclusion") == "failure" for step in steps)
    )


def packages_not_fetched(job: dict) -> bool:
    """Whether ``job`` failed in a step that only fetches system packages, and in no
    other step."""
    steps = job.get("steps") or []
    failed = [step.get("name") for step in steps if step.get("conclusion") == "failure"]
    return (
        job.get("conclusion") == "failure"
        and all(step.get("status") == "completed" for step in steps)
        and bool(failed)
        and all(name in PACKAGE_STEPS for name in failed)
    )


def _where(job: dict) -> str:
    """The job's name with the step that never ended or that failed."""
    steps = job.get("steps") or []
    stopped = [step.get("name") for step in steps if step.get("status") != "completed"]
    failed = [step.get("name") for step in steps if step.get("conclusion") == "failure"]
    if lost_its_runner(job):
        return f"{job.get('name')} (runner lost in {stopped[0]!r})"
    if failed:
        return f"{job.get('name')} ({failed[0]!r} failed)"
    return f"{job.get('name')} ({job.get('conclusion')})"


def verdict(run: dict, jobs: list[dict], later_runs: int) -> tuple[bool, str]:
    """Whether to run the failed jobs of ``run`` again, and why or why not. ``jobs`` are
    those of its last attempt and ``later_runs`` the number of runs of the same workflow
    and branch started after it."""
    if run.get("status") != "completed" or run.get("conclusion") != "failure":
        ended = run.get("conclusion") or run.get("status")
        return False, f"the run is {ended}, not a failure that has ended"
    not_passed = [job for job in jobs if job.get("conclusion") not in PASSED]
    if not not_passed:
        return False, "no job of the run's last attempt failed"
    theirs = [
        job
        for job in not_passed
        if (lost_its_runner(job) or packages_not_fetched(job)) and job.get("name") not in PUBLISHING
    ]
    ours = [job for job in not_passed if job not in theirs]
    if ours:
        return False, "to look at: " + ", ".join(_where(job) for job in ours)
    failed = ", ".join(_where(job) for job in theirs)
    attempt = run.get("run_attempt", 1)
    if attempt >= MAX_ATTEMPTS:
        return False, f"attempt {attempt} of {MAX_ATTEMPTS} failed the same way: {failed}"
    if later_runs:
        return False, f"a later run exists for this branch ({failed})"
    return True, f"attempt {attempt}: {failed}"


def _gh(*arguments: str) -> str:
    done = subprocess.run(["gh", *arguments], capture_output=True, text=True)
    if done.returncode:
        raise SystemExit(f"gh {' '.join(arguments)}: {done.stderr.strip()}")
    return done.stdout


def _later_runs(run: dict) -> int:
    """The number of runs of the same workflow and branch started after ``run``, whatever
    started them: a push to main and a run started by hand on main are the same ref to
    CI. The newest hundred are read: a later run is among them. Without a branch's name
    (GitHub gives none for some events) any later run counts."""
    fields = ["-F", "per_page=100"]
    if run.get("head_branch"):
        fields += ["-f", f"branch={run['head_branch']}"]
    path = f"repos/{{owner}}/{{repo}}/actions/workflows/{run['workflow_id']}/runs"
    listed = json.loads(_gh("api", "--method", "GET", path, *fields))
    return sum(other["run_number"] > run["run_number"] for other in listed["workflow_runs"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("run_id", type=int, help="the id of the CI run that ended")
    run_id = parser.parse_args(argv).run_id
    run = json.loads(_gh("api", f"repos/{{owner}}/{{repo}}/actions/runs/{run_id}"))
    jobs_path = f"repos/{{owner}}/{{repo}}/actions/runs/{run_id}/jobs?filter=latest&per_page=100"
    pages = json.loads(_gh("api", "--paginate", "--slurp", jobs_path))
    jobs = [job for page in pages for job in page["jobs"]]
    again, why = verdict(run, jobs, _later_runs(run))
    said = f"Run {run_id} is {'run again' if again else 'left as it is'}: {why}."
    print(said)
    if os.environ.get("GITHUB_STEP_SUMMARY"):  # shown on the page of this workflow's run
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as summary:
            summary.write(said + "\n")
    if again:
        _gh("run", "rerun", str(run_id), "--failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
