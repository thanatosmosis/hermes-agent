from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from hermes_cli import governed_review as gr
from hermes_cli import kanban_db as kb


@pytest.fixture
def conn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    connection = kb.connect()
    yield connection
    connection.close()


@pytest.fixture
def git_workspace(tmp_path: Path) -> tuple[Path, str, str]:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    commands = [
        ["git", "init", "-b", "candidate"],
        ["git", "config", "user.email", "test@example.invalid"],
        ["git", "config", "user.name", "Test"],
    ]
    for command in commands:
        subprocess.run(command, cwd=workspace, check=True, capture_output=True)
    (workspace / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=workspace, check=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=workspace, check=True, capture_output=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=workspace, check=True, text=True, capture_output=True
    ).stdout.strip()
    return workspace, head, "candidate"


def make_spec(task_id: str, workspace: Path, head: str, branch: str) -> gr.GovernedReviewSpec:
    return gr.GovernedReviewSpec(
        task_id=task_id,
        workspace=workspace,
        expected_commit=head,
        expected_branch=branch,
        prompt="bounded governed review",
        delivery=gr.GitHubIssueTarget("thanatosmosis/Vacation-Bid-V2", 2),
        session_id="session-test",
        model="test-model",
        metadata={"candidate": True},
    )


def test_preflight_passes_with_exact_pin_profile_skills_dependencies_and_delivery(
    conn, git_workspace
) -> None:
    workspace, head, branch = git_workspace
    parent = kb.create_task(conn, title="parent", assignee="r4k7")
    kb.complete_task(conn, parent, summary="done")
    task_id = kb.create_task(
        conn,
        title="candidate",
        assignee="r4k7",
        parents=[parent],
        skills=["kanban-operations", "software-development-workflows"],
    )
    result = gr.preflight_governed_review(
        conn,
        make_spec(task_id, workspace, head, branch),
        profile_resolver=lambda profile: profile == "r4k7",
        skill_resolver=lambda profile, skill: profile == "r4k7" and skill in {
            "kanban-operations", "software-development-workflows"
        },
        delivery_readiness=lambda target: (target.issue == 2, "stub-ready"),
    )

    assert result.ok
    assert not result.errors
    assert result.evidence["actual_commit"] == head
    assert result.evidence["tracked_or_staged_clean"] is True
    events = gr._events(conn, task_id, "governance_preflight_passed")
    assert len(events) == 1
    assert events[0].payload["dedupe_key"] == result.dedupe_key


def test_preflight_fail_closed_is_single_durable_block_and_scheduler_pause(
    conn, git_workspace
) -> None:
    workspace, head, branch = git_workspace
    task_id = kb.create_task(
        conn, title="candidate", assignee="missing-profile", skills=["missing-skill"]
    )
    (workspace / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    pauses: list[str] = []
    spec = make_spec(task_id, workspace, head, branch)

    for _ in range(2):
        result = gr.preflight_governed_review(
            conn,
            spec,
            profile_resolver=lambda profile: False,
            skill_resolver=lambda profile, skill: False,
            delivery_readiness=lambda target: (False, "no durable sink"),
            pause_scheduler=pauses.append,
        )
        assert not result.ok

    assert kb.get_task(conn, task_id).status == "blocked"
    assert len(gr._events(conn, task_id, "governance_preflight_blocked")) == 1
    assert len(gr._events(conn, task_id, "blocked")) == 1
    assert len(gr._events(conn, task_id, "governance_scheduler_paused")) == 1
    assert len(pauses) == 1
    errors = result.errors
    assert any("actual assignee profile" in error for error in errors)
    assert any("delivery is not ready" in error for error in errors)
    assert any("tracked or staged" in error for error in errors)


def test_review_required_is_terminal_sticky_and_does_not_release_descendant(conn) -> None:
    parent = kb.create_task(conn, title="parent", assignee="worker")
    child = kb.create_task(conn, title="child", assignee="worker", parents=[parent])
    claimed = kb.claim_task(conn, parent)
    assert claimed is not None
    assert gr.hold_for_review(conn, parent, "human sign-off")

    for _ in range(3):
        assert kb.recompute_ready(conn) == 0
    assert kb.get_task(conn, parent).status == "blocked"
    assert kb.get_task(conn, child).status == "todo"
    assert gr.release_one_child(conn, parent, child, actor="reviewer")[0] is False


def test_delivery_retry_is_idempotent_does_not_rerun_product_and_releases_one_child(
    conn, git_workspace
) -> None:
    workspace, head, branch = git_workspace
    parent = kb.create_task(conn, title="parent", assignee="worker")
    child_a = kb.create_task(conn, title="child-a", assignee="worker", parents=[parent])
    child_b = kb.create_task(conn, title="child-b", assignee="worker", parents=[parent])
    assert kb.claim_task(conn, parent)
    spec = make_spec(parent, workspace, head, branch)
    product_calls: list[str] = []
    delivery_calls: list[str] = []

    def product_work():
        product_calls.append("ran")
        return {"tests": "passed", "commit": head}

    def flaky_sender(target, body):
        delivery_calls.append(body)
        if len(delivery_calls) == 1:
            raise RuntimeError("simulated GitHub outage")

    assert not gr.complete_with_issue_delivery(
        conn, spec, summary="candidate complete", product_work=product_work, sender=flaky_sender
    )
    assert kb.get_task(conn, parent).status == "blocked"
    assert len(product_calls) == 1

    assert gr.complete_with_issue_delivery(
        conn, spec, summary="candidate complete", product_work=product_work, sender=flaky_sender
    )
    assert kb.get_task(conn, parent).status == "done"
    assert len(product_calls) == 1
    assert len(delivery_calls) == 2
    assert delivery_calls[0] == delivery_calls[1]
    assert kb.get_task(conn, child_a).status == "todo"
    assert kb.get_task(conn, child_b).status == "todo"

    # A manually held child stays sticky while the reviewer explicitly releases
    # only its sibling.
    conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (child_b,))
    conn.commit()
    assert kb.block_task(conn, child_b, reason="manual hold")
    ok, error = gr.release_one_child(conn, parent, child_a, actor="reviewer")
    assert ok and error is None
    assert kb.get_task(conn, child_a).status == "ready"
    assert kb.get_task(conn, child_b).status == "blocked"
    assert kb.recompute_ready(conn) == 0
    assert kb.get_task(conn, child_b).status == "blocked"

    # Re-entry after success is idempotent: neither product nor delivery reruns.
    assert gr.complete_with_issue_delivery(
        conn, spec, summary="candidate complete", product_work=product_work, sender=flaky_sender
    )
    assert len(product_calls) == 1
    assert len(delivery_calls) == 2


def test_delivery_exhaustion_blocks_and_pauses_without_third_send(conn, git_workspace) -> None:
    workspace, head, branch = git_workspace
    task_id = kb.create_task(conn, title="candidate", assignee="worker")
    assert kb.claim_task(conn, task_id)
    spec = make_spec(task_id, workspace, head, branch)
    product_calls: list[int] = []
    sends: list[int] = []
    pauses: list[str] = []

    def product_work():
        product_calls.append(1)
        return {"state": "preserved"}

    def failing_sender(target, body):
        sends.append(1)
        raise RuntimeError("offline")

    assert not gr.complete_with_issue_delivery(
        conn, spec, summary="done", product_work=product_work, sender=failing_sender
    )
    # Delivery-only retry is allowed from the preserved blocked state.
    assert not gr.complete_with_issue_delivery(
        conn, spec, summary="done", product_work=product_work, sender=failing_sender
    )
    # Third invocation is rejected before send and pauses the scheduler once.
    assert not gr.complete_with_issue_delivery(
        conn,
        spec,
        summary="done",
        product_work=product_work,
        sender=failing_sender,
        pause_scheduler=pauses.append,
    )
    assert len(product_calls) == 1
    assert len(sends) == 2
    assert len(pauses) == 1
    assert kb.get_task(conn, task_id).status == "blocked"


def test_exhaustion_allows_one_preserved_state_recovery_and_rejects_second(conn) -> None:
    task_id = kb.create_task(conn, title="candidate", assignee="worker")
    conn.execute(
        "UPDATE tasks SET status='blocked', consecutive_failures=2, last_failure_error='turns exhausted' "
        "WHERE id=?",
        (task_id,),
    )
    conn.execute(
        "INSERT INTO task_events(task_id, kind, payload, created_at) VALUES (?, 'gave_up', ?, 1)",
        (task_id, '{"error":"turns exhausted"}'),
    )
    conn.commit()

    ok, detail = gr.request_preserved_state_recovery(
        conn, task_id, actor="B1-DO", reason="authorized recovery"
    )
    assert ok, detail
    assert kb.get_task(conn, task_id).status == "ready"
    recovery = gr._events(conn, task_id, "governance_preserved_recovery")
    assert len(recovery) == 1
    assert recovery[0].payload["prior_consecutive_failures"] == 2
    assert recovery[0].payload["prior_last_failure_error"] == "turns exhausted"

    conn.execute("UPDATE tasks SET status='blocked' WHERE id=?", (task_id,))
    conn.commit()
    pauses: list[str] = []
    ok, detail = gr.request_preserved_state_recovery(
        conn,
        task_id,
        actor="B1-DO",
        reason="second recovery",
        pause_scheduler=pauses.append,
    )
    assert not ok
    assert "already used" in detail
    assert kb.get_task(conn, task_id).status == "blocked"
    assert len(pauses) == 1


def test_complete_task_default_release_is_unchanged_and_opt_out_is_explicit(conn) -> None:
    parent_default = kb.create_task(conn, title="default parent")
    child_default = kb.create_task(conn, title="default child", parents=[parent_default])
    assert kb.complete_task(conn, parent_default, summary="done")
    assert kb.get_task(conn, child_default).status == "ready"

    parent_governed = kb.create_task(conn, title="governed parent")
    child_governed = kb.create_task(conn, title="governed child", parents=[parent_governed])
    assert kb.complete_task(
        conn, parent_governed, summary="done", recompute_dependents=False
    )
    assert kb.get_task(conn, child_governed).status == "todo"


def test_product_failure_blocks_and_pauses_without_delivery(conn, git_workspace) -> None:
    workspace, head, branch = git_workspace
    task_id = kb.create_task(conn, title="candidate", assignee="worker")
    assert kb.claim_task(conn, task_id)
    spec = make_spec(task_id, workspace, head, branch)
    sends: list[str] = []
    pauses: list[str] = []

    def product_work():
        raise RuntimeError("deterministic build failure")

    assert not gr.complete_with_issue_delivery(
        conn,
        spec,
        summary="not done",
        product_work=product_work,
        sender=lambda target, body: sends.append(body),
        pause_scheduler=pauses.append,
    )
    task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.status == "blocked"
    assert not sends
    assert len(pauses) == 1
    assert len(gr._events(conn, task_id, "governance_product_failed")) == 1


def test_default_sender_uses_comment_marker_as_remote_dedupe(monkeypatch) -> None:
    target = gr.GitHubIssueTarget("owner/repo", 2)
    marker = "<!-- hermes-governed-review:stable-key -->"
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout='[[{"body":"already delivered\\n' + marker + '"}]]',
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    gr._default_delivery_sender(target, "summary\n\n" + marker)
    assert len(commands) == 1
    assert commands[0][:3] == ["gh", "api", "--paginate"]


def test_hourly_templates_are_dormant_and_bounded() -> None:
    path = (
        Path(__file__).parents[2]
        / "website"
        / "static"
        / "kanban"
        / "governed-review-hourly.yaml"
    )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["enabled"] is False
    assert data["target"] == {
        "repository": "thanatosmosis/Vacation-Bid-V2",
        "issue": 2,
    }
    jobs = {job["name"]: job for job in data["jobs"]}
    worker = jobs["governed-review-worker-hourly"]
    review = jobs["governed-review-b1-review-hourly"]
    assert worker["enabled"] is False
    assert worker["schedule"] == "1 * * * *"
    assert worker["task_max_runtime_seconds"] == 180
    assert review["enabled"] is False
    assert review["schedule"] == "15 * * * *"
