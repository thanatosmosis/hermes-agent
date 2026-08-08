from __future__ import annotations

import json
import subprocess
import threading
from dataclasses import replace
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
    subprocess.run(
        ["git", "commit", "-m", "baseline"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    return workspace, head, "candidate"


def make_spec(
    task_id: str, workspace: Path, head: str, branch: str
) -> gr.GovernedReviewSpec:
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


def delivery_ready(target: gr.GitHubIssueTarget) -> tuple[bool, str]:
    return True, f"comment-ready:{target.repository}#{target.issue}"


def require_task(conn, task_id: str) -> kb.Task:
    task = kb.get_task(conn, task_id)
    assert task is not None
    return task


def pass_preflight(conn, spec: gr.GovernedReviewSpec) -> gr.PreflightResult:
    result = gr.preflight_governed_review(
        conn,
        spec,
        profile_resolver=lambda profile: bool(profile),
        skill_resolver=lambda profile, skill: True,
        delivery_readiness=delivery_ready,
    )
    assert result.ok, result.errors
    return result


def complete_ready(conn, spec: gr.GovernedReviewSpec, **kwargs) -> bool:
    kwargs.setdefault("profile_resolver", lambda profile: bool(profile))
    kwargs.setdefault("skill_resolver", lambda profile, skill: True)
    return gr.complete_with_issue_delivery(
        conn,
        spec,
        delivery_readiness=delivery_ready,
        **kwargs,
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
        skill_resolver=lambda profile, skill: (
            profile == "r4k7"
            and skill in {"kanban-operations", "software-development-workflows"}
        ),
        delivery_readiness=lambda target: (target.issue == 2, "stub-ready"),
    )

    assert result.ok
    assert not result.errors
    assert result.evidence["actual_commit"] == head
    assert result.evidence["tracked_or_staged_clean"] is True
    events = gr._events(conn, task_id, "governance_preflight_passed")
    assert len(events) == 1
    assert events[0].payload["dedupe_key"] == result.dedupe_key


def test_identity_change_requires_and_accepts_fresh_preflight(
    conn, git_workspace
) -> None:
    workspace, head, branch = git_workspace

    stale_task = kb.create_task(conn, title="stale", assignee="worker")
    assert kb.claim_task(conn, stale_task)
    stale_spec = make_spec(stale_task, workspace, head, branch)
    pass_preflight(conn, stale_spec)
    conn.execute("UPDATE tasks SET assignee='reviewer' WHERE id=?", (stale_task,))
    product_calls: list[str] = []

    assert not complete_ready(
        conn,
        stale_spec,
        summary="must not run",
        product_work=lambda: product_calls.append("stale") or {},
        sender=lambda target, body: None,
        profile_resolver=lambda profile: profile in {"worker", "reviewer"},
    )
    assert product_calls == []

    refreshed_task = kb.create_task(conn, title="refreshed", assignee="worker")
    assert kb.claim_task(conn, refreshed_task)
    refreshed_spec = make_spec(refreshed_task, workspace, head, branch)
    pass_preflight(conn, refreshed_spec)
    conn.execute("UPDATE tasks SET assignee='reviewer' WHERE id=?", (refreshed_task,))
    refreshed = gr.preflight_governed_review(
        conn,
        refreshed_spec,
        profile_resolver=lambda profile: profile == "reviewer",
        skill_resolver=lambda profile, skill: True,
        delivery_readiness=delivery_ready,
    )

    assert refreshed.ok
    passes = gr._events(conn, refreshed_task, "governance_preflight_passed")
    assert len(passes) == 2
    refreshed_payload = passes[-1].payload
    assert refreshed_payload is not None
    assert refreshed_payload["actual_profile"] == "reviewer"
    assert complete_ready(
        conn,
        refreshed_spec,
        summary="fresh identity",
        product_work=lambda: product_calls.append("fresh") or {"commit": head},
        sender=lambda target, body: None,
        profile_resolver=lambda profile: profile == "reviewer",
    )
    assert product_calls == ["fresh"]


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

    assert require_task(conn, task_id).status == "blocked"
    assert len(gr._events(conn, task_id, "governance_preflight_blocked")) == 1
    assert len(gr._events(conn, task_id, "blocked")) == 1
    assert len(gr._events(conn, task_id, "governance_scheduler_paused")) == 1
    assert len(pauses) == 1
    errors = result.errors
    assert any("actual assignee profile" in error for error in errors)
    assert any("delivery is not ready" in error for error in errors)
    assert any("tracked or staged" in error for error in errors)


def test_preflight_rejects_untracked_material(conn, git_workspace) -> None:
    workspace, head, branch = git_workspace
    task_id = kb.create_task(conn, title="candidate", assignee="worker")
    (workspace / "untracked.txt").write_text("not immutable\n", encoding="utf-8")

    result = gr.preflight_governed_review(
        conn,
        make_spec(task_id, workspace, head, branch),
        profile_resolver=lambda profile: True,
        delivery_readiness=delivery_ready,
    )

    assert not result.ok
    assert result.evidence["tracked_or_staged_clean"] is False
    assert any("worktree state is not clean" in error for error in result.errors)


def test_preflight_nonexistent_task_returns_controlled_result_without_audit_write(
    conn, git_workspace
) -> None:
    workspace, head, branch = git_workspace
    probes: list[str] = []

    result = gr.preflight_governed_review(
        conn,
        make_spec("t_doesnotexist", workspace, head, branch),
        profile_resolver=lambda profile: probes.append(profile) or True,
        delivery_readiness=lambda target: (
            probes.append(target.repository) or (True, "ready")
        ),
    )

    assert not result.ok
    assert result.errors == ("task t_doesnotexist does not exist",)
    assert probes == []
    assert conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0] == 0


def test_review_required_is_terminal_sticky_and_does_not_release_descendant(
    conn,
) -> None:
    parent = kb.create_task(conn, title="parent", assignee="worker")
    child = kb.create_task(conn, title="child", assignee="worker", parents=[parent])
    claimed = kb.claim_task(conn, parent)
    assert claimed is not None
    assert gr.hold_for_review(conn, parent, "human sign-off")

    for _ in range(3):
        assert kb.recompute_ready(conn) == 0
    assert require_task(conn, parent).status == "blocked"
    assert require_task(conn, child).status == "todo"
    assert gr.release_one_child(conn, parent, child, actor="reviewer")[0] is False


def test_product_work_requires_current_matching_preflight_for_same_task_and_spec(
    conn, git_workspace
) -> None:
    workspace, head, branch = git_workspace
    preflight_task = kb.create_task(conn, title="preflight", assignee="worker")
    target_task = kb.create_task(conn, title="target", assignee="worker")
    assert kb.claim_task(conn, preflight_task)
    assert kb.claim_task(conn, target_task)
    pass_preflight(conn, make_spec(preflight_task, workspace, head, branch))
    product_calls: list[str] = []

    assert not complete_ready(
        conn,
        make_spec(target_task, workspace, head, branch),
        summary="must not run",
        product_work=lambda: product_calls.append("ran") or {},
        sender=lambda target, body: None,
    )

    assert product_calls == []
    assert len(gr._events(conn, target_task, "governance_execution_blocked")) == 1


def test_product_work_refuses_review_required_hold_after_successful_preflight(
    conn, git_workspace
) -> None:
    workspace, head, branch = git_workspace
    task_id = kb.create_task(conn, title="candidate", assignee="worker")
    assert kb.claim_task(conn, task_id)
    spec = make_spec(task_id, workspace, head, branch)
    pass_preflight(conn, spec)
    assert gr.hold_for_review(conn, task_id, "human sign-off")
    product_calls: list[str] = []

    assert not complete_ready(
        conn,
        spec,
        summary="must not run",
        product_work=lambda: product_calls.append("ran") or {},
        sender=lambda target, body: None,
    )

    assert product_calls == []
    assert require_task(conn, task_id).status == "blocked"


def test_delivery_only_retry_refuses_changed_immutable_state_without_rerunning_product(
    conn, git_workspace
) -> None:
    workspace, head, branch = git_workspace
    task_id = kb.create_task(conn, title="candidate", assignee="worker")
    assert kb.claim_task(conn, task_id)
    spec = make_spec(task_id, workspace, head, branch)
    pass_preflight(conn, spec)
    product_calls: list[str] = []
    sends: list[str] = []

    def product_work():
        product_calls.append("ran")
        return {"commit": head}

    def sender(target, body):
        sends.append(body)
        raise RuntimeError("offline")

    assert not complete_ready(
        conn, spec, summary="done", product_work=product_work, sender=sender
    )
    (workspace / "untracked-after-product.txt").write_text("drift\n", encoding="utf-8")
    assert not complete_ready(
        conn, spec, summary="done", product_work=product_work, sender=sender
    )
    assert product_calls == ["ran"]
    assert len(sends) == 1


def test_delivery_only_retry_refuses_lost_comment_capability_without_rerunning_product(
    conn, git_workspace
) -> None:
    workspace, head, branch = git_workspace
    task_id = kb.create_task(conn, title="candidate", assignee="worker")
    assert kb.claim_task(conn, task_id)
    spec = make_spec(task_id, workspace, head, branch)
    pass_preflight(conn, spec)
    product_calls: list[str] = []
    sends: list[str] = []

    def product_work():
        product_calls.append("ran")
        return {"commit": head}

    def sender(target, body):
        sends.append(body)
        raise RuntimeError("offline")

    assert not complete_ready(
        conn, spec, summary="done", product_work=product_work, sender=sender
    )
    assert not gr.complete_with_issue_delivery(
        conn,
        spec,
        summary="done",
        product_work=product_work,
        sender=sender,
        delivery_readiness=lambda target: (False, "comment permission revoked"),
    )
    assert product_calls == ["ran"]
    assert len(sends) == 1


def test_delivery_only_retry_refuses_replayed_delivery_hold_without_rerunning_product(
    conn, git_workspace
) -> None:
    workspace, head, branch = git_workspace
    task_id = kb.create_task(conn, title="candidate", assignee="worker")
    assert kb.claim_task(conn, task_id)
    spec = make_spec(task_id, workspace, head, branch)
    pass_preflight(conn, spec)
    product_calls: list[str] = []
    sends: list[str] = []

    def product_work():
        product_calls.append("ran")
        return {"commit": head}

    def sender(target, body):
        sends.append(body)
        raise RuntimeError("offline")

    assert not complete_ready(
        conn, spec, summary="done", product_work=product_work, sender=sender
    )
    assert kb.unblock_task(conn, task_id)
    task = kb.get_task(conn, task_id)
    assert task is not None
    assert kb.block_task(
        conn,
        task_id,
        reason="delivery-failed: offline",
        expected_run_id=task.current_run_id,
    )

    assert not complete_ready(
        conn, spec, summary="done", product_work=product_work, sender=sender
    )
    assert product_calls == ["ran"]
    assert len(sends) == 1


def test_atomic_product_reservation_blocks_concurrent_and_reentrant_callbacks(
    conn, git_workspace
) -> None:
    workspace, head, branch = git_workspace
    task_id = kb.create_task(conn, title="candidate", assignee="worker")
    assert kb.claim_task(conn, task_id)
    spec = make_spec(task_id, workspace, head, branch)
    pass_preflight(conn, spec)
    db_path = Path(conn.execute("PRAGMA database_list").fetchone()[2])
    callback_entered = threading.Event()
    release_callback = threading.Event()
    product_calls: list[str] = []
    reentrant_results: list[bool] = []
    thread_results: list[bool] = []
    thread_errors: list[BaseException] = []

    def invoke() -> None:
        local = kb.connect(db_path)

        def product_work():
            product_calls.append("ran")
            reentrant_results.append(
                complete_ready(
                    local,
                    spec,
                    summary="reentrant",
                    product_work=lambda: {"must": "not run"},
                    sender=lambda target, body: None,
                )
            )
            callback_entered.set()
            assert release_callback.wait(5)
            return {"commit": head}

        try:
            thread_results.append(
                complete_ready(
                    local,
                    spec,
                    summary="done",
                    product_work=product_work,
                    sender=lambda target, body: None,
                )
            )
        except BaseException as exc:
            thread_errors.append(exc)
        finally:
            local.close()

    owner = threading.Thread(target=invoke)
    owner.start()
    assert callback_entered.wait(5)
    contender = threading.Thread(target=invoke)
    contender.start()
    contender.join(5)
    assert not contender.is_alive()
    release_callback.set()
    owner.join(5)

    assert not owner.is_alive()
    assert thread_errors == []
    assert reentrant_results == [False]
    assert sorted(thread_results) == [False, True]
    assert product_calls == ["ran"]
    assert len(gr._events(conn, task_id, "governance_product_reserved")) == 1
    assert len(gr._events(conn, task_id, "governance_product_completed")) == 1


def test_delivery_reservation_is_atomic_under_concurrent_retry(
    conn, git_workspace
) -> None:
    workspace, head, branch = git_workspace
    task_id = kb.create_task(conn, title="candidate", assignee="worker")
    assert kb.claim_task(conn, task_id)
    spec = make_spec(task_id, workspace, head, branch)
    pass_preflight(conn, spec)
    sends: list[str] = []

    def first_failure(target, body):
        sends.append("first")
        raise RuntimeError("offline")

    assert not complete_ready(
        conn,
        spec,
        summary="done",
        product_work=lambda: {"commit": head},
        sender=first_failure,
    )

    db_path = Path(conn.execute("PRAGMA database_list").fetchone()[2])
    retry_entered = threading.Event()
    release_retry = threading.Event()
    results: list[bool] = []
    errors: list[BaseException] = []

    def retry_sender(target, body):
        sends.append("retry")
        retry_entered.set()
        assert release_retry.wait(5)

    def invoke_retry() -> None:
        local = kb.connect(db_path)
        try:
            results.append(
                complete_ready(
                    local,
                    spec,
                    summary="done",
                    product_work=lambda: {"must": "not rerun"},
                    sender=retry_sender,
                )
            )
        except BaseException as exc:
            errors.append(exc)
        finally:
            local.close()

    owner = threading.Thread(target=invoke_retry)
    owner.start()
    assert retry_entered.wait(5)
    contender = threading.Thread(target=invoke_retry)
    contender.start()
    contender.join(5)
    assert not contender.is_alive()
    release_retry.set()
    owner.join(5)

    assert not owner.is_alive()
    assert errors == []
    assert sorted(results) == [False, True]
    assert sends == ["first", "retry"]
    reservations = gr._events(conn, task_id, "governance_delivery_reserved")
    assert [event.payload["attempt"] for event in reservations] == [1, 2]
    assert (
        gr._events(conn, task_id, "governance_delivery_reconciliation_required") == []
    )
    task = kb.get_task(conn, task_id)
    assert task is not None and task.status == "done"


def test_interrupted_delivery_is_never_automatically_resent(
    conn, git_workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, head, branch = git_workspace
    task_id = kb.create_task(conn, title="candidate", assignee="worker")
    assert kb.claim_task(conn, task_id)
    spec = make_spec(task_id, workspace, head, branch)
    pass_preflight(conn, spec)
    sends: list[str] = []
    now = [1_000]
    monkeypatch.setattr(gr.time, "time", lambda: now[0])

    class SimulatedInterruption(BaseException):
        pass

    def interrupted_sender(target, body):
        sends.append(body)
        raise SimulatedInterruption()

    with pytest.raises(SimulatedInterruption):
        complete_ready(
            conn,
            spec,
            summary="done",
            product_work=lambda: {"commit": head},
            sender=interrupted_sender,
        )

    # The unexpired owner is reported in-flight: no send, pause, or barrier.
    assert not complete_ready(
        conn,
        spec,
        summary="done",
        product_work=lambda: {"must": "not rerun"},
        sender=interrupted_sender,
    )
    assert (
        gr._events(conn, task_id, "governance_delivery_reconciliation_required") == []
    )
    assert gr._events(conn, task_id, "governance_scheduler_pause_reserved") == []

    now[0] += gr.DELIVERY_RESERVATION_LEASE_SECONDS + 1
    pauses: list[str] = []
    assert not complete_ready(
        conn,
        spec,
        summary="done",
        product_work=lambda: {"must": "not rerun"},
        sender=interrupted_sender,
        pause_scheduler=pauses.append,
    )
    assert len(sends) == 1
    assert len(gr._events(conn, task_id, "governance_delivery_reserved")) == 1
    assert gr._events(conn, task_id, "governance_delivery_failed") == []
    assert (
        len(gr._events(conn, task_id, "governance_delivery_reconciliation_required"))
        == 1
    )
    assert len(pauses) == 1
    task = kb.get_task(conn, task_id)
    assert task is not None and task.status == "blocked"

    dedupe = gr._dedupe_key(spec, gr._sha256(spec.prompt))
    ok, detail = gr.reconcile_delivery_attempt(
        conn,
        task_id,
        dedupe=dedupe,
        attempt=1,
        actor="reviewer",
        actual_outcome="delivered",
        evidence="verified the remote issue comment marker",
        target=str(spec.delivery),
    )
    assert ok and detail == "delivery reconciled"
    assert complete_ready(
        conn,
        spec,
        summary="done",
        product_work=lambda: {"must": "not rerun"},
        sender=interrupted_sender,
    )
    assert len(sends) == 1
    assert len(gr._events(conn, task_id, "governance_delivery_reconciled")) == 1
    task = kb.get_task(conn, task_id)
    assert task is not None and task.status == "done"


def test_execution_and_delivery_retry_freshly_resolve_profile_and_skills(
    conn, git_workspace
) -> None:
    workspace, head, branch = git_workspace
    task_id = kb.create_task(
        conn, title="candidate", assignee="worker", skills=["required-skill"]
    )
    assert kb.claim_task(conn, task_id)
    spec = make_spec(task_id, workspace, head, branch)
    pass_preflight(conn, spec)
    profile_calls: list[str] = []
    skill_calls: list[tuple[str, str]] = []
    skill_available = True
    sends: list[str] = []

    def profile_resolver(profile: str) -> bool:
        profile_calls.append(profile)
        return True

    def skill_resolver(profile: str, skill: str) -> bool:
        skill_calls.append((profile, skill))
        return skill_available

    def failing_sender(target, body):
        sends.append(body)
        raise RuntimeError("offline")

    assert not complete_ready(
        conn,
        spec,
        summary="done",
        product_work=lambda: {"commit": head},
        sender=failing_sender,
        profile_resolver=profile_resolver,
        skill_resolver=skill_resolver,
    )
    skill_available = False
    assert not complete_ready(
        conn,
        spec,
        summary="done",
        product_work=lambda: {"must": "not rerun"},
        sender=failing_sender,
        profile_resolver=profile_resolver,
        skill_resolver=skill_resolver,
    )

    assert profile_calls == ["worker", "worker"]
    assert skill_calls == [
        ("worker", "required-skill"),
        ("worker", "required-skill"),
    ]
    assert len(sends) == 1


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("rollback_path", "revert the candidate and verify it"),
        ("rollback_path", {}),
        (
            "rollback_path",
            {
                "steps": [
                    {
                        "order": 1,
                        "action": "revert",
                        "target": "candidate commit",
                        "max_attempts": 1,
                    }
                ],
                "verification": {"condition": "candidate is absent", "evidence": []},
            },
        ),
        (
            "pause_path",
            {
                "target": "unrelated database records",
                "steps": [
                    {
                        "order": 1,
                        "action": "pause",
                        "target": "database records",
                        "max_attempts": 1,
                    }
                ],
                "verification": {
                    "condition": "database records look plausible",
                    "evidence": [
                        {"method": "status_readback", "expected": "records exist"}
                    ],
                },
            },
        ),
        (
            "pause_path",
            {
                "target": "governed-review scheduler",
                "steps": [
                    {
                        "order": 1,
                        "action": "observe",
                        "target": "governed-review scheduler",
                        "max_attempts": 1,
                    }
                ],
                "verification": {
                    "condition": "scheduler reports paused",
                    "evidence": [
                        {
                            "method": "scheduler_status",
                            "expected": "scheduler is paused",
                        }
                    ],
                },
            },
        ),
        (
            "rollback_path",
            {
                "target": "candidate commit",
                "steps": [
                    {
                        "order": 1,
                        "action": "revert",
                        "target": "do not revert candidate",
                        "max_attempts": 1,
                    }
                ],
                "verification": {
                    "condition": "candidate commit is absent",
                    "evidence": [
                        {"method": "git_show", "expected": "candidate is absent"}
                    ],
                },
            },
        ),
        (
            "pause_path",
            {
                "target": "governed-review scheduler",
                "steps": [
                    {
                        "order": 1,
                        "action": "pause",
                        "target": "governed-review scheduler",
                        "max_attempts": 1,
                    }
                ],
            },
        ),
    ],
)
def test_preflight_rejects_non_concrete_rollback_and_pause_procedures(
    conn, git_workspace, field_name: str, value
) -> None:
    workspace, head, branch = git_workspace
    task_id = kb.create_task(conn, title="candidate", assignee="worker")
    spec = replace(make_spec(task_id, workspace, head, branch), **{field_name: value})

    result = gr.preflight_governed_review(
        conn,
        spec,
        profile_resolver=lambda profile: True,
        skill_resolver=lambda profile, skill: True,
        delivery_readiness=delivery_ready,
    )

    assert not result.ok
    assert any(
        "structured, bounded operational procedure" in error for error in result.errors
    )


def test_preflight_accepts_valid_bounded_structured_procedures(
    conn, git_workspace
) -> None:
    workspace, head, branch = git_workspace
    task_id = kb.create_task(conn, title="candidate", assignee="worker")
    spec = replace(
        make_spec(task_id, workspace, head, branch),
        rollback_path={
            "target": "candidate release commit",
            "steps": [
                {
                    "order": 1,
                    "action": "revert",
                    "target": "candidate release commit",
                    "max_attempts": 1,
                }
            ],
            "verification": {
                "condition": "candidate release commit has a recorded revert",
                "evidence": [
                    {
                        "method": "git_show",
                        "expected": "revert names candidate release commit",
                    }
                ],
            },
        },
        pause_path={
            "target": "governed-review scheduler job",
            "steps": [
                {
                    "order": 1,
                    "action": "pause",
                    "target": "governed-review scheduler job",
                    "max_attempts": 1,
                }
            ],
            "verification": {
                "condition": "governed-review scheduler job reports paused",
                "evidence": [
                    {
                        "method": "scheduler_status",
                        "expected": "scheduler job state is paused",
                    }
                ],
            },
        },
    )

    result = gr.preflight_governed_review(
        conn,
        spec,
        profile_resolver=lambda profile: True,
        skill_resolver=lambda profile, skill: True,
        delivery_readiness=delivery_ready,
    )
    assert result.ok, result.errors


def test_delivery_retry_is_idempotent_does_not_rerun_product_and_releases_one_child(
    conn, git_workspace
) -> None:
    workspace, head, branch = git_workspace
    parent = kb.create_task(conn, title="parent", assignee="worker")
    child_a = kb.create_task(conn, title="child-a", assignee="worker", parents=[parent])
    child_b = kb.create_task(conn, title="child-b", assignee="worker", parents=[parent])
    assert kb.claim_task(conn, parent)
    spec = make_spec(parent, workspace, head, branch)
    pass_preflight(conn, spec)
    product_calls: list[str] = []
    delivery_calls: list[str] = []

    def product_work():
        product_calls.append("ran")
        return {"tests": "passed", "commit": head}

    def flaky_sender(target, body):
        delivery_calls.append(body)
        if len(delivery_calls) == 1:
            raise RuntimeError("simulated GitHub outage")

    assert not complete_ready(
        conn,
        spec,
        summary="candidate complete",
        product_work=product_work,
        sender=flaky_sender,
    )
    assert require_task(conn, parent).status == "blocked"
    assert len(product_calls) == 1

    assert complete_ready(
        conn,
        spec,
        summary="candidate complete",
        product_work=product_work,
        sender=flaky_sender,
    )
    assert require_task(conn, parent).status == "done"
    assert len(product_calls) == 1
    assert len(delivery_calls) == 2
    assert delivery_calls[0] == delivery_calls[1]
    assert require_task(conn, child_a).status == "blocked"
    assert require_task(conn, child_b).status == "blocked"
    assert len(gr._events(conn, parent, "governance_child_held")) == 2
    for _ in range(3):
        assert kb.recompute_ready(conn) == 0
    assert kb.claim_task(conn, child_a) is None
    assert kb.claim_task(conn, child_b) is None

    # A manually held child stays sticky while the reviewer explicitly releases
    # only its sibling.
    conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (child_b,))
    conn.commit()
    assert kb.block_task(conn, child_b, reason="manual hold")
    ok, error = gr.release_one_child(conn, parent, child_a, actor="reviewer")
    assert ok and error is None
    assert require_task(conn, child_a).status == "ready"
    assert require_task(conn, child_b).status == "blocked"
    assert kb.recompute_ready(conn) == 0
    assert require_task(conn, child_b).status == "blocked"

    # Re-entry after success is idempotent: neither product nor delivery reruns.
    assert complete_ready(
        conn,
        spec,
        summary="candidate complete",
        product_work=product_work,
        sender=flaky_sender,
    )
    assert len(product_calls) == 1
    assert len(delivery_calls) == 2


def test_delivery_exhaustion_preserves_distinct_truthful_pause_outcomes(
    conn, git_workspace
) -> None:
    workspace, head, branch = git_workspace
    task_id = kb.create_task(conn, title="candidate", assignee="worker")
    assert kb.claim_task(conn, task_id)
    spec = make_spec(task_id, workspace, head, branch)
    pass_preflight(conn, spec)
    product_calls: list[int] = []
    sends: list[int] = []
    successful_pauses: list[str] = []
    failed_pause_calls: list[str] = []

    def product_work():
        product_calls.append(1)
        return {"state": "preserved"}

    def failing_sender(target, body):
        sends.append(1)
        raise RuntimeError("offline")

    def failing_pause(reason: str) -> None:
        failed_pause_calls.append(reason)
        raise RuntimeError("pause executor offline")

    assert not complete_ready(
        conn,
        spec,
        summary="done",
        product_work=product_work,
        sender=failing_sender,
        pause_scheduler=successful_pauses.append,
    )
    assert not complete_ready(
        conn,
        spec,
        summary="done",
        product_work=product_work,
        sender=failing_sender,
        pause_scheduler=failing_pause,
    )
    # Exhaustion has no executor. Re-entry is idempotent and never creates attempt 3.
    for _ in range(2):
        assert not complete_ready(
            conn,
            spec,
            summary="done",
            product_work=product_work,
            sender=failing_sender,
        )

    assert len(product_calls) == 1
    assert len(sends) == 2
    assert len(successful_pauses) == 1
    assert len(failed_pause_calls) == 1
    assert len(gr._events(conn, task_id, "governance_delivery_failed")) == 2
    assert len(gr._events(conn, task_id, "governance_delivery_reserved")) == 2
    assert len(gr._events(conn, task_id, "governance_scheduler_paused")) == 1
    assert len(gr._events(conn, task_id, "governance_scheduler_pause_failed")) == 1
    assert len(gr._events(conn, task_id, "governance_scheduler_pause_unavailable")) == 1
    reservations = gr._events(conn, task_id, "governance_scheduler_pause_reserved")
    blocker_keys = {(event.payload or {})["blocker_key"] for event in reservations}
    assert len(blocker_keys) == 3
    assert any(key.startswith("delivery-failed:") for key in blocker_keys)
    assert any(key.startswith("delivery-exhausted:") for key in blocker_keys)
    task = kb.get_task(conn, task_id)
    assert task is not None and task.status == "blocked"


def test_exhaustion_allows_one_preserved_state_recovery_and_rejects_second(
    conn,
) -> None:
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
    assert require_task(conn, task_id).status == "ready"
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
    assert require_task(conn, task_id).status == "blocked"
    assert len(pauses) == 1


@pytest.mark.parametrize(
    "hold_reason",
    [
        "manual hold",
        "review-required: human approval",
        "doctrine hold",
        "security hold",
        "policy hold",
        "governance-preflight: invalid state",
        "delivery-failed: offline",
    ],
    ids=[
        "manual",
        "review-required",
        "doctrine",
        "security",
        "policy",
        "preflight",
        "delivery",
    ],
)
def test_preserved_recovery_refuses_substantive_holds_even_with_historical_exhaustion(
    conn, hold_reason: str
) -> None:
    task_id = kb.create_task(conn, title="candidate", assignee="worker")
    assert kb.claim_task(conn, task_id)
    conn.execute(
        "INSERT INTO task_events(task_id, kind, payload, created_at) VALUES (?, 'gave_up', ?, 1)",
        (task_id, '{"error":"historical exhaustion"}'),
    )
    conn.commit()
    assert kb.block_task(conn, task_id, reason=hold_reason)

    ok, detail = gr.request_preserved_state_recovery(
        conn, task_id, actor="B1-DO", reason="must fail closed"
    )

    assert not ok
    assert "substantive hold" in detail
    assert require_task(conn, task_id).status == "blocked"
    assert gr._events(conn, task_id, "governance_preserved_recovery") == []


def test_preserved_recovery_requires_durable_operational_failure_evidence(conn) -> None:
    task_id = kb.create_task(conn, title="candidate", assignee="worker")
    conn.execute("UPDATE tasks SET status='blocked' WHERE id=?", (task_id,))
    conn.commit()

    ok, detail = gr.request_preserved_state_recovery(
        conn, task_id, actor="B1-DO", reason="no evidence"
    )

    assert not ok
    assert "durable operational failure" in detail
    assert require_task(conn, task_id).status == "blocked"


def test_preserved_recovery_requires_failure_evidence_matching_current_breaker(
    conn,
) -> None:
    task_id = kb.create_task(conn, title="candidate", assignee="worker")
    conn.execute(
        "UPDATE tasks SET status='blocked', consecutive_failures=2, "
        "last_failure_error='current failure' WHERE id=?",
        (task_id,),
    )
    conn.execute(
        "INSERT INTO task_events(task_id, kind, payload, created_at) "
        "VALUES (?, 'gave_up', ?, 1)",
        (task_id, '{"error":"historical failure","failures":2}'),
    )
    conn.commit()

    ok, detail = gr.request_preserved_state_recovery(
        conn, task_id, actor="B1-DO", reason="mismatched evidence"
    )

    assert not ok
    assert "durable operational failure" in detail
    task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.status == "blocked"


def test_complete_task_default_release_is_unchanged_and_opt_out_is_explicit(
    conn,
) -> None:
    parent_default = kb.create_task(conn, title="default parent")
    child_default = kb.create_task(
        conn, title="default child", parents=[parent_default]
    )
    assert kb.complete_task(conn, parent_default, summary="done")
    assert require_task(conn, child_default).status == "ready"

    parent_governed = kb.create_task(conn, title="governed parent")
    child_governed = kb.create_task(
        conn, title="governed child", parents=[parent_governed]
    )
    assert kb.complete_task(
        conn, parent_governed, summary="done", recompute_dependents=False
    )
    assert require_task(conn, child_governed).status == "todo"


def test_product_failure_blocks_and_pauses_without_delivery(
    conn, git_workspace
) -> None:
    workspace, head, branch = git_workspace
    task_id = kb.create_task(conn, title="candidate", assignee="worker")
    assert kb.claim_task(conn, task_id)
    spec = make_spec(task_id, workspace, head, branch)
    pass_preflight(conn, spec)
    sends: list[str] = []
    pauses: list[str] = []

    def product_work():
        raise RuntimeError("deterministic build failure")

    assert not complete_ready(
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


def test_scheduler_pause_interruption_requires_reconciliation(
    conn, monkeypatch
) -> None:
    task_id = kb.create_task(conn, title="candidate", assignee="worker")
    now = [4_000]
    monkeypatch.setattr(gr.time, "time", lambda: now[0])

    class SimulatedInterruption(BaseException):
        pass

    def interrupted_pause(reason: str) -> None:
        raise SimulatedInterruption(reason)

    with pytest.raises(SimulatedInterruption):
        gr._pause_once(
            conn,
            task_id,
            "pause-interruption",
            "interrupted pause",
            interrupted_pause,
        )
    assert (
        gr._pause_once(
            conn,
            task_id,
            "pause-interruption",
            "must not execute concurrently",
            pytest.fail,
        )
        == "in_flight"
    )

    now[0] += gr.PAUSE_RESERVATION_LEASE_SECONDS + 1
    assert (
        gr._pause_once(
            conn,
            task_id,
            "pause-interruption",
            "pause outcome unknown",
            pytest.fail,
        )
        == "reconciliation_required"
    )
    gr._sticky_block_once(conn, task_id, "pause outcome unknown")
    ok, error = gr.reconcile_scheduler_pause(
        conn,
        task_id,
        blocker_key="pause-interruption",
        actor="reviewer",
        actual_outcome="paused",
        evidence="OPS-7",
    )
    assert ok and error == "scheduler pause reconciled"
    assert len(gr._events(conn, task_id, "governance_scheduler_paused")) == 1


def test_release_one_child_cas_rejects_replaced_hold_relation(
    conn, monkeypatch
) -> None:
    parent = kb.create_task(conn, title="parent", assignee="worker")
    child = kb.create_task(
        conn,
        title="child",
        assignee="worker",
        parents=[parent],
    )
    assert kb.complete_task(conn, parent, recompute_dependents=False)
    gr._hold_governed_children(conn, parent)
    original_events = gr._events
    injected = False

    def events_with_replacement(connection, task_id, kind=None):
        nonlocal injected
        events = original_events(connection, task_id, kind)
        if task_id == child and kind is None and not injected:
            injected = True
            kb._append_event(
                connection,
                child,
                "blocked",
                {"reason": "manual replacement hold"},
            )
        return events

    monkeypatch.setattr(gr, "_events", events_with_replacement)
    ok, error = gr.release_one_child(
        conn,
        parent,
        child,
        actor="reviewer",
    )
    assert not ok
    assert error == f"child {child} hold changed during governed release"
    task = kb.get_task(conn, child)
    assert task is not None and task.status == "blocked"
    assert not gr._events(conn, parent, "governance_child_released")


def test_stale_delivery_owner_cannot_record_after_replacement(
    conn, git_workspace, monkeypatch
) -> None:
    workspace, head, branch = git_workspace
    task_id = kb.create_task(conn, title="candidate", assignee="worker")
    spec = make_spec(task_id, workspace, head, branch)
    dedupe_key = gr._dedupe_key(spec, gr._sha256(spec.prompt))
    now = [5_000]
    monkeypatch.setattr(gr.time, "time", lambda: now[0])

    state, attempt, old_owner = gr._reserve_delivery_attempt(conn, task_id, dedupe_key)
    assert state == "owner" and attempt == 1 and old_owner
    now[0] += gr.DELIVERY_RESERVATION_LEASE_SECONDS + 1
    state, blocked_attempt, new_owner = gr._reserve_delivery_attempt(
        conn, task_id, dedupe_key
    )
    assert state == "reconciliation"
    assert blocked_attempt == 1 and new_owner is None
    assert not gr._record_delivery_outcome(
        conn,
        task_id,
        dedupe=dedupe_key,
        attempt=1,
        reservation_id=old_owner,
        kind="governance_delivery_succeeded",
        payload={"target": str(spec.delivery)},
    )
    reconciliations = gr._events(
        conn,
        task_id,
        "governance_delivery_reconciliation_required",
    )
    assert len(reconciliations) == 1
    assert (reconciliations[0].payload or {})["reservation_id"] == old_owner


def test_interrupted_product_lease_requires_manual_reconciliation(
    conn, git_workspace, monkeypatch
) -> None:
    workspace, head, branch = git_workspace
    task_id = kb.create_task(conn, title="candidate", assignee="worker")
    assert kb.claim_task(conn, task_id)
    spec = make_spec(task_id, workspace, head, branch)
    pass_preflight(conn, spec)
    now = [6_000]
    monkeypatch.setattr(gr.time, "time", lambda: now[0])
    product_calls: list[str] = []
    sends: list[str] = []

    class SimulatedInterruption(BaseException):
        pass

    def interrupted_product():
        product_calls.append("interrupted")
        raise SimulatedInterruption("worker terminated")

    with pytest.raises(SimulatedInterruption):
        complete_ready(
            conn,
            spec,
            summary="done",
            product_work=interrupted_product,
            sender=lambda target, body: sends.append(body),
        )
    assert not complete_ready(
        conn,
        spec,
        summary="done",
        product_work=lambda: product_calls.append("unexpected"),
        sender=lambda target, body: sends.append(body),
    )
    assert product_calls == ["interrupted"]

    now[0] += gr.PRODUCT_RESERVATION_LEASE_SECONDS + 1
    assert not complete_ready(
        conn,
        spec,
        summary="done",
        product_work=lambda: product_calls.append("unexpected"),
        sender=lambda target, body: sends.append(body),
    )
    assert product_calls == ["interrupted"]
    assert (
        len(gr._events(conn, task_id, "governance_product_reconciliation_required"))
        == 1
    )

    ok, error = gr.reconcile_product_execution(
        conn,
        task_id,
        dedupe=gr._dedupe_key(spec, gr._sha256(spec.prompt)),
        actor="reviewer",
        actual_outcome="completed",
        evidence={"tests": "verified externally"},
    )
    assert ok and error == "product reconciled"
    assert complete_ready(
        conn,
        spec,
        summary="done",
        product_work=lambda: product_calls.append("unexpected"),
        sender=lambda target, body: sends.append(body),
    )
    assert product_calls == ["interrupted"]
    assert len(sends) == 1
    task = kb.get_task(conn, task_id)
    assert task is not None and task.status == "done"


def test_preserved_state_recovery_rolls_back_on_interruption(conn, monkeypatch) -> None:
    task_id = kb.create_task(conn, title="candidate", assignee="worker")
    conn.execute(
        "UPDATE tasks SET status='blocked', consecutive_failures=2, "
        "last_failure_error='turns exhausted' WHERE id=?",
        (task_id,),
    )
    conn.execute(
        "INSERT INTO task_events(task_id, kind, payload, created_at) "
        "VALUES (?, 'gave_up', ?, 1)",
        (task_id, '{"error":"turns exhausted"}'),
    )
    conn.commit()
    original_append = kb._append_event

    class SimulatedInterruption(BaseException):
        pass

    def interrupting_append(connection, event_task_id, kind, payload):
        if kind == "promoted_manual":
            raise SimulatedInterruption("worker terminated during recovery")
        return original_append(connection, event_task_id, kind, payload)

    monkeypatch.setattr(kb, "_append_event", interrupting_append)
    with pytest.raises(SimulatedInterruption):
        gr.request_preserved_state_recovery(
            conn,
            task_id,
            actor="reviewer",
            reason="one bounded retry",
        )
    task = kb.get_task(conn, task_id)
    assert task is not None and task.status == "blocked"
    assert not gr._events(conn, task_id, "governance_preserved_recovery")

    monkeypatch.setattr(kb, "_append_event", original_append)
    ok, detail = gr.request_preserved_state_recovery(
        conn,
        task_id,
        actor="reviewer",
        reason="one bounded retry",
    )
    assert ok and detail == "recovery released"
    task = kb.get_task(conn, task_id)
    assert task is not None and task.status == "ready"
    assert len(gr._events(conn, task_id, "governance_preserved_recovery")) == 1


def test_preserved_state_recovery_is_atomic_under_concurrency(conn) -> None:
    task_id = kb.create_task(conn, title="candidate", assignee="worker")
    conn.execute(
        "UPDATE tasks SET status='blocked', consecutive_failures=2, "
        "last_failure_error='turns exhausted' WHERE id=?",
        (task_id,),
    )
    conn.execute(
        "INSERT INTO task_events(task_id, kind, payload, created_at) "
        "VALUES (?, 'gave_up', ?, 1)",
        (task_id, '{"error":"turns exhausted"}'),
    )
    conn.commit()
    db_path = Path(conn.execute("PRAGMA database_list").fetchone()[2])
    barrier = threading.Barrier(2)
    results: list[tuple[bool, str]] = []
    errors: list[BaseException] = []

    def recover() -> None:
        local = kb.connect(db_path)
        try:
            barrier.wait()
            results.append(
                gr.request_preserved_state_recovery(
                    local,
                    task_id,
                    actor="reviewer",
                    reason="one bounded retry",
                )
            )
        except BaseException as exc:
            errors.append(exc)
        finally:
            local.close()

    threads = [threading.Thread(target=recover) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not errors
    assert sorted(ok for ok, _ in results) == [False, True]
    task = kb.get_task(conn, task_id)
    assert task is not None and task.status == "ready"
    assert len(gr._events(conn, task_id, "governance_preserved_recovery")) == 1
    assert len(gr._events(conn, task_id, "governance_scheduler_paused")) == 0
    assert len(gr._events(conn, task_id, "governance_scheduler_pause_unavailable")) == 1


@pytest.mark.parametrize(
    ("viewer_permission", "viewer_can_update", "expected_ready"),
    [
        ("WRITE", True, True),
        ("ADMIN", True, True),
        ("READ", True, False),
        ("WRITE", False, False),
        (None, True, False),
    ],
)
def test_default_readiness_proves_sufficient_comment_permission_without_probe_comment(
    monkeypatch,
    viewer_permission: str | None,
    viewer_can_update: bool,
    expected_ready: bool,
) -> None:
    target = gr.GitHubIssueTarget("owner/repo", 2)
    commands: list[list[str]] = []
    monkeypatch.setattr(gr.shutil, "which", lambda executable: "/usr/bin/gh")

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[:3] == ["gh", "auth", "status"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        assert command[:3] == ["gh", "api", "graphql"]
        assert any("viewerPermission" in arg for arg in command)
        assert any("viewerCanUpdate" in arg for arg in command)
        payload = {
            "data": {
                "repository": {
                    "viewerPermission": viewer_permission,
                    "issue": {
                        "number": 2,
                        "viewerCanUpdate": viewer_can_update,
                    },
                }
            }
        }
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(payload), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    ready, detail = gr._default_delivery_readiness(target)

    assert ready is expected_ready
    assert (detail == "ready") is expected_ready
    assert len(commands) == 2
    assert all("comment" not in command for command in commands)
    assert all("issue" not in command or "view" not in command for command in commands)


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
