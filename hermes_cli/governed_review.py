"""Opt-in governed code-review packet primitives for Hermes Kanban.

This module is intentionally dormant: importing it changes no configuration,
scheduler, gateway, profile, or repository state. Callers must explicitly invoke
its functions and provide an immutable git pin plus a durable issue target.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from hermes_cli import kanban_db as kb

PROTOCOL_VERSION = "governed-review-v1"
MAX_DELIVERY_ATTEMPTS = 2  # initial attempt + one delivery-only retry


@dataclass(frozen=True)
class GitHubIssueTarget:
    repository: str
    issue: int

    def validate(self) -> Optional[str]:
        repo = self.repository.strip()
        if not repo or "/" not in repo or repo.startswith("/") or repo.endswith("/"):
            return "delivery repository must be configured as owner/name"
        try:
            issue = int(self.issue)
        except (TypeError, ValueError):
            return "delivery issue must be a positive integer"
        if issue <= 0:
            return "delivery issue must be a positive integer"
        return None


@dataclass(frozen=True)
class GovernedReviewSpec:
    task_id: str
    workspace: Path
    expected_commit: str
    prompt: str
    delivery: GitHubIssueTarget
    expected_branch: Optional[str] = None
    session_id: Optional[str] = None
    model: Optional[str] = None
    rollback_path: str = "revert the isolated candidate commit; keep templates disabled"
    pause_path: str = "pause the opt-in scheduler and preserve the Kanban event log"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreflightResult:
    ok: bool
    errors: tuple[str, ...]
    evidence: dict[str, Any]
    dedupe_key: str


ProfileResolver = Callable[[str], bool]
SkillResolver = Callable[[str, str], bool]
ReadinessProbe = Callable[[GitHubIssueTarget], tuple[bool, str]]
PauseCallback = Callable[[str], None]
DeliverySender = Callable[[GitHubIssueTarget, str], None]
ProductCallback = Callable[[], dict[str, Any]]


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _run_git(workspace: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(workspace),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
        check=False,
    )


def _default_profile_resolver(profile: str) -> bool:
    from hermes_cli.profiles import profile_exists

    return bool(profile_exists(profile))


def _default_skill_resolver(profile: str, skill: str) -> bool:
    """Resolve a requested skill in the actual assignee profile home.

    The dispatcher switches HERMES_HOME before the worker preloads task skills,
    so this deliberately checks the assignee's effective home rather than the
    dispatcher's current profile. Tests may inject the exact CLI resolver.
    """
    from agent.skill_utils import iter_skill_index_files, parse_frontmatter
    from hermes_cli.profiles import resolve_profile_env

    try:
        root = Path(resolve_profile_env(profile)) / "skills"
        for skill_md in iter_skill_index_files(root, "SKILL.md"):
            if skill_md.parent.name == skill:
                return True
            try:
                frontmatter, _ = parse_frontmatter(skill_md.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if str((frontmatter or {}).get("name") or "") == skill:
                return True
    except (FileNotFoundError, OSError):
        return False
    return False


def _default_delivery_readiness(target: GitHubIssueTarget) -> tuple[bool, str]:
    """Non-mutating proof that the authenticated viewer may comment."""
    if shutil.which("gh") is None:
        return False, "gh executable is unavailable"
    auth = subprocess.run(
        ["gh", "auth", "status"],
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=15,
        check=False,
    )
    if auth.returncode != 0:
        return False, "gh authentication unavailable"

    owner, name = target.repository.split("/", 1)
    query = (
        "query($owner:String!,$name:String!,$number:Int!){"
        "repository(owner:$owner,name:$name){viewerPermission "
        "issue(number:$number){number viewerCanUpdate}}}"
    )
    permission = subprocess.run(
        [
            "gh",
            "api",
            "graphql",
            "-f",
            f"query={query}",
            "-F",
            f"owner={owner}",
            "-F",
            f"name={name}",
            "-F",
            f"number={int(target.issue)}",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=15,
        check=False,
    )
    if permission.returncode != 0:
        return False, "configured GitHub issue comment permission is unavailable"
    try:
        payload = json.loads(permission.stdout or "{}")
        repository = payload["data"]["repository"]
        issue = repository["issue"]
        viewer_permission = repository["viewerPermission"]
        viewer_can_update = issue["viewerCanUpdate"]
        issue_number = issue["number"]
    except (KeyError, TypeError, ValueError):
        return False, "configured GitHub issue comment permission could not be verified"
    write_permissions = {"WRITE", "MAINTAIN", "ADMIN"}
    if (
        viewer_permission not in write_permissions
        or viewer_can_update is not True
        or issue_number != int(target.issue)
    ):
        return False, "authenticated viewer lacks verified issue-comment capability"
    return True, "ready"


def _default_delivery_sender(target: GitHubIssueTarget, body: str) -> None:
    marker_start = body.rfind("<!-- hermes-governed-review:")
    marker = body[marker_start:].strip() if marker_start >= 0 else ""
    if marker:
        probe = subprocess.run(
            [
                "gh", "api", "--paginate", "--slurp",
                f"repos/{target.repository}/issues/{target.issue}/comments",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
        if probe.returncode != 0:
            raise RuntimeError("GitHub delivery dedupe probe failed")
        try:
            pages = json.loads(probe.stdout or "[]")
            comments = [comment for page in pages for comment in page]
        except (TypeError, ValueError):
            raise RuntimeError("GitHub delivery dedupe probe returned invalid JSON")
        if any(marker in str(comment.get("body") or "") for comment in comments):
            return

    proc = subprocess.run(
        [
            "gh", "issue", "comment", str(target.issue),
            "--repo", target.repository, "--body", body,
        ],
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or "GitHub issue delivery failed").strip()[:500])


def _events(conn, task_id: str, kind: Optional[str] = None):
    events = kb.list_events(conn, task_id)
    return [event for event in events if kind is None or event.kind == kind]


def _append(conn, task_id: str, kind: str, payload: dict[str, Any]) -> None:
    with kb.write_txn(conn):
        kb._append_event(conn, task_id, kind, payload)  # package-internal audit path


def _append_once(
    conn,
    task_id: str,
    kind: str,
    payload: dict[str, Any],
    *,
    identity_keys: tuple[str, ...],
) -> bool:
    """Atomically append one event for the selected payload identity."""
    with kb.write_txn(conn):
        for event in _events(conn, task_id, kind):
            prior = event.payload or {}
            if all(prior.get(key) == payload.get(key) for key in identity_keys):
                return False
        kb._append_event(conn, task_id, kind, payload)
    return True


def _pause_once(
    conn,
    task_id: str,
    blocker_key: str,
    reason: str,
    pause_scheduler: Optional[PauseCallback],
) -> None:
    reservation_id = uuid.uuid4().hex
    with kb.write_txn(conn):
        prior = [
            event
            for event in _events(conn, task_id)
            if event.kind in (
                "governance_scheduler_pause_reserved",
                "governance_scheduler_paused",
                "governance_scheduler_pause_failed",
            )
            and (event.payload or {}).get("blocker_key") == blocker_key
        ]
        if prior:
            return
        kb._append_event(
            conn,
            task_id,
            "governance_scheduler_pause_reserved",
            {
                "blocker_key": blocker_key,
                "reason": reason,
                "reservation_id": reservation_id,
            },
        )

    try:
        if pause_scheduler is not None:
            pause_scheduler(reason)
    except Exception as exc:
        _append(
            conn,
            task_id,
            "governance_scheduler_pause_failed",
            {
                "blocker_key": blocker_key,
                "reason": reason,
                "reservation_id": reservation_id,
                "error": str(exc)[:500],
            },
        )
        return
    _append(
        conn,
        task_id,
        "governance_scheduler_paused",
        {
            "blocker_key": blocker_key,
            "reason": reason,
            "reservation_id": reservation_id,
        },
    )


def _sticky_block_once(conn, task_id: str, reason: str) -> None:
    task = kb.get_task(conn, task_id)
    if task is None or task.status == "blocked":
        return
    if kb.block_task(conn, task_id, reason=reason):
        return
    # A dependency-gated task may still be todo when preflight runs. Preserve
    # the same explicit-block event contract without bypassing a running claim.
    with kb.write_txn(conn):
        cur = conn.execute(
            "UPDATE tasks SET status='blocked' WHERE id=? AND status='todo'",
            (task_id,),
        )
        if cur.rowcount == 1:
            kb._append_event(conn, task_id, "blocked", {"reason": reason})


def _dedupe_key(spec: GovernedReviewSpec, prompt_hash: str) -> str:
    material = "|".join(
        [
            PROTOCOL_VERSION,
            spec.task_id,
            spec.expected_commit,
            f"{spec.delivery.repository}#{spec.delivery.issue}",
            prompt_hash,
        ]
    )
    return f"governed-review:{_sha256(material)}"


def _validate_procedure(label: str, value: str) -> Optional[str]:
    text = str(value or "").strip()
    normalized = " ".join(text.casefold().split()).strip(" .;:-_")
    placeholders = {
        "n/a",
        "na",
        "none",
        "not applicable",
        "placeholder",
        "tbd",
        "todo",
        "to do",
        "unknown",
        "later",
    }
    action_words = {
        "block",
        "disable",
        "keep",
        "pause",
        "preserve",
        "restore",
        "revert",
        "roll back",
        "rollback",
        "stop",
    }
    if (
        not text
        or normalized in placeholders
        or len(text) < 20
        or len(text.split()) < 4
        or not any(action in normalized for action in action_words)
    ):
        return f"{label} procedure must be a concrete, actionable procedure"
    return None


def _preflight_identity(
    spec: GovernedReviewSpec,
    prompt_hash: str,
    dedupe: str,
) -> dict[str, Any]:
    return {
        "task_id": spec.task_id,
        "workspace": str(spec.workspace.resolve()),
        "expected_commit": spec.expected_commit,
        "expected_branch": spec.expected_branch,
        "delivery_target": f"{spec.delivery.repository}#{spec.delivery.issue}",
        "prompt_hash": prompt_hash,
        "dedupe_key": dedupe,
        "rollback_path": spec.rollback_path,
        "pause_path": spec.pause_path,
    }


def _governance_gate_events(conn, task_id: str):
    kinds = {
        "governance_preflight_passed",
        "governance_preflight_blocked",
        "governance_execution_blocked",
    }
    return [event for event in _events(conn, task_id) if event.kind in kinds]


def _payload_matches_identity(payload: Optional[dict[str, Any]], identity: dict[str, Any]) -> bool:
    return bool(payload) and all(payload.get(key) == value for key, value in identity.items())


def preflight_governed_review(
    conn,
    spec: GovernedReviewSpec,
    *,
    profile_resolver: ProfileResolver = _default_profile_resolver,
    skill_resolver: SkillResolver = _default_skill_resolver,
    delivery_readiness: ReadinessProbe = _default_delivery_readiness,
    pause_scheduler: Optional[PauseCallback] = None,
) -> PreflightResult:
    """Fail closed before product work and persist one blocker on failure."""
    task = kb.get_task(conn, spec.task_id)
    prompt_hash = _sha256(spec.prompt)
    dedupe = _dedupe_key(spec, prompt_hash)
    errors: list[str] = []
    evidence: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "checked_at": int(time.time()),
        "task_id": spec.task_id,
        "session_id": spec.session_id,
        "model": spec.model,
        "prompt_hash": prompt_hash,
        "dedupe_key": dedupe,
        "expected_commit": spec.expected_commit,
        "expected_branch": spec.expected_branch,
        "delivery_target": f"{spec.delivery.repository}#{spec.delivery.issue}",
        "rollback_path": spec.rollback_path,
        "pause_path": spec.pause_path,
        "metadata": dict(spec.metadata),
    }

    if task is None:
        return PreflightResult(
            False,
            (f"task {spec.task_id} does not exist",),
            evidence,
            dedupe,
        )

    rollback_error = _validate_procedure("rollback", spec.rollback_path)
    pause_error = _validate_procedure("pause", spec.pause_path)
    if rollback_error:
        errors.append(rollback_error)
    if pause_error:
        errors.append(pause_error)

    evidence["actual_profile"] = task.assignee
    evidence["requested_skills"] = list(task.skills or [])
    try:
        profile_ready = bool(task.assignee and profile_resolver(task.assignee))
    except Exception as exc:
        profile_ready = False
        errors.append(f"actual assignee profile resolution failed: {exc}")
    if not profile_ready and not any(
        "profile resolution failed" in error for error in errors
    ):
        errors.append(f"actual assignee profile does not exist: {task.assignee!r}")
    if profile_ready:
        assert task.assignee is not None
        missing: list[str] = []
        for skill in task.skills or []:
            try:
                resolved = skill_resolver(task.assignee, skill)
            except Exception:
                resolved = False
            if not resolved:
                missing.append(skill)
        if missing:
            errors.append(
                f"requested skills do not resolve for {task.assignee}: "
                f"{', '.join(missing)}"
            )

    parent_rows = conn.execute(
        "SELECT p.id, p.status FROM tasks p JOIN task_links l ON l.parent_id=p.id "
        "WHERE l.child_id=? ORDER BY p.id",
        (spec.task_id,),
    ).fetchall()
    evidence["dependencies"] = [dict(row) for row in parent_rows]
    open_parents = [
        row["id"] for row in parent_rows
        if row["status"] not in ("done", "archived")
    ]
    if open_parents:
        errors.append(f"unsatisfied dependencies: {', '.join(open_parents)}")

    target_error = spec.delivery.validate()
    if target_error:
        errors.append(target_error)
    else:
        try:
            ready, readiness_detail = delivery_readiness(spec.delivery)
        except Exception as exc:  # readiness itself must fail closed
            ready, readiness_detail = False, f"readiness probe failed: {exc}"
        evidence["delivery_readiness"] = readiness_detail
        if not ready:
            errors.append(f"durable GitHub issue delivery is not ready: {readiness_detail}")

    workspace = spec.workspace.resolve()
    evidence["workspace"] = str(workspace)
    if not workspace.is_dir():
        errors.append(f"workspace does not exist: {workspace}")
    else:
        head = _run_git(workspace, "rev-parse", "HEAD")
        branch = _run_git(workspace, "branch", "--show-current")
        status = _run_git(workspace, "status", "--porcelain=v1")
        if head.returncode != 0 or status.returncode != 0 or branch.returncode != 0:
            errors.append("workspace is not a readable git worktree")
        else:
            actual_commit = head.stdout.strip()
            actual_branch = branch.stdout.strip()
            evidence["actual_commit"] = actual_commit
            evidence["actual_branch"] = actual_branch
            evidence["tracked_or_staged_clean"] = not bool(status.stdout.strip())
            if actual_commit != spec.expected_commit:
                errors.append(
                    f"immutable commit pin mismatch: expected {spec.expected_commit}, got {actual_commit}"
                )
            if spec.expected_branch and actual_branch != spec.expected_branch:
                errors.append(
                    f"branch mismatch: expected {spec.expected_branch}, got {actual_branch}"
                )
            if status.stdout.strip():
                errors.append(
                    "tracked or staged worktree state is not clean; "
                    "untracked material is also forbidden"
                )

    result = PreflightResult(not errors, tuple(errors), evidence, dedupe)
    gate_events = _governance_gate_events(conn, spec.task_id)
    latest_gate = gate_events[-1] if gate_events else None
    identity = _preflight_identity(spec, prompt_hash, dedupe)
    identity.update(
        {
            "actual_profile": task.assignee,
            "requested_skills": list(task.skills or []),
        }
    )
    if result.ok:
        if (
            latest_gate is None
            or latest_gate.kind != "governance_preflight_passed"
            or not _payload_matches_identity(latest_gate.payload, identity)
        ):
            _append(conn, spec.task_id, "governance_preflight_passed", evidence)
        return result

    blocker_material = json.dumps(
        {"identity": identity, "errors": errors}, sort_keys=True, default=str
    )
    blocker_key = f"preflight:{dedupe}:{_sha256(blocker_material)}"
    if (
        latest_gate is None
        or latest_gate.kind != "governance_preflight_blocked"
        or (latest_gate.payload or {}).get("blocker_key") != blocker_key
    ):
        payload = dict(evidence)
        payload.update({"blocker_key": blocker_key, "errors": list(errors)})
        _append(conn, spec.task_id, "governance_preflight_blocked", payload)
        _sticky_block_once(conn, spec.task_id, "governance-preflight: " + "; ".join(errors))
        _pause_once(conn, spec.task_id, blocker_key, "; ".join(errors), pause_scheduler)
    return result


def hold_for_review(conn, task_id: str, reason: str) -> bool:
    """Create a sticky terminal review hold; descendants remain unreleased."""
    if not reason.startswith("review-required:"):
        reason = f"review-required: {reason}"
    task = kb.get_task(conn, task_id)
    if task is None:
        return False
    if task.status == "blocked":
        return True
    return kb.block_task(conn, task_id, reason=reason, expected_run_id=task.current_run_id)


def release_one_child(conn, parent_id: str, child_id: str, *, actor: str) -> tuple[bool, Optional[str]]:
    """Release exactly one child whose active hold belongs to this parent."""
    if child_id not in kb.child_ids(conn, parent_id):
        return False, f"{child_id} is not a child of {parent_id}"
    parent = kb.get_task(conn, parent_id)
    if parent is None or parent.status not in ("done", "archived"):
        return False, f"parent {parent_id} is not terminal"
    child = kb.get_task(conn, child_id)
    if child is None:
        return False, f"child {child_id} does not exist"
    hold_reason = f"governance-child-hold:{parent_id}"
    active_reason = _active_block_reason(_events(conn, child_id))
    if child.status != "blocked" or active_reason != hold_reason:
        return False, f"child {child_id} is not held for governed release from {parent_id}"
    ok, error = kb.promote_task(
        conn,
        child_id,
        actor=actor,
        reason=f"explicit governed release from {parent_id}",
    )
    if ok:
        _append(
            conn,
            child_id,
            "unblocked",
            {
                "status": "ready",
                "reason": f"explicit governed release from {parent_id}",
            },
        )
        _append(
            conn,
            parent_id,
            "governance_child_released",
            {"child_id": child_id, "actor": actor},
        )
    return ok, error


def _active_block_reason(events) -> Optional[str]:
    state_events = [event for event in events if event.kind in ("blocked", "unblocked")]
    if not state_events or state_events[-1].kind != "blocked":
        return None
    return str((state_events[-1].payload or {}).get("reason") or "")


def _execution_gate_errors(
    conn,
    spec: GovernedReviewSpec,
    *,
    task,
    prompt_hash: str,
    dedupe: str,
    product_already_completed: bool,
    delivery_readiness: ReadinessProbe,
    profile_resolver: ProfileResolver,
    skill_resolver: SkillResolver,
) -> list[str]:
    errors: list[str] = []

    try:
        profile_ready = bool(task.assignee and profile_resolver(task.assignee))
    except Exception as exc:
        profile_ready = False
        errors.append(f"actual assignee profile resolution failed: {exc}")
    if not profile_ready and not any("profile resolution failed" in error for error in errors):
        errors.append(f"actual assignee profile does not exist: {task.assignee!r}")
    if profile_ready:
        missing_skills: list[str] = []
        for skill in task.skills or []:
            try:
                resolved = skill_resolver(task.assignee, skill)
            except Exception:
                resolved = False
            if not resolved:
                missing_skills.append(skill)
        if missing_skills:
            errors.append(
                f"requested skills do not resolve for {task.assignee}: "
                f"{', '.join(missing_skills)}"
            )

    for label, value in (("rollback", spec.rollback_path), ("pause", spec.pause_path)):
        procedure_error = _validate_procedure(label, value)
        if procedure_error:
            errors.append(procedure_error)

    identity = _preflight_identity(spec, prompt_hash, dedupe)
    identity.update(
        {
            "actual_profile": task.assignee,
            "requested_skills": list(task.skills or []),
        }
    )
    gate_events = _governance_gate_events(conn, spec.task_id)
    latest_gate = gate_events[-1] if gate_events else None
    matching_preflight = bool(
        latest_gate is not None
        and latest_gate.kind == "governance_preflight_passed"
        and _payload_matches_identity(latest_gate.payload, identity)
    )
    if not matching_preflight:
        errors.append("no current matching successful governance preflight")
        preflight_payload: dict[str, Any] = {}
    else:
        assert latest_gate is not None
        preflight_payload = latest_gate.payload or {}
        if preflight_payload.get("actual_profile") != task.assignee:
            errors.append("task assignee changed after governance preflight")
        if preflight_payload.get("requested_skills") != list(task.skills or []):
            errors.append("task skills changed after governance preflight")
    parent_rows = conn.execute(
        "SELECT p.id, p.status FROM tasks p JOIN task_links l ON l.parent_id=p.id "
        "WHERE l.child_id=? ORDER BY p.id",
        (spec.task_id,),
    ).fetchall()
    open_parents = [
        row["id"] for row in parent_rows
        if row["status"] not in ("done", "archived")
    ]
    if open_parents:
        errors.append(f"unsatisfied dependencies: {', '.join(open_parents)}")

    events = _events(conn, spec.task_id)
    if task.status == "blocked":
        indexed_events = list(enumerate(events))
        state_events = [
            (index, event)
            for index, event in indexed_events
            if event.kind in ("blocked", "unblocked")
        ]
        active_block = (
            state_events[-1]
            if state_events and state_events[-1][1].kind == "blocked"
            else None
        )
        last_unblocked_index = max(
            (
                index
                for index, event in state_events
                if event.kind == "unblocked"
            ),
            default=-1,
        )
        matching_delivery_failures = [
            (index, event)
            for index, event in indexed_events
            if event.kind == "governance_delivery_failed"
            and (event.payload or {}).get("dedupe_key") == dedupe
            and active_block is not None
            and last_unblocked_index < index < active_block[0]
        ]
        matching_delivery_failure = (
            matching_delivery_failures[-1] if matching_delivery_failures else None
        )
        block_reason = (
            str((active_block[1].payload or {}).get("reason") or "")
            if active_block is not None
            else ""
        )
        delivery_error = (
            str((matching_delivery_failure[1].payload or {}).get("error") or "")
            if matching_delivery_failure is not None
            else ""
        )
        delivery_retry_state = (
            product_already_completed
            and matching_delivery_failure is not None
            and bool(delivery_error)
            and block_reason == f"delivery-failed: {delivery_error[:300]}"
        )
        if not delivery_retry_state:
            errors.append("task is blocked by a substantive hold")
    elif task.status not in ("running", "ready"):
        errors.append(f"task status does not permit governed execution: {task.status}")

    workspace = spec.workspace.resolve()
    if not workspace.is_dir():
        errors.append(f"workspace does not exist: {workspace}")
    else:
        head = _run_git(workspace, "rev-parse", "HEAD")
        branch = _run_git(workspace, "branch", "--show-current")
        status = _run_git(workspace, "status", "--porcelain=v1")
        if head.returncode != 0 or branch.returncode != 0 or status.returncode != 0:
            errors.append("workspace is not a readable git worktree")
        else:
            actual_commit = head.stdout.strip()
            actual_branch = branch.stdout.strip()
            if actual_commit != spec.expected_commit:
                errors.append(
                    f"immutable commit pin mismatch: expected {spec.expected_commit}, "
                    f"got {actual_commit}"
                )
            if spec.expected_branch and actual_branch != spec.expected_branch:
                errors.append(
                    f"branch mismatch: expected {spec.expected_branch}, got {actual_branch}"
                )
            if status.stdout.strip():
                errors.append("immutable worktree is not clean, including untracked material")

    target_error = spec.delivery.validate()
    if target_error:
        errors.append(target_error)
    else:
        try:
            ready, detail = delivery_readiness(spec.delivery)
        except Exception as exc:
            ready, detail = False, f"readiness probe failed: {exc}"
        if not ready:
            errors.append(f"durable GitHub issue delivery is not ready: {detail}")
    return errors


def _record_execution_block(
    conn,
    spec: GovernedReviewSpec,
    *,
    prompt_hash: str,
    dedupe: str,
    errors: list[str],
    pause_scheduler: Optional[PauseCallback],
) -> None:
    blocker_key = f"execution:{dedupe}:{_sha256(json.dumps(errors, sort_keys=True))}"
    latest_gate_events = _governance_gate_events(conn, spec.task_id)
    latest_gate = latest_gate_events[-1] if latest_gate_events else None
    if (
        latest_gate is not None
        and latest_gate.kind == "governance_execution_blocked"
        and (latest_gate.payload or {}).get("blocker_key") == blocker_key
    ):
        return
    payload = _preflight_identity(spec, prompt_hash, dedupe)
    payload.update({"blocker_key": blocker_key, "errors": list(errors)})
    _append(conn, spec.task_id, "governance_execution_blocked", payload)
    _sticky_block_once(conn, spec.task_id, "governance-execution: " + "; ".join(errors))
    _pause_once(conn, spec.task_id, blocker_key, "; ".join(errors), pause_scheduler)


def _reserve_product_execution(conn, task_id: str, dedupe: str) -> tuple[str, Optional[str]]:
    """Atomically reserve the sole product callback owner for a dedupe key."""
    reservation_id = uuid.uuid4().hex
    with kb.write_txn(conn):
        matching = [
            event
            for event in _events(conn, task_id)
            if (event.payload or {}).get("dedupe_key") == dedupe
        ]
        if any(event.kind == "governance_product_completed" for event in matching):
            return "completed", None
        if any(event.kind == "governance_product_failed" for event in matching):
            return "failed", None
        if any(event.kind == "governance_product_reserved" for event in matching):
            return "reserved", None
        kb._append_event(
            conn,
            task_id,
            "governance_product_reserved",
            {
                "dedupe_key": dedupe,
                "reservation_id": reservation_id,
                "reserved_at": int(time.time()),
            },
        )
    return "owner", reservation_id


def _reserve_delivery_attempt(
    conn,
    task_id: str,
    dedupe: str,
) -> tuple[str, Optional[int], Optional[str]]:
    """Atomically reserve one sender attempt or identify a fail-closed state."""
    reservation_id = uuid.uuid4().hex
    with kb.write_txn(conn):
        matching = [
            event
            for event in _events(conn, task_id)
            if (event.payload or {}).get("dedupe_key") == dedupe
            and event.kind in (
                "governance_delivery_reserved",
                "governance_delivery_failed",
                "governance_delivery_succeeded",
            )
        ]
        if any(event.kind == "governance_delivery_succeeded" for event in matching):
            return "succeeded", None, None

        reservations: set[int] = set()
        outcomes: set[int] = set()
        for event in matching:
            raw_attempt = (event.payload or {}).get("attempt")
            if raw_attempt is None:
                continue
            try:
                attempt = int(raw_attempt)
            except (TypeError, ValueError):
                continue
            if event.kind == "governance_delivery_reserved":
                reservations.add(attempt)
            else:
                outcomes.add(attempt)

        unknown = sorted(reservations - outcomes)
        if unknown:
            return "unknown", unknown[0], None

        used_attempts = reservations | outcomes
        attempt = max(used_attempts, default=0) + 1
        if attempt > MAX_DELIVERY_ATTEMPTS:
            return "exhausted", None, None

        kb._append_event(
            conn,
            task_id,
            "governance_delivery_reserved",
            {
                "dedupe_key": dedupe,
                "attempt": attempt,
                "reservation_id": reservation_id,
                "reserved_at": int(time.time()),
            },
        )
    return "owner", attempt, reservation_id


def _hold_governed_children(conn, parent_id: str) -> list[str]:
    """Durably hold every linked child before governed parent completion."""
    errors: list[str] = []
    hold_reason = f"governance-child-hold:{parent_id}"
    with kb.write_txn(conn):
        children = conn.execute(
            "SELECT t.id, t.status FROM tasks t "
            "JOIN task_links l ON l.child_id=t.id "
            "WHERE l.parent_id=? ORDER BY t.id",
            (parent_id,),
        ).fetchall()
        parent_events = _events(conn, parent_id, "governance_child_held")
        recorded = {
            (event.payload or {}).get("child_id")
            for event in parent_events
        }
        for child in children:
            child_id = child["id"]
            status = child["status"]
            preexisting_reason = _active_block_reason(_events(conn, child_id))
            if status in ("todo", "ready"):
                cur = conn.execute(
                    "UPDATE tasks SET status='blocked' "
                    "WHERE id=? AND status=?",
                    (child_id, status),
                )
                if cur.rowcount != 1:
                    errors.append(f"child {child_id} changed while applying governed hold")
                    continue
                kb._append_event(conn, child_id, "blocked", {"reason": hold_reason})
                effective_reason = hold_reason
            elif status == "blocked":
                if preexisting_reason:
                    effective_reason = preexisting_reason
                else:
                    kb._append_event(conn, child_id, "blocked", {"reason": hold_reason})
                    effective_reason = hold_reason
            else:
                errors.append(
                    f"child {child_id} status {status!r} cannot be durably held"
                )
                continue

            if child_id not in recorded:
                kb._append_event(
                    conn,
                    parent_id,
                    "governance_child_held",
                    {
                        "child_id": child_id,
                        "prior_status": status,
                        "hold_reason": effective_reason,
                        "governed_release_required": effective_reason == hold_reason,
                    },
                )
    return errors


def _record_delivery_reconciliation(
    conn,
    spec: GovernedReviewSpec,
    *,
    dedupe: str,
    attempt: int,
    pause_scheduler: Optional[PauseCallback],
) -> None:
    payload = {
        "dedupe_key": dedupe,
        "attempt": attempt,
        "reason": "reserved delivery has no durable outcome; manual reconciliation required",
    }
    _append_once(
        conn,
        spec.task_id,
        "governance_delivery_reconciliation_required",
        payload,
        identity_keys=("dedupe_key", "attempt"),
    )
    _sticky_block_once(
        conn,
        spec.task_id,
        "delivery-unknown: manual reconciliation required",
    )
    _pause_once(
        conn,
        spec.task_id,
        f"delivery-unknown:{dedupe}:{attempt}",
        "delivery outcome is unknown; manual reconciliation required",
        pause_scheduler,
    )


def complete_with_issue_delivery(
    conn,
    spec: GovernedReviewSpec,
    *,
    summary: str,
    product_work: ProductCallback,
    sender: DeliverySender = _default_delivery_sender,
    delivery_readiness: ReadinessProbe = _default_delivery_readiness,
    profile_resolver: ProfileResolver = _default_profile_resolver,
    skill_resolver: SkillResolver = _default_skill_resolver,
    pause_scheduler: Optional[PauseCallback] = None,
) -> bool:
    """Run product and delivery callbacks only after durable atomic reservations.

    A known failed delivery permits one delivery-only retry. A reserved attempt
    without a durable success/failure outcome is never retried automatically.
    """
    prompt_hash = _sha256(spec.prompt)
    dedupe = _dedupe_key(spec, prompt_hash)
    task = kb.get_task(conn, spec.task_id)
    if task is None:
        return False

    success_events = [
        event
        for event in _events(conn, spec.task_id, "governance_delivery_succeeded")
        if (event.payload or {}).get("dedupe_key") == dedupe
    ]
    if task.status == "done":
        return bool(success_events)

    product_events = [
        event
        for event in _events(conn, spec.task_id, "governance_product_completed")
        if (event.payload or {}).get("dedupe_key") == dedupe
    ]
    gate_errors = _execution_gate_errors(
        conn,
        spec,
        prompt_hash=prompt_hash,
        dedupe=dedupe,
        task=task,
        product_already_completed=bool(product_events),
        delivery_readiness=delivery_readiness,
        profile_resolver=profile_resolver,
        skill_resolver=skill_resolver,
    )
    if gate_errors:
        _record_execution_block(
            conn,
            spec,
            prompt_hash=prompt_hash,
            dedupe=dedupe,
            errors=gate_errors,
            pause_scheduler=pause_scheduler,
        )
        return False

    if not product_events:
        product_state, product_reservation_id = _reserve_product_execution(
            conn, spec.task_id, dedupe
        )
        if product_state == "completed":
            product_events = [
                event
                for event in _events(conn, spec.task_id, "governance_product_completed")
                if (event.payload or {}).get("dedupe_key") == dedupe
            ]
        elif product_state != "owner" or product_reservation_id is None:
            return False
        else:
            try:
                evidence = product_work()
            except Exception as exc:
                blocker_key = f"product-failed:{dedupe}"
                _append(
                    conn,
                    spec.task_id,
                    "governance_product_failed",
                    {
                        "dedupe_key": dedupe,
                        "reservation_id": product_reservation_id,
                        "error": str(exc)[:500],
                    },
                )
                _sticky_block_once(
                    conn, spec.task_id, f"product-failed: {str(exc)[:300]}"
                )
                _pause_once(
                    conn,
                    spec.task_id,
                    blocker_key,
                    "product work failed",
                    pause_scheduler,
                )
                return False
            if not isinstance(evidence, dict):
                blocker_key = f"product-evidence-invalid:{dedupe}"
                _append(
                    conn,
                    spec.task_id,
                    "governance_product_failed",
                    {
                        "dedupe_key": dedupe,
                        "reservation_id": product_reservation_id,
                        "error": "product evidence was not a dict",
                    },
                )
                _sticky_block_once(
                    conn, spec.task_id, "product-failed: invalid evidence shape"
                )
                _pause_once(
                    conn,
                    spec.task_id,
                    blocker_key,
                    "product evidence invalid",
                    pause_scheduler,
                )
                return False
            _append(
                conn,
                spec.task_id,
                "governance_product_completed",
                {
                    "dedupe_key": dedupe,
                    "reservation_id": product_reservation_id,
                    "prompt_hash": prompt_hash,
                    "evidence": evidence,
                    "completed_at": int(time.time()),
                },
            )
            product_events = [
                event
                for event in _events(conn, spec.task_id, "governance_product_completed")
                if (event.payload or {}).get("dedupe_key") == dedupe
            ]

    if not success_events:
        delivery_state, attempt, delivery_reservation_id = _reserve_delivery_attempt(
            conn, spec.task_id, dedupe
        )
        if delivery_state == "unknown":
            assert attempt is not None
            _record_delivery_reconciliation(
                conn,
                spec,
                dedupe=dedupe,
                attempt=attempt,
                pause_scheduler=pause_scheduler,
            )
            return False
        if delivery_state == "exhausted":
            blocker_key = f"delivery-exhausted:{dedupe}"
            _sticky_block_once(
                conn,
                spec.task_id,
                "delivery-exhausted: manual intervention required",
            )
            _pause_once(
                conn,
                spec.task_id,
                blocker_key,
                "delivery retry exhausted",
                pause_scheduler,
            )
            return False
        if delivery_state == "succeeded":
            success_events = [
                event
                for event in _events(conn, spec.task_id, "governance_delivery_succeeded")
                if (event.payload or {}).get("dedupe_key") == dedupe
            ]
        elif delivery_state == "owner":
            assert attempt is not None and delivery_reservation_id is not None
            body = (
                f"{summary}\n\n"
                f"Protocol: {PROTOCOL_VERSION}\n"
                f"Task: {spec.task_id}\n"
                f"Commit pin: {spec.expected_commit}\n"
                f"Prompt hash: {prompt_hash}\n"
                f"Dedupe: {dedupe}\n\n"
                f"<!-- hermes-governed-review:{dedupe} -->"
            )
            try:
                sender(spec.delivery, body)
            except Exception as exc:
                _append(
                    conn,
                    spec.task_id,
                    "governance_delivery_failed",
                    {
                        "dedupe_key": dedupe,
                        "attempt": attempt,
                        "reservation_id": delivery_reservation_id,
                        "error": str(exc)[:500],
                        "delivery_only_retry_available": attempt < MAX_DELIVERY_ATTEMPTS,
                    },
                )
                _sticky_block_once(
                    conn, spec.task_id, f"delivery-failed: {str(exc)[:300]}"
                )
                _pause_once(
                    conn,
                    spec.task_id,
                    f"delivery-failed:{dedupe}:{attempt}",
                    f"delivery attempt {attempt} failed",
                    pause_scheduler,
                )
                return False
            _append(
                conn,
                spec.task_id,
                "governance_delivery_succeeded",
                {
                    "dedupe_key": dedupe,
                    "attempt": attempt,
                    "reservation_id": delivery_reservation_id,
                    "target": f"{spec.delivery.repository}#{spec.delivery.issue}",
                },
            )
            success_events = [
                event
                for event in _events(conn, spec.task_id, "governance_delivery_succeeded")
                if (event.payload or {}).get("dedupe_key") == dedupe
            ]
        else:
            return False

    child_hold_errors = _hold_governed_children(conn, spec.task_id)
    if child_hold_errors:
        _record_execution_block(
            conn,
            spec,
            prompt_hash=prompt_hash,
            dedupe=dedupe,
            errors=child_hold_errors,
            pause_scheduler=pause_scheduler,
        )
        return False

    product_payload = product_events[-1].payload or {}
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "dedupe_key": dedupe,
        "prompt_hash": prompt_hash,
        "delivery_target": f"{spec.delivery.repository}#{spec.delivery.issue}",
        "product_evidence": product_payload.get("evidence"),
        "rollback_path": spec.rollback_path,
        "pause_path": spec.pause_path,
        **dict(spec.metadata),
    }
    task = kb.get_task(conn, spec.task_id)
    return bool(
        task
        and kb.complete_task(
            conn,
            spec.task_id,
            summary=summary,
            metadata=metadata,
            expected_run_id=task.current_run_id,
            recompute_dependents=False,
        )
    )


def request_preserved_state_recovery(
    conn,
    task_id: str,
    *,
    actor: str,
    reason: str,
    pause_scheduler: Optional[PauseCallback] = None,
) -> tuple[bool, str]:
    """Allow exactly one explicit recovery while preserving failure evidence."""
    task = kb.get_task(conn, task_id)
    if task is None:
        return False, "task not found"
    prior = _events(conn, task_id, "governance_preserved_recovery")
    if prior:
        blocker_key = f"recovery-exhausted:{task_id}"
        _pause_once(conn, task_id, blocker_key, "second preserved-state recovery rejected", pause_scheduler)
        return False, "preserved-state recovery already used"
    if task.status != "blocked":
        return False, f"task must be blocked, got {task.status}"

    events = _events(conn, task_id)
    active_block_reason = _active_block_reason(events)
    if active_block_reason is not None:
        return False, f"preserved-state recovery refuses substantive hold: {active_block_reason}"

    breaker_events = [
        event for event in events if event.kind == "gave_up"
    ]
    matching_breaker_events = [
        event
        for event in breaker_events
        if (event.payload or {}).get("error") == task.last_failure_error
        and (
            (event.payload or {}).get("failures") is None
            or (event.payload or {}).get("failures") == task.consecutive_failures
        )
    ]
    if (
        task.consecutive_failures <= 0
        or not task.last_failure_error
        or not matching_breaker_events
    ):
        return False, "preserved-state recovery requires durable operational failure evidence"

    failure_events = [
        {
            "kind": event.kind,
            "created_at": event.created_at,
            "payload": event.payload,
        }
        for event in matching_breaker_events
    ]

    evidence = {
        "actor": actor,
        "reason": reason,
        "prior_consecutive_failures": task.consecutive_failures,
        "prior_last_failure_error": task.last_failure_error,
        "failure_evidence_hash": _sha256(json.dumps(failure_events, sort_keys=True, default=str)),
        "recovered_at": int(time.time()),
    }
    ok, error = kb.promote_task(
        conn,
        task_id,
        actor=actor,
        reason=f"one preserved-state recovery: {reason}",
    )
    if not ok:
        return False, error or "promotion refused"
    _append(conn, task_id, "governance_preserved_recovery", evidence)
    return True, "recovery released"
