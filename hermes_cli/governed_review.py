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
    issue = subprocess.run(
        [
            "gh", "issue", "view", str(target.issue),
            "--repo", target.repository, "--json", "number",
        ],
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=15,
        check=False,
    )
    if issue.returncode != 0:
        return False, "configured GitHub issue is not readable"
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


def _pause_once(
    conn,
    task_id: str,
    blocker_key: str,
    reason: str,
    pause_scheduler: Optional[PauseCallback],
) -> None:
    prior = [
        event for event in _events(conn, task_id, "governance_scheduler_paused")
        if (event.payload or {}).get("blocker_key") == blocker_key
    ]
    if prior:
        return
    if pause_scheduler is not None:
        pause_scheduler(reason)
    _append(
        conn,
        task_id,
        "governance_scheduler_paused",
        {"blocker_key": blocker_key, "reason": reason},
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
        errors.append(f"task {spec.task_id} does not exist")
    else:
        evidence["actual_profile"] = task.assignee
        evidence["requested_skills"] = list(task.skills or [])
        missing: list[str] = []
        if not task.assignee or not profile_resolver(task.assignee):
            errors.append(f"actual assignee profile does not exist: {task.assignee!r}")
        else:
            missing = [
                skill for skill in (task.skills or [])
                if not skill_resolver(task.assignee, skill)
            ]
        if missing:
            errors.append(f"requested skills do not resolve for {task.assignee}: {', '.join(missing)}")

        parent_rows = conn.execute(
            "SELECT p.id, p.status FROM tasks p JOIN task_links l ON l.parent_id=p.id "
            "WHERE l.child_id=? ORDER BY p.id",
            (spec.task_id,),
        ).fetchall()
        evidence["dependencies"] = [dict(row) for row in parent_rows]
        open_parents = [row["id"] for row in parent_rows if row["status"] not in ("done", "archived")]
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
        status = _run_git(workspace, "status", "--porcelain=v1", "--untracked-files=no")
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
                errors.append("tracked or staged worktree state is not clean")

    result = PreflightResult(not errors, tuple(errors), evidence, dedupe)
    if result.ok:
        prior = [
            event for event in _events(conn, spec.task_id, "governance_preflight_passed")
            if (event.payload or {}).get("dedupe_key") == dedupe
        ]
        if not prior:
            _append(conn, spec.task_id, "governance_preflight_passed", evidence)
        return result

    blocker_key = f"preflight:{dedupe}"
    prior = [
        event for event in _events(conn, spec.task_id, "governance_preflight_blocked")
        if (event.payload or {}).get("blocker_key") == blocker_key
    ]
    if not prior:
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
    """Release exactly one linked child after all of its parents are terminal."""
    if child_id not in kb.child_ids(conn, parent_id):
        return False, f"{child_id} is not a child of {parent_id}"
    parent = kb.get_task(conn, parent_id)
    if parent is None or parent.status not in ("done", "archived"):
        return False, f"parent {parent_id} is not terminal"
    ok, error = kb.promote_task(
        conn,
        child_id,
        actor=actor,
        reason=f"explicit governed release from {parent_id}",
    )
    if ok:
        _append(
            conn,
            parent_id,
            "governance_child_released",
            {"child_id": child_id, "actor": actor},
        )
    return ok, error


def complete_with_issue_delivery(
    conn,
    spec: GovernedReviewSpec,
    *,
    summary: str,
    product_work: ProductCallback,
    sender: DeliverySender = _default_delivery_sender,
    pause_scheduler: Optional[PauseCallback] = None,
) -> bool:
    """Persist product evidence, deliver once, then complete without fan-out.

    A failed first delivery sticky-blocks completion. A second invocation is a
    delivery-only retry because the durable product event suppresses rerunning
    ``product_work``. No more than one retry is allowed.
    """
    prompt_hash = _sha256(spec.prompt)
    dedupe = _dedupe_key(spec, prompt_hash)
    success_events = [
        event for event in _events(conn, spec.task_id, "governance_delivery_succeeded")
        if (event.payload or {}).get("dedupe_key") == dedupe
    ]
    if kb.get_task(conn, spec.task_id) and kb.get_task(conn, spec.task_id).status == "done":
        return bool(success_events)

    product_events = [
        event for event in _events(conn, spec.task_id, "governance_product_completed")
        if (event.payload or {}).get("dedupe_key") == dedupe
    ]
    if not product_events:
        try:
            evidence = product_work()
        except Exception as exc:
            blocker_key = f"product-failed:{dedupe}"
            _append(
                conn,
                spec.task_id,
                "governance_product_failed",
                {"dedupe_key": dedupe, "error": str(exc)[:500]},
            )
            _sticky_block_once(conn, spec.task_id, f"product-failed: {str(exc)[:300]}")
            _pause_once(conn, spec.task_id, blocker_key, "product work failed", pause_scheduler)
            return False
        if not isinstance(evidence, dict):
            blocker_key = f"product-evidence-invalid:{dedupe}"
            _append(
                conn,
                spec.task_id,
                "governance_product_failed",
                {"dedupe_key": dedupe, "error": "product evidence was not a dict"},
            )
            _sticky_block_once(conn, spec.task_id, "product-failed: invalid evidence shape")
            _pause_once(conn, spec.task_id, blocker_key, "product evidence invalid", pause_scheduler)
            return False
        _append(
            conn,
            spec.task_id,
            "governance_product_completed",
            {
                "dedupe_key": dedupe,
                "prompt_hash": prompt_hash,
                "evidence": evidence,
                "completed_at": int(time.time()),
            },
        )
        product_events = _events(conn, spec.task_id, "governance_product_completed")

    if not success_events:
        attempts = [
            event for event in _events(conn, spec.task_id)
            if event.kind in ("governance_delivery_failed", "governance_delivery_succeeded")
            and (event.payload or {}).get("dedupe_key") == dedupe
        ]
        if len(attempts) >= MAX_DELIVERY_ATTEMPTS:
            blocker_key = f"delivery-exhausted:{dedupe}"
            _sticky_block_once(conn, spec.task_id, "delivery-exhausted: manual intervention required")
            _pause_once(conn, spec.task_id, blocker_key, "delivery retry exhausted", pause_scheduler)
            return False
        attempt = len(attempts) + 1
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
                    "error": str(exc)[:500],
                    "delivery_only_retry_available": attempt < MAX_DELIVERY_ATTEMPTS,
                },
            )
            _sticky_block_once(conn, spec.task_id, f"delivery-failed: {str(exc)[:300]}")
            return False
        _append(
            conn,
            spec.task_id,
            "governance_delivery_succeeded",
            {
                "dedupe_key": dedupe,
                "attempt": attempt,
                "target": f"{spec.delivery.repository}#{spec.delivery.issue}",
            },
        )

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
    failure_events = [
        {
            "kind": event.kind,
            "created_at": event.created_at,
            "payload": event.payload,
        }
        for event in _events(conn, task_id)
        if event.kind in ("gave_up", "timed_out", "crashed", "spawn_failed")
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
