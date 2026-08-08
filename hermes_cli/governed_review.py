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
DELIVERY_RESERVATION_LEASE_SECONDS = 300
PRODUCT_RESERVATION_LEASE_SECONDS = 900
PAUSE_RESERVATION_LEASE_SECONDS = 120


def _default_rollback_procedure() -> dict[str, Any]:
    return {
        "target": "isolated candidate commit and governed-review scheduler templates",
        "steps": [
            {
                "order": 1,
                "action": "revert",
                "target": "isolated candidate commit",
                "max_attempts": 1,
            },
            {
                "order": 2,
                "action": "disable",
                "target": "governed-review scheduler templates",
                "max_attempts": 1,
            },
        ],
        "verification": {
            "condition": "candidate revert is recorded and scheduler templates are disabled",
            "evidence": [
                {
                    "method": "git_show",
                    "expected": "revert commit identifies the isolated candidate commit",
                },
                {
                    "method": "template_parse",
                    "expected": "all governed-review scheduler enabled fields are false",
                },
            ],
        },
    }


def _default_pause_procedure() -> dict[str, Any]:
    return {
        "target": "opt-in governed-review scheduler and Kanban governance event log",
        "steps": [
            {
                "order": 1,
                "action": "pause",
                "target": "opt-in governed-review scheduler",
                "max_attempts": 1,
            },
            {
                "order": 2,
                "action": "preserve",
                "target": "Kanban governance event log",
                "max_attempts": 1,
            },
        ],
        "verification": {
            "condition": "scheduler reports paused and the governance event log remains readable",
            "evidence": [
                {
                    "method": "scheduler_status",
                    "expected": "opt-in governed-review scheduler is paused",
                },
                {
                    "method": "kanban_event_query",
                    "expected": "governance reservation and outcome events are readable",
                },
            ],
        },
    }


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
    rollback_path: dict[str, Any] = field(default_factory=_default_rollback_procedure)
    pause_path: dict[str, Any] = field(default_factory=_default_pause_procedure)
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
                "gh",
                "api",
                "--paginate",
                "--slurp",
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
            "gh",
            "issue",
            "comment",
            str(target.issue),
            "--repo",
            target.repository,
            "--body",
            body,
        ],
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            (proc.stderr or "GitHub issue delivery failed").strip()[:500]
        )


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


def _pause_outcome_cas(
    conn,
    task_id: str,
    *,
    blocker_key: str,
    reservation_id: str,
    kind: str,
    payload: dict[str, Any],
) -> bool:
    """Append a truthful pause outcome only for the still-current owner."""
    with kb.write_txn(conn):
        matching = [
            event
            for event in _events(conn, task_id)
            if (event.payload or {}).get("blocker_key") == blocker_key
        ]
        reservation = next(
            (
                event
                for event in reversed(matching)
                if event.kind == "governance_scheduler_pause_reserved"
                and (event.payload or {}).get("reservation_id") == reservation_id
            ),
            None,
        )
        terminal_kinds = {
            "governance_scheduler_paused",
            "governance_scheduler_pause_failed",
            "governance_scheduler_pause_unavailable",
            "governance_scheduler_pause_reconciliation_required",
        }
        if reservation is None or any(
            event.kind in terminal_kinds for event in matching
        ):
            return False
        kb._append_event(conn, task_id, kind, payload)
        return True


def _pause_once(
    conn,
    task_id: str,
    blocker_key: str,
    reason: str,
    pause_scheduler: Optional[PauseCallback],
) -> str:
    """Reserve one pause executor and record only its confirmed outcome."""
    now = int(time.time())
    reservation_id = uuid.uuid4().hex
    executor = (
        getattr(pause_scheduler, "__qualname__", None)
        or getattr(pause_scheduler, "__name__", None)
        or type(pause_scheduler).__name__
        if pause_scheduler is not None
        else None
    )
    with kb.write_txn(conn):
        matching = [
            event
            for event in _events(conn, task_id)
            if (event.payload or {}).get("blocker_key") == blocker_key
            and event.kind.startswith("governance_scheduler_pause")
        ]
        if any(event.kind == "governance_scheduler_paused" for event in matching):
            return "paused"
        if any(
            event.kind
            in {
                "governance_scheduler_pause_failed",
                "governance_scheduler_pause_unavailable",
                "governance_scheduler_pause_reconciliation_required",
            }
            for event in matching
        ):
            return "failed_closed"
        reservation = next(
            (
                event
                for event in reversed(matching)
                if event.kind == "governance_scheduler_pause_reserved"
            ),
            None,
        )
        if reservation is not None:
            lease_expires_at = int(
                (reservation.payload or {}).get("lease_expires_at") or 0
            )
            if now < lease_expires_at:
                return "in_flight"
            kb._append_event(
                conn,
                task_id,
                "governance_scheduler_pause_reconciliation_required",
                {
                    "blocker_key": blocker_key,
                    "reason": "pause executor reservation expired without a durable outcome",
                    "reservation_id": (reservation.payload or {}).get("reservation_id"),
                },
            )
            return "reconciliation_required"
        kb._append_event(
            conn,
            task_id,
            "governance_scheduler_pause_reserved",
            {
                "blocker_key": blocker_key,
                "reason": reason,
                "reservation_id": reservation_id,
                "reserved_at": now,
                "lease_expires_at": now + PAUSE_RESERVATION_LEASE_SECONDS,
                "executor": executor,
            },
        )

    base_payload = {
        "blocker_key": blocker_key,
        "reason": reason,
        "reservation_id": reservation_id,
        "executor": executor,
    }
    if pause_scheduler is None:
        _pause_outcome_cas(
            conn,
            task_id,
            blocker_key=blocker_key,
            reservation_id=reservation_id,
            kind="governance_scheduler_pause_unavailable",
            payload={**base_payload, "error": "no pause executor was provided"},
        )
        return "unavailable"
    try:
        pause_scheduler(reason)
    except Exception as exc:
        _pause_outcome_cas(
            conn,
            task_id,
            blocker_key=blocker_key,
            reservation_id=reservation_id,
            kind="governance_scheduler_pause_failed",
            payload={**base_payload, "error": str(exc)[:500]},
        )
        return "failed"
    recorded = _pause_outcome_cas(
        conn,
        task_id,
        blocker_key=blocker_key,
        reservation_id=reservation_id,
        kind="governance_scheduler_paused",
        payload=base_payload,
    )
    return "paused" if recorded else "reconciliation_required"


def reconcile_scheduler_pause(
    conn,
    task_id: str,
    *,
    blocker_key: str,
    actor: str,
    actual_outcome: str,
    evidence: str,
) -> tuple[bool, str]:
    """Audit the actual outcome of an expired pause executor reservation."""
    if actual_outcome not in {"paused", "not_paused"}:
        return False, "actual_outcome must be 'paused' or 'not_paused'"
    if not actor.strip() or not evidence.strip():
        return False, "actor and reconciliation evidence are required"
    with kb.write_txn(conn):
        events = [
            event
            for event in _events(conn, task_id)
            if (event.payload or {}).get("blocker_key") == blocker_key
            and event.kind.startswith("governance_scheduler_pause")
        ]
        reservation = next(
            (
                event
                for event in reversed(events)
                if event.kind == "governance_scheduler_pause_reserved"
            ),
            None,
        )
        if reservation is None:
            return False, "pause reservation not found"
        reservation_id = str((reservation.payload or {}).get("reservation_id") or "")
        if not any(
            event.kind == "governance_scheduler_pause_reconciliation_required"
            and (event.payload or {}).get("reservation_id") == reservation_id
            for event in events
        ):
            return False, "pause is not awaiting reconciliation"
        if any(
            event.kind == "governance_scheduler_pause_reconciled" for event in events
        ):
            return False, "pause reservation already reconciled"
        payload = {
            "blocker_key": blocker_key,
            "reservation_id": reservation_id,
            "actor": actor.strip(),
            "actual_outcome": actual_outcome,
            "evidence": evidence.strip(),
            "executor": (reservation.payload or {}).get("executor"),
            "reconciled_at": int(time.time()),
        }
        kb._append_event(
            conn, task_id, "governance_scheduler_pause_reconciled", payload
        )
        kb._append_event(
            conn,
            task_id,
            (
                "governance_scheduler_paused"
                if actual_outcome == "paused"
                else "governance_scheduler_pause_failed"
            ),
            {**payload, "manual_reconciliation": True},
        )
    return True, "scheduler pause reconciled"


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
    material = "|".join([
        PROTOCOL_VERSION,
        spec.task_id,
        spec.expected_commit,
        f"{spec.delivery.repository}#{spec.delivery.issue}",
        prompt_hash,
    ])
    return f"governed-review:{_sha256(material)}"


def _validate_procedure(label: str, value: Any) -> Optional[str]:
    """Validate a bounded, machine-checkable operational procedure contract."""
    error = f"{label} procedure must be a structured, bounded operational procedure"
    if not isinstance(value, dict):
        return error

    placeholders = {
        "",
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
        "somehow",
    }
    negations = (
        "do not ",
        "don't ",
        "dont ",
        "never ",
        "no action",
        "without ",
        "not execute",
        "not perform",
        "not required",
    )

    def concrete(raw: Any) -> bool:
        if not isinstance(raw, str):
            return False
        normalized = " ".join(raw.casefold().split()).strip(" .;:-_")
        return bool(
            normalized
            and normalized not in placeholders
            and len(normalized) >= 8
            and not any(phrase in normalized for phrase in negations)
        )

    target = value.get("target")
    if not concrete(target):
        return error
    target_text = str(target).casefold()
    relevant_terms = {
        "rollback": (
            "commit",
            "candidate",
            "template",
            "change",
            "release",
            "deployment",
        ),
        "pause": ("scheduler", "job", "worker", "dispatch", "gateway", "event log"),
    }
    if not any(term in target_text for term in relevant_terms.get(label, ())):
        return error

    allowed_actions = {
        "rollback": {"revert", "restore", "disable", "preserve", "block", "stop"},
        "pause": {"pause", "disable", "stop", "preserve", "block"},
    }
    steps = value.get("steps")
    if not isinstance(steps, list) or not 1 <= len(steps) <= 6:
        return error
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            return error
        if step.get("order") != index:
            return error
        if step.get("action") not in allowed_actions.get(label, set()):
            return error
        if not concrete(step.get("target")):
            return error
        limit = step.get("max_attempts")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 3:
            return error

    verification = value.get("verification")
    if not isinstance(verification, dict) or not concrete(
        verification.get("condition")
    ):
        return error
    evidence = verification.get("evidence")
    if not isinstance(evidence, list) or not 1 <= len(evidence) <= 6:
        return error
    for item in evidence:
        if not isinstance(item, dict):
            return error
        if not concrete(item.get("method")) or not concrete(item.get("expected")):
            return error
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


def _payload_matches_identity(
    payload: Optional[dict[str, Any]], identity: dict[str, Any]
) -> bool:
    return bool(payload) and all(
        payload.get(key) == value for key, value in identity.items()
    )


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
        row["id"] for row in parent_rows if row["status"] not in ("done", "archived")
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
            errors.append(
                f"durable GitHub issue delivery is not ready: {readiness_detail}"
            )

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
    identity.update({
        "actual_profile": task.assignee,
        "requested_skills": list(task.skills or []),
    })
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
        _sticky_block_once(
            conn, spec.task_id, "governance-preflight: " + "; ".join(errors)
        )
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
    return kb.block_task(
        conn, task_id, reason=reason, expected_run_id=task.current_run_id
    )


def release_one_child(
    conn, parent_id: str, child_id: str, *, actor: str
) -> tuple[bool, Optional[str]]:
    """Atomically release one child only while its exact governed hold is current."""
    hold_reason = f"governance-child-hold:{parent_id}"
    with kb.write_txn(conn):
        relation = conn.execute(
            "SELECT p.status AS parent_status, c.status AS child_status "
            "FROM task_links l "
            "JOIN tasks p ON p.id=l.parent_id "
            "JOIN tasks c ON c.id=l.child_id "
            "WHERE l.parent_id=? AND l.child_id=?",
            (parent_id, child_id),
        ).fetchone()
        if relation is None:
            return False, f"{child_id} is not a child of {parent_id}"
        if relation["parent_status"] not in ("done", "archived"):
            return False, f"parent {parent_id} is not terminal"
        if relation["child_status"] != "blocked":
            return (
                False,
                f"child {child_id} is not held for governed release from {parent_id}",
            )

        state_events = [
            event
            for event in _events(conn, child_id)
            if event.kind in ("blocked", "unblocked")
        ]
        active_hold = state_events[-1] if state_events else None
        if (
            active_hold is None
            or active_hold.kind != "blocked"
            or str((active_hold.payload or {}).get("reason") or "") != hold_reason
        ):
            return (
                False,
                f"child {child_id} is not held for governed release from {parent_id}",
            )

        cur = conn.execute(
            "UPDATE tasks SET status='ready' "
            "WHERE id=? AND status='blocked' "
            "AND EXISTS (SELECT 1 FROM task_links l JOIN tasks p ON p.id=l.parent_id "
            "            WHERE l.parent_id=? AND l.child_id=tasks.id "
            "              AND p.status IN ('done','archived')) "
            "AND (SELECT id FROM task_events "
            "     WHERE task_id=tasks.id AND kind IN ('blocked','unblocked') "
            "     ORDER BY created_at DESC, id DESC LIMIT 1)=?",
            (child_id, parent_id, active_hold.id),
        )
        if cur.rowcount != 1:
            return False, f"child {child_id} hold changed during governed release"
        reason = f"explicit governed release from {parent_id}"
        kb._append_event(
            conn,
            child_id,
            "promoted_manual",
            {"actor": actor, "reason": reason, "forced": False},
        )
        kb._append_event(
            conn,
            child_id,
            "unblocked",
            {"status": "ready", "reason": reason, "hold_event_id": active_hold.id},
        )
        kb._append_event(
            conn,
            parent_id,
            "governance_child_released",
            {"child_id": child_id, "actor": actor, "hold_event_id": active_hold.id},
        )
    return True, None


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
    if not profile_ready and not any(
        "profile resolution failed" in error for error in errors
    ):
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
    identity.update({
        "actual_profile": task.assignee,
        "requested_skills": list(task.skills or []),
    })
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
        row["id"] for row in parent_rows if row["status"] not in ("done", "archived")
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
            (index for index, event in state_events if event.kind == "unblocked"),
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
                errors.append(
                    "immutable worktree is not clean, including untracked material"
                )

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


def _reserve_product_execution(
    conn, task_id: str, dedupe: str
) -> tuple[str, Optional[str]]:
    """Reserve one leased product owner or create an expired-owner barrier."""
    now = int(time.time())
    reservation_id = uuid.uuid4().hex
    with kb.write_txn(conn):
        matching = [
            event
            for event in _events(conn, task_id)
            if (event.payload or {}).get("dedupe_key") == dedupe
            and event.kind.startswith("governance_product_")
        ]
        if any(event.kind == "governance_product_completed" for event in matching):
            return "completed", None
        if any(event.kind == "governance_product_failed" for event in matching):
            return "failed", None
        unresolved_barrier = any(
            event.kind == "governance_product_reconciliation_required"
            for event in matching
        ) and not any(
            event.kind == "governance_product_reconciled" for event in matching
        )
        if unresolved_barrier:
            return "reconciliation", None
        reservation = next(
            (
                event
                for event in reversed(matching)
                if event.kind == "governance_product_reserved"
            ),
            None,
        )
        if reservation is not None:
            payload = reservation.payload or {}
            if now < int(payload.get("lease_expires_at") or 0):
                return "in_flight", None
            kb._append_event(
                conn,
                task_id,
                "governance_product_reconciliation_required",
                {
                    "dedupe_key": dedupe,
                    "reservation_id": payload.get("reservation_id"),
                    "reason": "product reservation expired without a durable outcome",
                },
            )
            return "reconciliation", None
        kb._append_event(
            conn,
            task_id,
            "governance_product_reserved",
            {
                "dedupe_key": dedupe,
                "reservation_id": reservation_id,
                "reserved_at": now,
                "lease_expires_at": now + PRODUCT_RESERVATION_LEASE_SECONDS,
            },
        )
    return "owner", reservation_id


def _record_product_outcome(
    conn,
    task_id: str,
    *,
    dedupe: str,
    reservation_id: str,
    kind: str,
    payload: dict[str, Any],
) -> bool:
    """Owner-token CAS for product success/failure; expiry becomes a barrier."""
    now = int(time.time())
    with kb.write_txn(conn):
        matching = [
            event
            for event in _events(conn, task_id)
            if (event.payload or {}).get("dedupe_key") == dedupe
            and event.kind.startswith("governance_product_")
        ]
        reservation = next(
            (
                event
                for event in reversed(matching)
                if event.kind == "governance_product_reserved"
                and (event.payload or {}).get("reservation_id") == reservation_id
            ),
            None,
        )
        if reservation is None:
            return False
        if any(
            event.kind
            in {
                "governance_product_completed",
                "governance_product_failed",
                "governance_product_reconciliation_required",
            }
            for event in matching
        ):
            return False
        if now >= int((reservation.payload or {}).get("lease_expires_at") or 0):
            kb._append_event(
                conn,
                task_id,
                "governance_product_reconciliation_required",
                {
                    "dedupe_key": dedupe,
                    "reservation_id": reservation_id,
                    "reason": "product owner lease expired before outcome recording",
                },
            )
            return False
        kb._append_event(conn, task_id, kind, payload)
        return True


def _reserve_delivery_attempt(
    conn,
    task_id: str,
    dedupe: str,
) -> tuple[str, Optional[int], Optional[str]]:
    """Reserve one leased sender; live contenders wait and expired owners reconcile."""
    now = int(time.time())
    reservation_id = uuid.uuid4().hex
    with kb.write_txn(conn):
        matching = [
            event
            for event in _events(conn, task_id)
            if (event.payload or {}).get("dedupe_key") == dedupe
            and event.kind.startswith("governance_delivery_")
        ]
        if any(event.kind == "governance_delivery_succeeded" for event in matching):
            return "succeeded", None, None

        reservations = [
            event
            for event in matching
            if event.kind == "governance_delivery_reserved"
            and isinstance((event.payload or {}).get("attempt"), int)
            and bool((event.payload or {}).get("reservation_id"))
        ]
        for reservation in reservations:
            payload = reservation.payload or {}
            owner = payload.get("reservation_id")
            outcomes = [
                event
                for event in matching
                if event.kind
                in ("governance_delivery_failed", "governance_delivery_succeeded")
                and (event.payload or {}).get("reservation_id") == owner
            ]
            reconciled = any(
                event.kind == "governance_delivery_reconciled"
                and (event.payload or {}).get("reservation_id") == owner
                for event in matching
            )
            barrier = any(
                event.kind == "governance_delivery_reconciliation_required"
                and (event.payload or {}).get("reservation_id") == owner
                for event in matching
            )
            if barrier and not reconciled:
                return "reconciliation", int(payload["attempt"]), None
            if outcomes:
                continue
            if now < int(payload.get("lease_expires_at") or 0):
                return "in_flight", int(payload["attempt"]), None
            kb._append_event(
                conn,
                task_id,
                "governance_delivery_reconciliation_required",
                {
                    "dedupe_key": dedupe,
                    "attempt": int(payload["attempt"]),
                    "reservation_id": owner,
                    "reason": "delivery reservation expired without a durable outcome",
                },
            )
            return "reconciliation", int(payload["attempt"]), None

        if len(reservations) >= MAX_DELIVERY_ATTEMPTS:
            return "exhausted", None, None
        attempt = (
            max(
                (
                    int((event.payload or {}).get("attempt") or 0)
                    for event in reservations
                ),
                default=0,
            )
            + 1
        )
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
                "reserved_at": now,
                "lease_expires_at": now + DELIVERY_RESERVATION_LEASE_SECONDS,
            },
        )
    return "owner", attempt, reservation_id


def _record_delivery_outcome(
    conn,
    task_id: str,
    *,
    dedupe: str,
    attempt: int,
    reservation_id: str,
    kind: str,
    payload: dict[str, Any],
) -> bool:
    """Owner-token CAS for a delivery outcome; stale owners cannot finalize."""
    now = int(time.time())
    with kb.write_txn(conn):
        matching = [
            event
            for event in _events(conn, task_id)
            if (event.payload or {}).get("dedupe_key") == dedupe
            and event.kind.startswith("governance_delivery_")
        ]
        reservation = next(
            (
                event
                for event in reversed(matching)
                if event.kind == "governance_delivery_reserved"
                and (event.payload or {}).get("attempt") == attempt
                and (event.payload or {}).get("reservation_id") == reservation_id
            ),
            None,
        )
        if reservation is None:
            return False
        if any(
            event.kind
            in {
                "governance_delivery_failed",
                "governance_delivery_succeeded",
                "governance_delivery_reconciliation_required",
            }
            and (event.payload or {}).get("reservation_id") == reservation_id
            for event in matching
        ):
            return False
        if now >= int((reservation.payload or {}).get("lease_expires_at") or 0):
            kb._append_event(
                conn,
                task_id,
                "governance_delivery_reconciliation_required",
                {
                    "dedupe_key": dedupe,
                    "attempt": attempt,
                    "reservation_id": reservation_id,
                    "reason": "delivery owner lease expired before outcome recording",
                },
            )
            return False
        kb._append_event(conn, task_id, kind, payload)
        return True


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
        recorded = {(event.payload or {}).get("child_id") for event in parent_events}
        for child in children:
            child_id = child["id"]
            status = child["status"]
            preexisting_reason = _active_block_reason(_events(conn, child_id))
            if status in ("todo", "ready"):
                cur = conn.execute(
                    "UPDATE tasks SET status='blocked' WHERE id=? AND status=?",
                    (child_id, status),
                )
                if cur.rowcount != 1:
                    errors.append(
                        f"child {child_id} changed while applying governed hold"
                    )
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


def _record_product_reconciliation(
    conn,
    spec: GovernedReviewSpec,
    *,
    dedupe: str,
    pause_scheduler: Optional[PauseCallback],
) -> None:
    _sticky_block_once(
        conn,
        spec.task_id,
        "product-unknown: manual reconciliation required",
    )
    _pause_once(
        conn,
        spec.task_id,
        f"product-unknown:{dedupe}",
        "product outcome is unknown; manual reconciliation required",
        pause_scheduler,
    )


def _record_delivery_reconciliation(
    conn,
    spec: GovernedReviewSpec,
    *,
    dedupe: str,
    attempt: int,
    pause_scheduler: Optional[PauseCallback],
) -> None:
    """Apply the task/pause barrier after the reservation txn records ambiguity."""
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


def reconcile_delivery_attempt(
    conn,
    task_id: str,
    *,
    dedupe: str,
    attempt: int,
    actor: str,
    actual_outcome: str,
    evidence: str,
    target: Optional[str] = None,
) -> tuple[bool, str]:
    """Explicitly resolve an unknown delivery after audited external inspection."""
    if actual_outcome not in {"delivered", "not_delivered"}:
        return False, "actual_outcome must be 'delivered' or 'not_delivered'"
    if not actor.strip() or not evidence.strip():
        return False, "actor and reconciliation evidence are required"
    with kb.write_txn(conn):
        events = [
            event
            for event in _events(conn, task_id)
            if (event.payload or {}).get("dedupe_key") == dedupe
            and event.kind.startswith("governance_delivery_")
        ]
        reservation = next(
            (
                event
                for event in reversed(events)
                if event.kind == "governance_delivery_reserved"
                and (event.payload or {}).get("attempt") == attempt
            ),
            None,
        )
        if reservation is None:
            return False, "delivery reservation not found"
        reservation_id = str((reservation.payload or {}).get("reservation_id") or "")
        barrier = any(
            event.kind == "governance_delivery_reconciliation_required"
            and (event.payload or {}).get("reservation_id") == reservation_id
            for event in events
        )
        if not barrier:
            return False, "delivery attempt is not awaiting reconciliation"
        if any(
            event.kind
            in {
                "governance_delivery_reconciled",
                "governance_delivery_succeeded",
                "governance_delivery_failed",
            }
            and (event.payload or {}).get("reservation_id") == reservation_id
            for event in events
        ):
            return False, "delivery attempt already has a durable outcome"
        active_reason = _active_block_reason(_events(conn, task_id))
        if active_reason != "delivery-unknown: manual reconciliation required":
            return (
                False,
                "delivery reconciliation refuses a replaced or substantive hold",
            )

        audit = {
            "dedupe_key": dedupe,
            "attempt": attempt,
            "reservation_id": reservation_id,
            "actor": actor.strip(),
            "actual_outcome": actual_outcome,
            "evidence": evidence.strip(),
            "reconciled_at": int(time.time()),
        }
        kb._append_event(conn, task_id, "governance_delivery_reconciled", audit)
        if actual_outcome == "delivered":
            kb._append_event(
                conn,
                task_id,
                "governance_delivery_succeeded",
                {**audit, "target": target, "manual_reconciliation": True},
            )
        else:
            kb._append_event(
                conn,
                task_id,
                "governance_delivery_failed",
                {
                    **audit,
                    "error": "manual reconciliation confirmed delivery did not occur",
                    "delivery_only_retry_available": attempt < MAX_DELIVERY_ATTEMPTS,
                    "manual_reconciliation": True,
                },
            )
        cur = conn.execute(
            "UPDATE tasks SET status='ready' WHERE id=? AND status='blocked'",
            (task_id,),
        )
        if cur.rowcount != 1:
            raise RuntimeError("task status changed during delivery reconciliation")
        kb._append_event(
            conn,
            task_id,
            "unblocked",
            {"status": "ready", "reason": "audited delivery reconciliation"},
        )
    return True, "delivery reconciled"


def reconcile_product_execution(
    conn,
    task_id: str,
    *,
    dedupe: str,
    actor: str,
    actual_outcome: str,
    evidence: dict[str, Any],
) -> tuple[bool, str]:
    """Explicitly resolve an expired product reservation without rerunning work."""
    if actual_outcome not in {"completed", "failed"}:
        return False, "actual_outcome must be 'completed' or 'failed'"
    if not actor.strip() or not isinstance(evidence, dict) or not evidence:
        return False, "actor and structured reconciliation evidence are required"
    with kb.write_txn(conn):
        events = [
            event
            for event in _events(conn, task_id)
            if (event.payload or {}).get("dedupe_key") == dedupe
            and event.kind.startswith("governance_product_")
        ]
        reservation = next(
            (
                event
                for event in reversed(events)
                if event.kind == "governance_product_reserved"
            ),
            None,
        )
        if reservation is None:
            return False, "product reservation not found"
        reservation_id = str((reservation.payload or {}).get("reservation_id") or "")
        if not any(
            event.kind == "governance_product_reconciliation_required"
            and (event.payload or {}).get("reservation_id") == reservation_id
            for event in events
        ):
            return False, "product execution is not awaiting reconciliation"
        if any(
            event.kind
            in {
                "governance_product_reconciled",
                "governance_product_completed",
                "governance_product_failed",
            }
            for event in events
        ):
            return False, "product execution already has a durable outcome"
        active_reason = _active_block_reason(_events(conn, task_id))
        if active_reason != "product-unknown: manual reconciliation required":
            return (
                False,
                "product reconciliation refuses a replaced or substantive hold",
            )
        audit = {
            "dedupe_key": dedupe,
            "reservation_id": reservation_id,
            "actor": actor.strip(),
            "actual_outcome": actual_outcome,
            "evidence": evidence,
            "reconciled_at": int(time.time()),
        }
        kb._append_event(conn, task_id, "governance_product_reconciled", audit)
        if actual_outcome == "completed":
            kb._append_event(
                conn,
                task_id,
                "governance_product_completed",
                {
                    **audit,
                    "completed_at": int(time.time()),
                    "manual_reconciliation": True,
                },
            )
            cur = conn.execute(
                "UPDATE tasks SET status='ready' WHERE id=? AND status='blocked'",
                (task_id,),
            )
            if cur.rowcount != 1:
                raise RuntimeError("task status changed during product reconciliation")
            kb._append_event(
                conn,
                task_id,
                "unblocked",
                {"status": "ready", "reason": "audited product reconciliation"},
            )
        else:
            kb._append_event(
                conn,
                task_id,
                "governance_product_failed",
                {**audit, "error": "manual reconciliation confirmed product failure"},
            )
    return True, "product reconciled"


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
        elif product_state == "in_flight":
            return False
        elif product_state == "reconciliation":
            _record_product_reconciliation(
                conn,
                spec,
                dedupe=dedupe,
                pause_scheduler=pause_scheduler,
            )
            return False
        elif product_state != "owner" or product_reservation_id is None:
            return False
        else:
            try:
                evidence = product_work()
            except Exception as exc:
                blocker_key = f"product-failed:{dedupe}"
                recorded = _record_product_outcome(
                    conn,
                    spec.task_id,
                    dedupe=dedupe,
                    reservation_id=product_reservation_id,
                    kind="governance_product_failed",
                    payload={
                        "dedupe_key": dedupe,
                        "reservation_id": product_reservation_id,
                        "error": str(exc)[:500],
                    },
                )
                if not recorded:
                    _record_product_reconciliation(
                        conn, spec, dedupe=dedupe, pause_scheduler=pause_scheduler
                    )
                    return False
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
                recorded = _record_product_outcome(
                    conn,
                    spec.task_id,
                    dedupe=dedupe,
                    reservation_id=product_reservation_id,
                    kind="governance_product_failed",
                    payload={
                        "dedupe_key": dedupe,
                        "reservation_id": product_reservation_id,
                        "error": "product evidence was not a dict",
                    },
                )
                if not recorded:
                    _record_product_reconciliation(
                        conn, spec, dedupe=dedupe, pause_scheduler=pause_scheduler
                    )
                    return False
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
            recorded = _record_product_outcome(
                conn,
                spec.task_id,
                dedupe=dedupe,
                reservation_id=product_reservation_id,
                kind="governance_product_completed",
                payload={
                    "dedupe_key": dedupe,
                    "reservation_id": product_reservation_id,
                    "prompt_hash": prompt_hash,
                    "evidence": evidence,
                    "completed_at": int(time.time()),
                },
            )
            if not recorded:
                _record_product_reconciliation(
                    conn, spec, dedupe=dedupe, pause_scheduler=pause_scheduler
                )
                return False
            product_events = [
                event
                for event in _events(conn, spec.task_id, "governance_product_completed")
                if (event.payload or {}).get("dedupe_key") == dedupe
            ]

    if not success_events:
        delivery_state, attempt, delivery_reservation_id = _reserve_delivery_attempt(
            conn, spec.task_id, dedupe
        )
        if delivery_state == "in_flight":
            return False
        if delivery_state == "reconciliation":
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
                for event in _events(
                    conn, spec.task_id, "governance_delivery_succeeded"
                )
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
                recorded = _record_delivery_outcome(
                    conn,
                    spec.task_id,
                    dedupe=dedupe,
                    attempt=attempt,
                    reservation_id=delivery_reservation_id,
                    kind="governance_delivery_failed",
                    payload={
                        "dedupe_key": dedupe,
                        "attempt": attempt,
                        "reservation_id": delivery_reservation_id,
                        "error": str(exc)[:500],
                        "delivery_only_retry_available": attempt
                        < MAX_DELIVERY_ATTEMPTS,
                    },
                )
                if not recorded:
                    _record_delivery_reconciliation(
                        conn,
                        spec,
                        dedupe=dedupe,
                        attempt=attempt,
                        pause_scheduler=pause_scheduler,
                    )
                    return False
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
            recorded = _record_delivery_outcome(
                conn,
                spec.task_id,
                dedupe=dedupe,
                attempt=attempt,
                reservation_id=delivery_reservation_id,
                kind="governance_delivery_succeeded",
                payload={
                    "dedupe_key": dedupe,
                    "attempt": attempt,
                    "reservation_id": delivery_reservation_id,
                    "target": f"{spec.delivery.repository}#{spec.delivery.issue}",
                },
            )
            if not recorded:
                _record_delivery_reconciliation(
                    conn,
                    spec,
                    dedupe=dedupe,
                    attempt=attempt,
                    pause_scheduler=pause_scheduler,
                )
                return False
            success_events = [
                event
                for event in _events(
                    conn, spec.task_id, "governance_delivery_succeeded"
                )
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
    """Atomically consume the one recovery while promoting the preserved task."""
    exhausted = False
    try:
        with kb.write_txn(conn):
            row = conn.execute(
                "SELECT status, consecutive_failures, last_failure_error "
                "FROM tasks WHERE id=?",
                (task_id,),
            ).fetchone()
            if row is None:
                return False, "task not found"
            events = _events(conn, task_id)
            if any(event.kind == "governance_preserved_recovery" for event in events):
                exhausted = True
            elif row["status"] != "blocked":
                return False, f"task must be blocked, got {row['status']}"
            else:
                active_block_reason = _active_block_reason(events)
                if active_block_reason is not None:
                    return False, (
                        "preserved-state recovery refuses substantive hold: "
                        f"{active_block_reason}"
                    )
                breaker_events = [event for event in events if event.kind == "gave_up"]
                matching_breaker_events = [
                    event
                    for event in breaker_events
                    if (event.payload or {}).get("error") == row["last_failure_error"]
                    and (
                        (event.payload or {}).get("failures") is None
                        or (event.payload or {}).get("failures")
                        == row["consecutive_failures"]
                    )
                ]
                if (
                    int(row["consecutive_failures"] or 0) <= 0
                    or not row["last_failure_error"]
                    or not matching_breaker_events
                ):
                    return False, (
                        "preserved-state recovery requires durable operational failure evidence"
                    )
                unsatisfied = conn.execute(
                    "SELECT p.id FROM tasks p JOIN task_links l ON l.parent_id=p.id "
                    "WHERE l.child_id=? AND p.status NOT IN ('done','archived')",
                    (task_id,),
                ).fetchall()
                if unsatisfied:
                    return (
                        False,
                        "preserved-state recovery has unsatisfied parent dependencies",
                    )

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
                    "prior_consecutive_failures": int(row["consecutive_failures"]),
                    "prior_last_failure_error": row["last_failure_error"],
                    "failure_evidence_hash": _sha256(
                        json.dumps(failure_events, sort_keys=True, default=str)
                    ),
                    "recovered_at": int(time.time()),
                }
                cur = conn.execute(
                    "UPDATE tasks SET status='ready' "
                    "WHERE id=? AND status='blocked' "
                    "AND consecutive_failures=? AND last_failure_error=?",
                    (
                        task_id,
                        int(row["consecutive_failures"]),
                        row["last_failure_error"],
                    ),
                )
                if cur.rowcount != 1:
                    return False, "task breaker state changed during preserved recovery"
                kb._append_event(
                    conn,
                    task_id,
                    "promoted_manual",
                    {
                        "actor": actor,
                        "reason": f"one preserved-state recovery: {reason}",
                        "forced": False,
                    },
                )
                kb._append_event(
                    conn,
                    task_id,
                    "governance_preserved_recovery",
                    evidence,
                )
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise

    if exhausted:
        blocker_key = f"recovery-exhausted:{task_id}"
        _pause_once(
            conn,
            task_id,
            blocker_key,
            "second preserved-state recovery rejected",
            pause_scheduler,
        )
        return False, "preserved-state recovery already used"
    return True, "recovery released"
