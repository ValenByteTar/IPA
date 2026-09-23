"""Dev-time work permits (Permit-to-Work) for parallel agent sessions.

Same lease discipline as PAT-007 (vram.lock / tier0.lock) applied to
development sessions: a permit authorizes a scope of paths for a bounded
time, lists the EKS records governing that scope as precautions, and must
be closed — the closeout is where session knowledge gets harvested.

State lives under `outputs/devin/permits/` (gitignored — coordination is
ephemeral state, not EKS knowledge).
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Iterable

try:
    from .eks_repository import EKSRepository, HOT_ZONE_THRESHOLD, glob_match, scopes_overlap
except ImportError:  # direct sys.path execution by scripts/tests
    from eks_repository import EKSRepository, HOT_ZONE_THRESHOLD, glob_match, scopes_overlap

PERMIT_TYPES = ("exclusive", "survey", "advisory")
DEFAULT_TTL_S = 7200

# Cross-process guard around the read-check-write of `acquire` (same lease
# discipline as PAT-007): without it two sessions can pass the conflict check
# simultaneously and both take an overlapping exclusive scope.
ACQUIRE_LOCK_NAME = ".acquire.lock"
ACQUIRE_LOCK_STALE_S = 30
ACQUIRE_LOCK_TIMEOUT_S = 5.0

_ID_RE = re.compile(r"^PW-\d{8}-\d{2,}$")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_ts(value: str) -> float:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError):
        return 0.0


@dataclass
class Permit:
    permit_id: str
    session: str
    type: str
    scope: list[str]
    task: str
    precautions: list[str]
    issued_at: str
    heartbeat: str
    ttl_s: int
    status: str  # active | closed | expired
    closed_at: str | None = None
    close_notes: str | None = None
    eks_draft: str | None = None

    def alive(self, now: float | None = None) -> bool:
        if self.status != "active":
            return False
        now = time.time() if now is None else now
        return now - _parse_ts(self.heartbeat or self.issued_at) <= self.ttl_s

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class PermitStore:
    """File-based permit registry — one JSON per permit, TTL-expired."""

    def __init__(self, permits_dir: Path, eks_repo: EKSRepository | None = None) -> None:
        self.dir = Path(permits_dir)
        self.repo = eks_repo

    @classmethod
    def default(cls, project_root: Path) -> "PermitStore":
        root = Path(project_root)
        return cls(root / "outputs" / "devin" / "permits",
                   EKSRepository(root / "knowledge"))

    def _load(self, path: Path) -> Permit | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return Permit(**data)
        except (OSError, ValueError, TypeError):
            return None

    def all(self) -> list[Permit]:
        if not self.dir.is_dir():
            return []
        return [p for p in (self._load(f) for f in sorted(self.dir.glob("PW-*.json"))) if p]

    def active(self, now: float | None = None) -> list[Permit]:
        return [p for p in self.all() if p.alive(now)]

    def _next_id(self) -> str:
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        seq = 1
        for permit in self.all():
            match = re.match(rf"PW-{day}-(\d+)", permit.permit_id)
            if match:
                seq = max(seq, int(match.group(1)) + 1)
        return f"PW-{day}-{seq:02d}"

    def conflicts(self, scope: Iterable[str], *, exclude_session: str | None = None,
                  now: float | None = None) -> list[Permit]:
        """Active exclusive permits whose scope overlaps the requested one."""
        hits = []
        for permit in self.active(now):
            if exclude_session and permit.session == exclude_session:
                continue
            if permit.type != "exclusive":
                continue
            if any(scopes_overlap(a, b) for a in permit.scope for b in scope):
                hits.append(permit)
        return hits

    @contextlib.contextmanager
    def _acquire_guard(self, timeout_s: float = ACQUIRE_LOCK_TIMEOUT_S) -> Iterator[None]:
        """Exclusive-create lockfile around acquire (atomic across processes).

        `O_CREAT | O_EXCL` is atomic on Windows and POSIX. A lock older than
        `ACQUIRE_LOCK_STALE_S` is treated as abandoned (killed holder) and
        stolen, so a crash cannot wedge the store forever.
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        lock = self.dir / ACQUIRE_LOCK_NAME
        deadline = time.monotonic() + timeout_s
        handle: int | None = None
        while handle is None:
            try:
                handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    if time.time() - lock.stat().st_mtime > ACQUIRE_LOCK_STALE_S:
                        lock.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"permit store busy: {lock.name} held for more than {timeout_s:g}s")
                time.sleep(0.05)
        try:
            os.write(handle, f"{os.getpid()} {_utcnow()}".encode("utf-8"))
            yield
        finally:
            os.close(handle)
            try:
                lock.unlink(missing_ok=True)
            except OSError:
                pass

    def acquire(self, session: str, scope: list[str], task: str,
                *, type: str = "exclusive", ttl_s: int = DEFAULT_TTL_S,
                force: bool = False) -> tuple[Permit | None, dict[str, Any]]:
        if not session or not session.strip():
            raise ValueError("session must be non-empty")
        if type not in PERMIT_TYPES:
            raise ValueError(f"type must be one of {PERMIT_TYPES}")
        scope = [s.replace("\\", "/").lstrip("/") for s in scope
                 if isinstance(s, str) and s.strip()]
        if not scope:
            raise ValueError("scope must be a non-empty list of globs")

        # Check + id assignment + write happen under one lock, so two sessions
        # cannot both win an overlapping exclusive scope.
        with self._acquire_guard():
            conflicts = self.conflicts(scope, exclude_session=session)
            if conflicts and not force:
                return None, {
                    "issued": False,
                    "conflicts": [p.to_dict() for p in conflicts],
                    "reason": "scope overlaps an active exclusive permit",
                }

            precautions: list[str] = []
            if self.repo is not None:
                precautions = [item["id"] for item in self.repo.governing_scope(scope)]

            stamp = _utcnow()
            permit = Permit(
                permit_id=self._next_id(), session=session.strip(), type=type,
                scope=scope, task=task.strip(), precautions=precautions,
                issued_at=stamp, heartbeat=stamp, ttl_s=ttl_s, status="active",
            )
            self._save(permit)
        return permit, {
            "issued": True,
            "permit": permit.to_dict(),
            "conflicts": [p.to_dict() for p in conflicts],  # forced: reported anyway
            "precautions": precautions,
            "hot_zone": len(precautions) >= HOT_ZONE_THRESHOLD,
        }

    def get(self, permit_id: str) -> Permit | None:
        if not _ID_RE.fullmatch(permit_id or ""):
            return None
        return self._load(self.dir / f"{permit_id}.json")

    def _save(self, permit: Permit) -> None:
        (self.dir / f"{permit.permit_id}.json").write_text(
            json.dumps(permit.to_dict(), indent=2), encoding="utf-8")

    def heartbeat(self, permit_id: str) -> Permit | None:
        permit = self.get(permit_id)
        if permit is None or permit.status != "active":
            return None
        permit.heartbeat = _utcnow()
        self._save(permit)
        return permit

    def close(self, permit_id: str, *, notes: str | None = None,
              eks_draft: str | None = None) -> Permit | None:
        permit = self.get(permit_id)
        if permit is None or permit.status == "closed":
            return None
        permit.status = "closed"
        permit.closed_at = _utcnow()
        permit.close_notes = notes
        permit.eks_draft = eks_draft
        self._save(permit)
        return permit

    def close_session(self, session: str, *, reason: str | None = None) -> int:
        count = 0
        for permit in self.active():
            if permit.session == session:
                self.close(permit.permit_id, notes=reason or "session ended")
                count += 1
        return count

    def check_path(self, path: str, *, session: str | None = None) -> dict[str, Any]:
        """Is `path` under a foreign exclusive permit? Used by pre-edit hooks."""
        now = time.time()
        path = path.replace("\\", "/").lstrip("/")
        blockers = [p for p in self.active(now)
                    if p.type == "exclusive" and p.session != session
                    and any(glob_match(path, s) for s in p.scope)]
        return {
            "path": path,
            "blocked": bool(blockers),
            "permits": [p.to_dict() for p in blockers],
        }


def main(argv: list[str] | None = None) -> int:  # thin self-test
    store = PermitStore(Path(argv[0]) if argv else Path("outputs/devin/permits"))
    print(json.dumps({"active": len(store.active())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
