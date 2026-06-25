from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto


class _Action(Enum):
    ACCEPT = auto()
    REPLACE = auto()
    REJECT = auto()


@dataclass
class GuardResult:
    _action: _Action
    _old_archive_name: str | None = None

    @property
    def accepted(self) -> bool:
        return self._action == _Action.ACCEPT

    @property
    def replace(self) -> bool:
        return self._action == _Action.REPLACE

    @property
    def rejected(self) -> bool:
        return self._action == _Action.REJECT

    @property
    def old_archive_name(self) -> str | None:
        return self._old_archive_name


class UploadGuard:
    """Tracks recent uploads to detect harmless duplicates and burst abuse.

    Scenario 1 — same IP + same slug within ``dedup_window_seconds``:
        Returns ``REPLACE`` so the caller can delete the old archive and store the
        latest upload.

    Scenario 2 — same IP sends ``rate_limit_max_slugs`` or more *distinct* slugs
    within ``rate_limit_window_seconds``:
        Returns ``REJECT`` and bans the IP for ``cooldown_seconds``.  This stops
        an IP that is flooding random workspace names (e.g. ``tmp*``).
    """

    def __init__(
        self,
        dedup_window_seconds: int = 900,
        rate_limit_max_slugs: int = 3,
        rate_limit_window_seconds: int = 30,
        cooldown_seconds: int = 600,
    ) -> None:
        self._dedup_window = dedup_window_seconds
        self._rate_max = rate_limit_max_slugs
        self._rate_window = rate_limit_window_seconds
        self._cooldown = cooldown_seconds

        # (ip, slug) → (archive_name, monotonic_ts)
        self._dedup: dict[tuple[str, str], tuple[str, float]] = {}
        # ip → list[(slug, monotonic_ts)]
        self._rate_log: dict[str, list[tuple[str, float]]] = {}
        # ip → banned_until_monotonic_ts
        self._banned: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, ip: str, slug: str) -> GuardResult:
        now = time.monotonic()

        # 1. Cooldown — IP is banned from a previous burst
        if ip in self._banned:
            if now < self._banned[ip]:
                return GuardResult(_Action.REJECT)
            del self._banned[ip]

        # 2. Burst rate limit — same IP, many different slugs in a short window
        result = self._check_rate(ip, slug, now)
        if result is not None:
            return result

        # 3. Harmless dedup — same IP + same slug, keep latest
        key = (ip, slug)
        if key in self._dedup:
            old_name, old_ts = self._dedup[key]
            if now - old_ts < self._dedup_window:
                return GuardResult(_Action.REPLACE, old_name)

        return GuardResult(_Action.ACCEPT)

    def record(self, ip: str, slug: str, archive_name: str) -> None:
        now = time.monotonic()
        self._dedup[(ip, slug)] = (archive_name, now)
        self._rate_log.setdefault(ip, []).append((slug, now))
        self._evict_expired(now)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_rate(self, ip: str, slug: str, now: float) -> GuardResult | None:
        entries = self._rate_log.get(ip)
        if not entries:
            return None

        # Evict expired entries for this IP
        cutoff = now - self._rate_window
        entries[:] = [(s, ts) for s, ts in entries if ts > cutoff]
        if not entries:
            return None

        # Count distinct slugs in the window (include the current slug)
        distinct = {s for s, _ in entries}
        distinct.add(slug)

        if len(distinct) >= self._rate_max:
            self._banned[ip] = now + self._cooldown
            del self._rate_log[ip]
            return GuardResult(_Action.REJECT)

        return None

    def _evict_expired(self, now: float) -> None:
        cutoff = now - self._dedup_window
        expired = [k for k, (_, ts) in self._dedup.items() if ts <= cutoff]
        for k in expired:
            del self._dedup[k]
