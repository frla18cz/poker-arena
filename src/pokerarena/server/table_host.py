"""Hosting a running table: starting a game, resuming one, listing past games.

Only the live table lives here. Reading an archive of finished hands is a
different application with different needs, and mixing the two produced one
file that was hard to reason about.
"""
from __future__ import annotations

import json
import secrets
import time
from pathlib import Path


class TableHost:
    """Holds at most one running table; more than one makes no sense locally."""

    def __init__(self, out_root: Path, public_base: str = "") -> None:
        self.out_root = out_root
        self.public_base = public_base
        self.session = None

    def create(self, payload: dict):
        from dataclasses import replace

        from ..engine.table_config import PRESETS, TableConfig
        from ..engine.table_session import TableSession

        if self.session is not None:
            # "New game" is a deliberate instruction, so the old table ends
            # rather than refusing — and we wait for its thread to finish.
            self.session.stop()
        if payload.get("resume"):
            # Resuming: the directory stays, and stacks, button and seat tokens
            # are restored, so everyone's existing link still works.
            out_dir = self.out_root / str(payload["resume"])
            if not (out_dir / TableSession.STATE_FILE).exists():
                raise LookupError(f"cannot resume {payload['resume']}")
            self.session = TableSession.load(out_dir)
            self.session.start(int(payload.get("hands", 20)))
            return self.session
        preset = payload.get("preset")
        config = (PRESETS[preset]() if preset in PRESETS
                  else TableConfig.from_dict(payload["config"]))
        # A new game gets a new deck. The seed otherwise defaults to 1 and the
        # UI does not send one, so every new game dealt the same cards in the
        # same order — the same hand again on the fourth deal, forever.
        #
        # Reproducibility is still available: send a seed and you get exactly
        # that game back, and the value is stored in `state.json`.
        if not (payload.get("config") or {}).get("seed"):
            config = replace(config, seed=secrets.randbelow(2 ** 31 - 1) + 1)
        label = str(payload.get("label") or preset or "table")
        out_dir = self._fresh_dir(label)
        self.session = TableSession(config, out_dir=out_dir)
        self.session.label = out_dir.name
        self.session.start(int(payload.get("hands", 20)))
        return self.session

    def resumable(self) -> list[dict]:
        """Games that can be resumed — the ones with saved state."""
        from ..engine.table_session import TableSession

        rows = []
        if not self.out_root.exists():
            return rows
        for path in sorted(self.out_root.iterdir()):
            state = path / TableSession.STATE_FILE
            if not state.is_file():
                continue
            try:
                data = json.loads(state.read_text("utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            # When the last hand was played. Taken from the hand log rather than
            # `state.json`, which is rewritten when a table stops too and would
            # therefore show when the game was last touched, not last played.
            hands_file = path / TableSession.SNAPSHOTS_FILE
            source = hands_file if hands_file.exists() else state
            try:
                last_played = source.stat().st_mtime
            except OSError:
                last_played = 0.0
            rows.append({
                "label": path.name,
                "hands_played": data.get("hands_played", 0),
                "last_hand_at": last_played,
                "seats": [s["name"] for s in data.get("config", {}).get("seats", [])],
                "humans": [s["name"] for s in data.get("config", {}).get("seats", [])
                           if s.get("kind") == "human"],
                "stacks": data.get("stacks", {}),
            })
        # Most recent first — that is what people reach for.
        return sorted(rows, key=lambda r: -r["last_hand_at"])

    def _fresh_dir(self, label: str) -> Path:
        """Each game gets its own directory, like ``game-0810-2135``.

        Reusing one directory meant a new game erased the previous one, and two
        different line-ups occasionally ended up in the same log.

        The name carries **the date and time it started**, not a counter:
        ``game-2`` and ``game-3`` are indistinguishable after a few days, and
        there was no telling which group played which. Seconds are added only on
        a collision, when two games start in the same minute.
        """
        slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in label) or "table"
        stamp = time.strftime("%m%d-%H%M")
        candidate = self.out_root / f"{slug}-{stamp}"
        if candidate.exists() and any(candidate.iterdir()):
            candidate = self.out_root / f"{slug}-{stamp}{time.strftime('%S')}"
        index = 2
        while candidate.exists() and any(candidate.iterdir()):
            candidate = self.out_root / f"{slug}-{stamp}-{index}"
            index += 1
        return candidate
