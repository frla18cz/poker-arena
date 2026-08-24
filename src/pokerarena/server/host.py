"""HTTP server for the table: one game, many seats, one browser tab each.

Plain ``ThreadingHTTPServer`` and polling — no websockets, no framework, no
build step. Start it, open the page, hand the other seats their links.

Three ways to play:

* **alone**, against built-in bots — nothing to configure,
* **on your Wi-Fi** (``--lan``) — friends open the seat links from their phones,
* **over the internet** (``--public-base``) — put a tunnel in front of it and
  pass the public address, so the links point somewhere reachable.

Seat links carry a secret token, and the token is what decides whose cards you
get to see. That is the whole access model, so it is worth saying plainly:
**anyone holding a seat link can play that seat.** Treat the links the way you
would treat a chair at a real table.
"""
from __future__ import annotations

import argparse
import json
import re
import secrets
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..engine.table_config import (
    AVAILABLE_MODELS, BLIND_STRUCTURES, CHIP_SCALE, PRESETS,
    SEAT_LABELS, SeatConfig, TableConfigError, known_seat_kinds,
)
from ..engine.prompt_catalog import catalog
from . import qr
from .table_host import TableHost

HERE = Path(__file__).resolve().parent
PAGE = HERE / "studio.html"
AVATARS = HERE.parent / "assets" / "avatars"
DEFAULT_TABLES = Path("games")


class ArenaHandler(BaseHTTPRequestHandler):
    """The table's endpoints. ``TableHost`` holds the state; this wraps it."""

    tables: TableHost
    owner_token: str = ""
    lock_setup: bool = False
    trust_local: bool = True

    def log_message(self, fmt: str, *args) -> None:      # noqa: A003
        """Silence. A line per polling request makes the log unreadable."""

    # -- permissions -------------------------------------------------------

    def _is_owner(self, query: dict) -> bool:
        """May this request start and stop games?

        Behind a tunnel the address **cannot** be trusted: the tunnel connects
        from ``127.0.0.1``, so every visitor from the internet would look like
        the owner. There, the owner proves it with the token from their link.
        """
        if not self.lock_setup:
            return True
        if self.owner_token and (query.get("owner") or [""])[0] == self.owner_token:
            return True
        return self.trust_local and self.client_address[0] in ("127.0.0.1", "::1")

    # -- GET ---------------------------------------------------------------

    def do_GET(self) -> None:                            # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path in ("/", "/index.html"):
                self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
                return
            if parsed.path.startswith("/avatars/"):
                self._avatar(parsed.path.removeprefix("/avatars/"))
                return
            if parsed.path == "/api/table/options":
                self._send_json(self._options())
                return
            if parsed.path == "/api/table/presets":
                self._send_json(self._presets())
                return
            if parsed.path == "/api/avatars":
                # The picker wants {id, name}: the filename to store on the
                # seat, and something readable to show in the menu.
                self._send_json({"avatars": [
                    {"id": f.name, "name": f.stem.removeprefix("bot_")}
                    for f in sorted(AVATARS.iterdir())
                    if f.suffix.lower() in (".svg", ".png")]})
                return
            if parsed.path == "/api/table/runs":
                self._send_json(self.tables.resumable())
                return
            if parsed.path == "/api/table/history":
                session = self.tables.session
                self._send_json(session.history() if session else [])
                return
            if parsed.path == "/api/table/state":
                session = self.tables.session
                if session is None:
                    self._send_json({"running": False, "seats": []})
                    return
                self._send_json(session.state((query.get("token") or [None])[0]))
                return
            if parsed.path == "/api/table/seats":
                self._send_json(self._seat_links(query))
                return
            hand = re.fullmatch(r"/api/table/hand/([^/]+)", parsed.path)
            if hand:
                session = self._session()
                self._send_json(session.finished_hand(
                    hand.group(1), (query.get("token") or [None])[0]))
                return
        except PermissionError as exc:
            self._send_text(str(exc), HTTPStatus.FORBIDDEN)
            return
        except LookupError as exc:
            self._send_text(str(exc), HTTPStatus.NOT_FOUND)
            return
        except Exception as exc:                         # noqa: BLE001
            self._send_text(f"{type(exc).__name__}: {exc}",
                            HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self._send_text("not found", HTTPStatus.NOT_FOUND)

    # -- POST --------------------------------------------------------------

    def do_POST(self) -> None:                           # noqa: N802
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_text("body is not JSON", HTTPStatus.BAD_REQUEST)
            return
        query = parse_qs(parsed.query)

        try:
            if parsed.path == "/api/table":
                if not self._is_owner(query):
                    raise PermissionError("only the owner can start a game")
                session = self.tables.create(payload)
                self._send_json(session.state(), HTTPStatus.ACCEPTED)
                return

            session = self._session()
            token = payload.get("token")
            if parsed.path == "/api/table/act":
                session.act(str(payload.get("action")),
                            float(payload.get("amount") or 0.0), token=token)
                self._send_json(session.state(token))
                return
            if parsed.path == "/api/table/stop":
                if not self._is_owner(query):
                    raise PermissionError("only the owner can stop the game")
                session.stop()
                self._send_json(session.state())
                return
            if parsed.path == "/api/table/pause":
                session.paused = bool(payload.get("paused", True))
                if not session.paused:
                    session.resume()
                self._send_json(session.state(token))
                return
            if parsed.path == "/api/table/next":
                session.resume()
                self._send_json(session.state(token))
                return
        except PermissionError as exc:
            self._send_text(str(exc), HTTPStatus.FORBIDDEN)
            return
        except LookupError as exc:
            self._send_text(str(exc), HTTPStatus.NOT_FOUND)
            return
        except (ValueError, KeyError) as exc:
            self._send_text(str(exc), HTTPStatus.BAD_REQUEST)
            return
        except Exception as exc:                         # noqa: BLE001
            self._send_text(f"{type(exc).__name__}: {exc}",
                            HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self._send_text("not found", HTTPStatus.NOT_FOUND)

    # -- pieces ------------------------------------------------------------

    def _session(self):
        session = self.tables.session
        if session is None:
            raise LookupError("no game is running")
        return session

    def _presets(self) -> dict:
        """Every preset, built. The editor needs the seats and the price, not
        just the name — it fills the menu from this and clones the seats into
        the setup panel."""
        out = {}
        for name in sorted(PRESETS):
            try:
                out[name] = PRESETS[name]().as_dict()
            except TableConfigError:
                # A preset that cannot be built must not take the page down
                # with it; leaving it out of the menu is enough.
                continue
        return out

    def _options(self) -> dict:
        """What a table can be built from; this fills the menus in the UI."""
        cat = catalog()
        defaults = SeatConfig("seat")
        return {
            "seat_kinds": sorted(known_seat_kinds()),
            "seat_labels": SEAT_LABELS,
            "models": AVAILABLE_MODELS,
            "prompt_variants": list(cat.variants),
            "recommended_prompt_variant": cat.recommended,
            "blind_structures": BLIND_STRUCTURES,
            "chip_scale": CHIP_SCALE,
            "presets": sorted(PRESETS),
            "defaults": defaults.as_dict(),
            # The UI gates setup behind the owner link when the server would
            # refuse it anyway; without this it shows the panel and only fails
            # on the click.
            "needs_owner": self.lock_setup,
        }

    def _seat_links(self, query: dict) -> list[dict]:
        """A link and a QR code for every human seat.

        Owner only: the link carries the seat's token, and whoever holds it
        plays that seat and sees its cards.
        """
        if not self._is_owner(query):
            raise PermissionError("seat links are for the owner only")
        session = self._session()
        base = self.tables.public_base or ""
        rows = []
        for seat, token in sorted(getattr(session, "tokens", {}).items()):
            url = f"{base}/?seat={token}#table"
            rows.append({"seat": seat, "url": url, "qr": qr.svg(url)})
        return rows

    def _avatar(self, name: str) -> None:
        # A bare filename only; ".." or a slash would serve anything on disk.
        if "/" in name or ".." in name:
            raise PermissionError("bad avatar name")
        path = AVATARS / name
        if not path.is_file():
            raise LookupError(name)
        kind = "image/svg+xml" if path.suffix == ".svg" else "image/png"
        self._send(path.read_bytes(), kind)

    # -- sending -----------------------------------------------------------

    def _send(self, body: bytes, content_type: str,
              status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send(json.dumps(payload, ensure_ascii=False, default=str)
                   .encode("utf-8"), "application/json; charset=utf-8", status)

    def _send_text(self, text: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send(text.encode("utf-8"), "text/plain; charset=utf-8", status)


def lan_address() -> str:
    """The IP the rest of the Wi-Fi can reach.

    Found by opening a UDP socket towards an address that goes nowhere — nothing
    is sent; it just makes the routing table reveal the right interface.
    ``gethostbyname`` tends to answer ``127.0.0.1`` on macOS.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("192.0.2.1", 9))              # TEST-NET-1, unroutable
            return probe.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def build_server(host: str, port: int, *, tables_root: Path,
                 public_base: str = "", owner_token: str = "",
                 lock_setup: bool = False,
                 trust_local: bool = True) -> ThreadingHTTPServer:
    handler = type("BoundArenaHandler", (ArenaHandler,), {
        "tables": TableHost(tables_root, public_base=public_base),
        "owner_token": owner_token,
        "lock_setup": lock_setup,
        "trust_local": trust_local,
    })
    return ThreadingHTTPServer((host, port), handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Host a poker table.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8771)
    parser.add_argument("--games", type=Path, default=DEFAULT_TABLES,
                        help="where to keep played games")
    parser.add_argument("--lan", action="store_true",
                        help="listen on the whole Wi-Fi so friends can join")
    parser.add_argument("--lock-setup", action="store_true",
                        help="only the owner may start or stop a game")
    parser.add_argument("--public-base", default="",
                        help="public address friends will reach (e.g. a tunnel "
                             "URL). Seat links and QR codes are built from it.")
    args = parser.parse_args(argv)
    if args.lan and args.host == "127.0.0.1":
        args.host = "0.0.0.0"                            # noqa: S104 — deliberate

    tunneled = bool(args.public_base)
    address = lan_address() if args.lan else "127.0.0.1"
    base = (args.public_base.rstrip("/") if tunneled
            else f"http://{address}:{args.port}")
    # Behind a tunnel the owner cannot be recognised by address, so the lock is
    # not optional there — see `_is_owner`.
    lock_setup = args.lock_setup or tunneled
    owner_token = secrets.token_urlsafe(9) if (args.lan or tunneled) else ""

    server = build_server(args.host, args.port, tables_root=args.games,
                          public_base=base, owner_token=owner_token,
                          lock_setup=lock_setup, trust_local=not tunneled)
    print(f"arena: {base}/")
    if owner_token:
        print(f"  owner link: {base}/?owner={owner_token}#table")
        print("  seat links for the others are in the table panel")
    if args.lan:
        print(f"  ⚠️  listening on the whole network ({args.host})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":                               # pragma: no cover
    raise SystemExit(main())
