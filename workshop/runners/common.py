"""Shared runner for the workshop labs.

Every lab has the same interface:

    python -m workshop.labs.lab01 [--session NAME] [--check] [--reset]
                                   [--verbose] [--show-solution]

Guarantees, enforced while the lab computes and writes its outputs:

* **Output isolation.** A lab writes only to
  `workshop_runs/<session>/<lab_id>/` (or under the root named by the
  `SPY_WORKSHOP_RUNS_DIR` environment variable). A process audit hook
  records any file write outside that directory, and the run then fails.
* **Zero model calls.** The same hook refuses every network connection,
  so a lab cannot reach a model provider even by accident.
* **Read-only evidence.** Labs read small fixtures from
  `workshop/fixtures/` and never open `evaluation_runs/` or `docs/`.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import platform
import re
import shutil
import socket
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSHOP_DIR = PROJECT_ROOT / "workshop"
FIXTURES_DIR = WORKSHOP_DIR / "fixtures"
EXPECTED_DIR = WORKSHOP_DIR / "expected"
MANIFEST_PATH = WORKSHOP_DIR / "lab_manifest.json"
PROVENANCE_PATH = WORKSHOP_DIR / "fixture_provenance.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "workshop_runs"
OUTPUT_ROOT_ENV = "SPY_WORKSHOP_RUNS_DIR"
DEFAULT_SESSION = "default"

# Directories no workshop run may write into, or be rooted in.
PROTECTED_DIRS = tuple(PROJECT_ROOT / name for name in ("evaluation_runs", "docs", "src", "tests", "experiments", "data", "assets")) + (
    FIXTURES_DIR,
    EXPECTED_DIR,
)

_SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

# The first-difference ladder, reused by every lab (docs/agent_engineering_playbook.md section 3).
LADDER = (
    "world snapshot",
    "Observation",
    "AgentRequest",
    "raw model output",
    "normalized action",
    "validation",
    "repair",
    "fallback",
    "commit order",
    "apply-time legality",
    "world outcome",
)


class WorkshopError(Exception):
    """A setup or safety problem a participant can fix (bad session name, bad output root...)."""


# --------------------------------------------------------------------------- paths


def _real(path: Path | str) -> Path:
    return Path(os.path.realpath(os.fspath(path)))


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def output_root() -> Path:
    """The workshop output root, validated so that it can never overlap the project's evidence or source."""
    raw = os.environ.get(OUTPUT_ROOT_ENV)
    root = _real(Path(raw).expanduser()) if raw else _real(DEFAULT_OUTPUT_ROOT)
    project = _real(PROJECT_ROOT)
    for protected in PROTECTED_DIRS:
        p = _real(protected)
        if _is_within(root, p):
            raise WorkshopError(f"output root {root} is inside the protected directory {p}")
        if _is_within(p, root):
            raise WorkshopError(f"output root {root} would contain the protected directory {p}")
    if _is_within(root, project) and not _is_within(root, _real(DEFAULT_OUTPUT_ROOT)):
        raise WorkshopError(f"inside the project, workshop outputs must live under {DEFAULT_OUTPUT_ROOT}, not {root}")
    return root


def validate_session(name: str) -> str:
    if not isinstance(name, str) or not _SESSION_RE.match(name):
        raise WorkshopError(
            f"invalid session name {name!r}: use 1-64 letters, digits, '_' or '-', starting with a letter or digit (e.g. --session alice)"
        )
    return name


def session_dir(session: str) -> Path:
    return output_root() / validate_session(session)


def lab_run_dir(session: str, lab_id: str) -> Path:
    return session_dir(session) / lab_id


def ensure_within_output_root(path: Path) -> Path:
    """Resolve `path` (following symlinks) and refuse it unless it lies strictly inside the workshop output root."""
    root = output_root()
    target = _real(path)
    if target == root or not _is_within(target, root):
        raise WorkshopError(f"refusing to use {target}: it is not inside the workshop output root {root}")
    return target


def remove_within_output_root(path: Path) -> bool:
    """Delete a directory tree, but only if it lies strictly inside the workshop output root."""
    target = ensure_within_output_root(path)
    if not target.exists():
        return False
    shutil.rmtree(target)
    return True


def display_path(path: Path) -> str:
    p = _real(path)
    project = _real(PROJECT_ROOT)
    return str(p.relative_to(project)) if _is_within(p, project) else str(p)


# --------------------------------------------------------------------------- manifest and fixtures


def load_manifest() -> dict[str, Any]:
    with open(MANIFEST_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def lab_spec(lab_id: str) -> dict[str, Any]:
    for entry in load_manifest()["labs"]:
        if entry["lab_id"] == lab_id:
            return entry
    raise WorkshopError(f"{lab_id} is not listed in {display_path(MANIFEST_PATH)}")


def load_provenance() -> dict[str, Any]:
    with open(PROVENANCE_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fixture_path(name: str) -> Path:
    return FIXTURES_DIR / f"{name}.json"


def load_fixture(name: str) -> dict[str, Any]:
    path = fixture_path(name)
    if not path.exists():
        raise WorkshopError(f"missing fixture {display_path(path)} - run `python -m workshop.scripts.preflight`")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- small helpers used by labs


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short(digest: str | None, n: int = 12) -> str:
    return digest[:n] if digest else "-"


def decision_label(action: Any, target: Any = None) -> str:
    """`move(desk_01)` with the action name lower-cased (raw model output mixes `MOVE` and `move`)."""
    name = str(action).strip().lower() if action is not None else "none"
    return f"{name}({target})" if target else name


def tally(labels: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def modal(counts: dict[str, int]) -> str:
    """Most frequent label; ties are broken alphabetically and flagged with '(tie)'."""
    if not counts:
        return "-"
    top = max(counts.values())
    winners = sorted(label for label, n in counts.items() if n == top)
    return winners[0] if len(winners) == 1 else f"{winners[0]} (tie)"


def format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{label} ×{n}" for label, n in counts.items()) or "-"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(v: Any) -> str:
        return str(v).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def code_block(text: str, lang: str = "text") -> str:
    return f"```{lang}\n{text.rstrip()}\n```"


# --------------------------------------------------------------------------- guard (network + writes)


_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
_LOCAL_HOSTS = {None, "", "localhost", "127.0.0.1", "::1"}
_PATH_EVENTS = {"os.remove", "os.rmdir", "os.mkdir", "shutil.rmtree", "os.truncate", "os.chmod", "os.utime"}


class _Guard:
    def __init__(self) -> None:
        self.active = False
        self.allowed: tuple[str, ...] = ()
        self.allowed_hosts: tuple[str, ...] = ()
        self.network_attempts: list[str] = []
        self.unexpected_writes: list[str] = []
        self.allowed_lookups: list[str] = []
        self.allowed_connections: list[str] = []

    def activate(self, allowed_dirs: list[Path], allowed_hosts: tuple[str, ...] = ()) -> None:
        """allowed_hosts (Stage 3 milestone 4B, provider-backed matches only): name lookups of exactly these
        hosts are allowed and recorded, and so are HTTPS (port 443) connections; every other lookup and
        connection is still refused and recorded. The default (no host) refuses all network, as before."""
        self.allowed = tuple(os.fspath(_real(p)) for p in allowed_dirs)
        self.allowed_hosts = tuple(allowed_hosts)
        self.network_attempts = []
        self.unexpected_writes = []
        self.allowed_lookups = []
        self.allowed_connections = []
        self.active = True

    def deactivate(self) -> None:
        self.active = False

    def allowed_path(self, path: str) -> bool:
        return any(path == a or path.startswith(a + os.sep) for a in self.allowed)


GUARD = _Guard()
_HOOK_INSTALLED = False


def _path_of(value: Any) -> str | None:
    if isinstance(value, int) or value is None:
        return None
    try:
        return os.path.realpath(os.fsdecode(value))
    except (TypeError, ValueError):
        return None


def _audit_hook(event: str, args: tuple) -> None:
    g = GUARD
    if not g.active:
        return
    if event == "socket.connect":
        sock = args[0] if args else None
        if getattr(sock, "family", None) == getattr(socket, "AF_UNIX", None):
            return
        address = args[1] if len(args) > 1 else None
        if g.allowed_hosts and isinstance(address, tuple) and len(address) >= 2 and address[1] == 443:
            # a host name gets here only through an allowed lookup; a literal IP on 443 would too (recorded)
            g.allowed_connections.append(f"connect {address[0]}:443")
            return
        g.network_attempts.append(f"connect {address if address is not None else '?'!r}")
        raise ConnectionRefusedError("workshop labs make zero network calls: this connection was blocked by the workshop guard")
    if event in ("socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyname_ex"):
        host = args[0] if args else None
        if isinstance(host, (bytes, bytearray)):
            host = host.decode("utf-8", "replace")
        if host in _LOCAL_HOSTS:
            return
        if host in g.allowed_hosts:
            g.allowed_lookups.append(str(host))
            return
        g.network_attempts.append(f"{event} {host!r}")
        raise OSError("workshop labs make zero network calls: this lookup was blocked by the workshop guard")
    if event == "open":
        path = _path_of(args[0] if args else None)
        if path is None:
            return
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and bool(flags & _WRITE_FLAGS))
        if writing and not g.allowed_path(path):
            g.unexpected_writes.append(f"open {path}")
        return
    if event in _PATH_EVENTS or event in ("os.rename", "shutil.copyfile", "shutil.move"):
        targets = args[1:2] if event in ("shutil.copyfile",) else args[:2] if event in ("os.rename", "shutil.move") else args[:1]
        for value in targets:
            path = _path_of(value)
            if path is not None and not g.allowed_path(path):
                g.unexpected_writes.append(f"{event} {path}")


def install_guard() -> None:
    global _HOOK_INSTALLED
    if not _HOOK_INSTALLED:
        sys.addaudithook(_audit_hook)
        _HOOK_INSTALLED = True


def prepare_environment() -> None:
    """Headless pygame, no telemetry file, no bytecode caches in the source tree."""
    sys.dont_write_bytecode = True
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.pop("SPY_VS_AI_TELEMETRY_PATH", None)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------------- lab contract


@dataclass
class LabContext:
    lab_id: str
    session: str
    run_dir: Path
    verbose: bool
    notes: list[str] = field(default_factory=list)
    fixtures_used: dict[str, str] = field(default_factory=dict)

    def fixture(self, name: str) -> dict[str, Any]:
        self.fixtures_used[name] = file_sha256(fixture_path(name)) if fixture_path(name).exists() else "missing"
        return load_fixture(name)

    def note(self, message: str) -> None:
        self.notes.append(message)


@dataclass
class LabOutput:
    """What a lab computes. Everything in `results` must be deterministic (no clocks, paths or thread timings)."""

    results: dict[str, Any]
    changed: str
    fixed: str
    inspect: str
    ladder: dict[str, str]
    first_difference: str
    lesson: str
    sections: list[tuple[str, str]] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    questions: list[str] = field(default_factory=list)
    solution: str = ""


def render_ladder(statuses: dict[str, str], first_difference: str, *, markdown: bool = False) -> str:
    arrow = "→" if markdown else "->"
    lines = []
    for i, layer in enumerate(LADDER):
        status = statuses.get(layer, "")
        marker = ("  ◀ FIRST DIFFERENCE" if markdown else "  <== FIRST DIFFERENCE") if layer == first_difference else ""
        prefix = f" {arrow} " if i else "   "
        lines.append(f"{prefix}{layer:<20} {status}{marker}".rstrip())
    return "\n".join(lines)


def check_against_expected(lab_id: str, results: dict[str, Any]) -> tuple[bool, list[str]]:
    path = EXPECTED_DIR / f"{lab_id}.json"
    if not path.exists():
        return False, [f"no reference file {display_path(path)}"]
    with open(path, encoding="utf-8") as fh:
        expected = json.load(fh).get("check", {})
    got = results.get("check", {})
    problems = []
    for key in sorted(set(expected) | set(got)):
        if key not in got:
            problems.append(f"{key}: missing from this run")
        elif key not in expected:
            problems.append(f"{key}: not in the reference file")
        elif got[key] != expected[key]:
            problems.append(f"{key}: expected {json.dumps(expected[key], sort_keys=True)}, got {json.dumps(got[key], sort_keys=True)}")
    return not problems, problems


def _dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def _observations_md(spec: dict[str, Any], session: str, out: LabOutput) -> str:
    parts = [
        f"# {spec['title']}",
        "",
        f"Lab `{spec['lab_id']}` · session `{session}` · zero model calls · experiment {', '.join(spec['source_w'])}",
        "",
        "## What changed",
        out.changed,
        "",
        "## What stayed fixed",
        out.fixed,
        "",
        "## What to inspect",
        out.inspect,
        "",
        "## First differing layer",
        f"**{out.first_difference}**",
        "",
        code_block(render_ladder(out.ladder, out.first_difference, markdown=True)),
        "",
        "## Engineering lesson",
        out.lesson,
        "",
    ]
    for heading, body in out.sections:
        parts += [f"## {heading}", body.rstrip(), ""]
    if out.questions:
        parts += ["## Questions", ""]
        parts += [f"{i}. {q}" for i, q in enumerate(out.questions, start=1)]
        parts += ["", f"Answers: `python -m workshop.labs.{spec['lab_id']} --show-solution`", ""]
    parts += ["## Evidence", ""]
    parts += [f"- {line}" for line in spec.get("evidence", [])]
    return "\n".join(parts).rstrip() + "\n"


def _print_view(spec: dict[str, Any], session: str, run_dir: Path, out: LabOutput, duration: float) -> None:
    bar = "=" * 72
    print(bar)
    print(f"{spec['title']}  (zero model calls, {duration:.1f} s)")
    print(f"Session: {session}    Output: {display_path(run_dir)}/")
    print(bar)
    print(f"WHAT CHANGED       {out.changed}")
    print(f"WHAT STAYED FIXED  {out.fixed}")
    print(f"INSPECT            {out.inspect}")
    print()
    print("FIRST-DIFFERENCE LADDER (find the first layer that differs)")
    print(render_ladder(out.ladder, out.first_difference))
    print()
    print(f"FIRST DIFFERING LAYER  {out.first_difference}")
    print(f"ENGINEERING LESSON     {out.lesson}")
    if out.questions:
        print()
        print("QUESTIONS (answers: --show-solution)")
        for i, q in enumerate(out.questions, start=1):
            print(f"  {i}. {q}")


def run_lab(lab_id: str, compute: Callable[[LabContext], LabOutput], argv: list[str] | None = None) -> int:
    """Run one lab end to end. Exit codes: 0 ok, 1 lab error or failed check, 2 usage/safety error, 3 guard violation."""
    try:
        spec = lab_spec(lab_id)
    except (OSError, WorkshopError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser(
        prog=f"python -m workshop.labs.{lab_id}",
        description=f"{spec['title']} - zero model calls; writes only to workshop_runs/<session>/{lab_id}/.",
    )
    parser.add_argument("--session", default=DEFAULT_SESSION, help="your session name (letters, digits, '_' or '-'); default: %(default)s")
    parser.add_argument("--check", action="store_true", help="compare the results with workshop/expected/%s.json and print PASS or the mismatch" % lab_id)
    parser.add_argument("--reset", action="store_true", help="delete this lab's output for the session, then exit")
    parser.add_argument("--verbose", action="store_true", help="also write verbose.log with the captured library output")
    parser.add_argument("--show-solution", action="store_true", help="print the answers to the lab questions")
    args = parser.parse_args(argv)

    try:
        run_dir = lab_run_dir(args.session, lab_id)
        if args.reset:
            removed = remove_within_output_root(run_dir)
            print(f"{'Removed' if removed else 'Nothing to remove at'} {display_path(run_dir)}/ "
                  "(evidence, docs, fixtures and expected outputs are never touched)")
            return 0
        prepare_environment()
        run_dir = ensure_within_output_root(run_dir)  # a symlinked session directory cannot redirect the outputs
        if run_dir.exists():
            remove_within_output_root(run_dir)
        run_dir.mkdir(parents=True)
    except WorkshopError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    ctx = LabContext(lab_id=lab_id, session=args.session, run_dir=run_dir, verbose=args.verbose)
    install_guard()
    GUARD.activate([run_dir, Path(os.devnull)])
    captured = io.StringIO()
    started_utc = datetime.now(timezone.utc)
    start = time.perf_counter()
    try:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            out = compute(ctx)
    except Exception:  # noqa: BLE001 - report any lab failure clearly, with the guard's findings
        GUARD.deactivate()
        sys.stderr.write(captured.getvalue()[-4000:] if args.verbose else "")
        traceback.print_exc()
        if GUARD.network_attempts:
            print(f"GUARD: blocked network access: {GUARD.network_attempts}", file=sys.stderr)
        print(f"ERROR: {lab_id} failed; see the traceback above and instructor_guide.md 'Recovery'.", file=sys.stderr)
        return 1
    duration = time.perf_counter() - start

    first_difference = out.first_difference
    if first_difference not in LADDER:
        GUARD.deactivate()
        print(f"ERROR: {lab_id} reported an unknown ladder layer {first_difference!r}", file=sys.stderr)
        return 1

    results = {
        "lab_id": lab_id,
        "title": spec["title"],
        "source_w": spec["source_w"],
        "first_difference": first_difference,
        "ladder": {layer: out.ladder.get(layer, "") for layer in LADDER},
        **out.results,
    }
    results_text = _dump(results)
    files = {
        "results.json": results_text,
        "observations.md": _observations_md(spec, args.session, out),
    }
    files.update(out.artifacts)
    if args.verbose:
        files["verbose.log"] = captured.getvalue()
    for name, text in files.items():
        (run_dir / name).write_text(text, encoding="utf-8")

    check_ok, problems = (True, [])
    if args.check:
        check_ok, problems = check_against_expected(lab_id, results)

    run_meta = {
        "session_id": args.session,
        "lab_id": lab_id,
        "title": spec["title"],
        "timestamp_utc": started_utc.isoformat(timespec="seconds"),
        "duration_s": round(duration, 3),
        "source_experiment": spec["source_w"],
        "evidence": spec.get("evidence", []),
        "runtime_mode": spec.get("runtime_mode", "zero-call"),
        "requires_provider": False,
        "model_calls": 0,
        "participant_options": {"check": args.check, "verbose": args.verbose, "show_solution": args.show_solution},
        "fixtures": dict(sorted(ctx.fixtures_used.items())),
        "results_sha256": sha256_text(results_text),
        "network_attempts": list(GUARD.network_attempts),
        "unexpected_writes": list(GUARD.unexpected_writes),
        "check": ({"passed": check_ok, "problems": problems} if args.check else None),
        "python": platform.python_version(),
        "platform": platform.platform(terse=True),
        "notes": ctx.notes,
        "files": sorted(list(files) + ["run.json"]),
    }
    (run_dir / "run.json").write_text(_dump(run_meta), encoding="utf-8")
    GUARD.deactivate()

    _print_view(spec, args.session, run_dir, out, duration)
    if args.show_solution and out.solution:
        print()
        print("SOLUTION")
        print(out.solution.rstrip())

    violations = GUARD.network_attempts or GUARD.unexpected_writes
    if violations:
        print()
        print("GUARD VIOLATION: this run touched the network or wrote outside its run directory:", file=sys.stderr)
        for item in GUARD.network_attempts + GUARD.unexpected_writes:
            print(f"  {item}", file=sys.stderr)
        return 3
    if args.check:
        print()
        if check_ok:
            n = len(results.get("check", {}))
            print(f"CHECK PASS - {n} values match workshop/expected/{lab_id}.json")
        else:
            print(f"CHECK FAIL - {len(problems)} value(s) differ from workshop/expected/{lab_id}.json:")
            for p in problems:
                print(f"  - {p}")
            print("This lab is deterministic, so a mismatch means the code, the fixtures or the environment changed.")
            print("See workshop/instructor_guide.md, 'Recovery if output differs'.")
            return 1
    return 0
