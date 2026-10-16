"""Validate a public Spy vs AI repository tree (spec KIRO_PUBLIC_PRIVATE_REPOSITORY_SPLIT.md, sections 18-19, 33).

    python -m tools.validate_public_repo                  # validate this tree (run from its root)
    python -m tools.validate_public_repo --root DIR       # validate another tree (a fresh export)
    python -m tools.validate_public_repo --skip-tests     # everything but the test suite

Offline and call-free: 0 provider calls. Every subprocess runs inside the tree being validated, with provider
credentials, AWS configuration files and profiles removed from its environment, so nothing can reach a provider.

Checks: required files; forbidden private paths absent; no secret and no credential file (code, docs, example
replays and the outputs of the matches run here); no absolute local path; no dead markdown link or image; policy
artifacts byte-identical to the frozen hashes, and every policy loads for every identity and personality; example
configs parse; CLI; the no-option launch opens the visual setup; a call-free B2-v1 vs B2-v2 match and a
fake-provider Hybrid match (with the Strands SDK blocked) run, save a replay and replay offline; every example replay
reproduces on identical game code; every project import resolves; the test suite passes.
Exit code 1 if any check fails. A missing LICENSE is reported as a publication blocker, not a failure.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROVENANCE = "PUBLIC_EXPORT_PROVENANCE.json"
REQUIRED = (
    "README.md", "CHANGELOG.md", "requirements.txt", "requirements-dev.txt", ".gitignore", PROVENANCE,
    "spy_vs_ai/play.py", "spy_vs_ai/replay.py", "docs/README.md",
    "docs/play/PLAYING_THE_GAME.md", "docs/play/PROVIDERS_AND_MODELS.md", "docs/research/STORY.md",
    "docs/research/RESULTS.md", "docs/project/ARCHITECTURE.md", "docs/project/GLOSSARY.md",
    "paper/SPY_VS_AI_PAPER.md", "examples/replays/README.md", "assets/README.md", "configs/demo_offline.yaml",
    "tests/conftest.py",
)
FORBIDDEN = ("workshop_runs", "evaluation_runs", "experiments", "runs", "runtime_logs", "replays", "docs/history",
             "workshop/policy_reports", ".venv", ".env", "PUBLIC_EXPORT_MANIFEST.yaml", "PRIVATE_RESEARCH_ARCHIVE.md",
             "public_overlay")
LOCAL_OUTPUTS = ("runs", "replays", "runtime_logs")  # a user's own traces: allowed locally, ignored by Git
BINARY = {".png", ".jpg", ".jpeg", ".gif", ".mp4", ".webp", ".ico", ".pdf", ".zip"}
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv"}
# credential shapes (spec 19). Test fixtures use obviously fake values that contain FAKE.
SECRET_PATTERNS = {
    "AWS access key id": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "AWS secret access key": re.compile(r"(?i)aws_secret_access_key\s*[=:]\s*['\"]?[A-Za-z0-9/+=]{40}"),
    "Bedrock API key": re.compile(r"\bABSK[A-Za-z0-9+/=]{20,}"),
    "OpenAI key": re.compile(r"\bsk-(?!ant-)(?:proj-)?[A-Za-z0-9_-]{20,}"),
    "Anthropic key": re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}"),
    "authorization header": re.compile(r"(?i)\bauthorization['\"]?\s*[:=]\s*['\"]?(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{12,}"),
    "bearer token": re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}
SECRET_FILES = re.compile(r"(^|/)(\.env(\..*)?|\.netrc|\.pgpass|id_rsa[^/]*|id_ed25519[^/]*|[^/]*\.pem|[^/]*\.p12|"
                          r"credentials(\.json)?|\.aws|\.[a-z]*_history|[^/]*_history)$")
LOCAL_PATH = re.compile(r"(?:/Users/|/home/)[A-Za-z0-9._-]+/|/private/var/[a-z]+/|/var/folders/[A-Za-z0-9_]{2,}/|"
                        r"\b[A-Za-z]:\\\\?Users\\\\?[A-Za-z]")
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
SRC_RE = re.compile(r"<(?:img|video|source)\b[^>]*\bsrc=\"([^\"]+)\"", re.I)
PROJECT_PACKAGES = ("src", "workshop", "spy_vs_ai", "tools", "tests")
CREDENTIAL_ENV = re.compile(r"^(AWS_|OPENAI_|ANTHROPIC_|AZURE_|GOOGLE_APPLICATION|BEDROCK_|OLLAMA_)")


class Validator:
    def __init__(self, root: Path, *, skip_tests: bool, python: str) -> None:
        self.root = root
        self.skip_tests = skip_tests
        self.python = python
        self.results: list[dict] = []
        self.provenance = json.loads((root / PROVENANCE).read_text(encoding="utf-8")) if (root / PROVENANCE).is_file() else {}
        self.work = Path(tempfile.mkdtemp(prefix="spy-vs-ai-validate-"))

    # -- plumbing -------------------------------------------------------------------------------------------------
    def record(self, name: str, ok: bool, detail: str = "", *, level: str | None = None) -> bool:
        status = level or ("PASS" if ok else "FAIL")
        self.results.append({"check": name, "status": status, "detail": detail})
        print(f"[{status}] {name}: {detail}", flush=True)
        return ok

    def env(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if not CREDENTIAL_ENV.match(k) and k not in ("PYTHONPATH", "VIRTUAL_ENV_PROMPT")}
        missing = str(self.work / "no-such-aws-file")
        env.update(AWS_CONFIG_FILE=missing, AWS_SHARED_CREDENTIALS_FILE=missing, AWS_EC2_METADATA_DISABLED="true",
                   PYTHONDONTWRITEBYTECODE="1", SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy",
                   PYGAME_HIDE_SUPPORT_PROMPT="1")
        return env

    def run(self, args: list[str], *, timeout: float = 900) -> subprocess.CompletedProcess:
        return subprocess.run([self.python, *args], cwd=self.root, env=self.env(), capture_output=True, text=True,
                              timeout=timeout)

    def files(self):
        for path in sorted(self.root.rglob("*")):
            rel = path.relative_to(self.root)
            if path.is_file() and not (set(rel.parts) & SKIP_DIRS):
                yield path, rel.as_posix()

    # -- structure ------------------------------------------------------------------------------------------------
    def check_required(self) -> None:
        missing = [p for p in REQUIRED if not (self.root / p).is_file()]
        self.record("required files", not missing, f"{len(REQUIRED) - len(missing)}/{len(REQUIRED)} present"
                    + (f"; missing {missing}" if missing else ""))
        has_license = any((self.root / n).is_file() for n in ("LICENSE", "LICENSE.md", "LICENSE.txt"))
        self.record("license", True, "PRESENT" if has_license else "MISSING - publication blocker (no license chosen yet)",
                    level="PASS" if has_license else "WARN")

    def check_forbidden(self) -> None:
        names = sorted(set(FORBIDDEN) | set(self.provenance.get("forbidden") or []))
        ignore = [ln.strip() for ln in (self.root / ".gitignore").read_text(encoding="utf-8").splitlines()] \
            if (self.root / ".gitignore").is_file() else []
        unignored = [n for n in LOCAL_OUTPUTS if f"/{n}/" not in ignore]
        local = [n for n in LOCAL_OUTPUTS if (self.root / n).exists()]
        present = [n for n in names if (self.root / n).exists() and n not in LOCAL_OUTPUTS]
        self.record("forbidden private paths absent", not present and not unignored, f"{len(names)} checked"
                    + (f"; present: {present}" if present else "")
                    + (f"; your local outputs {local} exist and are ignored by Git" if local else "")
                    + (f"; NOT ignored by .gitignore: {unignored}" if unignored else ""))

    # -- content scans --------------------------------------------------------------------------------------------
    @staticmethod
    def secrets_in(text: str) -> list[tuple[str, str]]:
        hits = []
        for label, pattern in SECRET_PATTERNS.items():
            for m in pattern.finditer(text):
                if "FAKE" not in m.group(0) and "fake" not in m.group(0):
                    hits.append((label, m.group(0)[:12] + "..."))
        return hits

    def scan_tree(self, root: Path | None = None, *, local_paths: bool = True) -> tuple[list[str], list[str]]:
        root = root or self.root
        secrets, local = [], []
        for path in sorted(root.rglob("*")):
            rel = path.relative_to(root).as_posix()
            if not path.is_file() or set(Path(rel).parts) & SKIP_DIRS:
                continue
            if SECRET_FILES.search(rel):
                secrets.append(f"{rel}: credential-like file name")
            if path.suffix.lower() in BINARY:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            secrets += [f"{rel}: {label} ({snippet})" for label, snippet in self.secrets_in(text)]
            if local_paths:
                local += [f"{rel}: {m.group(0)[:40]}" for m in LOCAL_PATH.finditer(text)]
        return secrets, local

    def check_scans(self) -> None:
        secrets, local = self.scan_tree()
        self.record("secret scan", not secrets, "no credential, key, token or credential file found" if not secrets
                    else f"{len(secrets)} finding(s): {secrets[:5]}")
        self.record("absolute local path scan", not local, "no machine-specific path" if not local
                    else f"{len(local)} finding(s): {local[:5]}")
        leftovers = [rel for path, rel in self.files() if path.suffix == ".md"
                     and "<!-- private:" in path.read_text(encoding="utf-8")]
        self.record("no archive-only block left", not leftovers, "every <!-- private --> block was removed"
                    if not leftovers else f"still in {leftovers}")

    # -- markdown -------------------------------------------------------------------------------------------------
    @staticmethod
    def _slug(heading: str) -> str:
        text = re.sub(r"[`*_]", "", heading.strip().lower())
        text = re.sub(r"[^\w\- ]", "", text)
        return text.replace(" ", "-")

    def _anchors(self, path: Path) -> set[str]:
        out, seen, in_code = set(), {}, False
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("```"):
                in_code = not in_code
            if in_code or not line.startswith("#"):
                continue
            base = self._slug(line.lstrip("#"))
            n = seen.get(base, 0)
            seen[base] = n + 1
            out.add(base if n == 0 else f"{base}-{n}")
        return out

    def check_links(self) -> None:
        problems, pages, links = [], 0, 0
        for path, rel in self.files():
            if path.suffix != ".md":
                continue
            pages += 1
            body = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.S)
            for target in LINK_RE.findall(body) + SRC_RE.findall(body):
                if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("mailto:"):
                    continue
                links += 1
                file_part, _, fragment = target.partition("#")
                dest = (path.parent / file_part).resolve() if file_part else path
                if not dest.exists():
                    problems.append(f"{rel}: dead link ({target})")
                elif not str(dest).startswith(str(self.root.resolve())):
                    problems.append(f"{rel}: link leaves the repository ({target})")
                elif fragment and dest.suffix == ".md" and fragment not in self._anchors(dest):
                    problems.append(f"{rel}: no heading #{fragment} in {dest.name}")
        self.record("markdown links", not problems, f"{pages} pages, {links} relative links and images"
                    + (f"; {len(problems)} problem(s): {problems[:6]}" if problems else ", none dead"))

    # -- scientific artifacts ---------------------------------------------------------------------------------------
    def check_artifacts(self) -> None:
        arts = self.provenance.get("scientific_artifacts") or []
        bad = []
        for a in arts:
            path = self.root / a["public_path"]
            got = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            expected = {a.get("exported_sha256"), a.get("source_sha256")}
            if a.get("frozen_sha256"):
                expected.add(a["frozen_sha256"])
            if got is None or expected != {got}:
                bad.append(a["public_path"])
        self.record("policy artifact hashes", bool(arts) and not bad,
                    f"{len(arts) - len(bad)}/{len(arts)} files byte-identical to their frozen hashes"
                    + (f"; mismatched {bad[:5]}" if bad else ""))
        code = ("from spy_vs_ai.runtime import policy_registry as pr\n"
                "n = 0\n"
                "for policy in pr.POLICY_IDS:\n"
                "    for identity in pr.IDENTITIES:\n"
                "        for personality in pr.PERSONALITIES:\n"
                "            b = pr.load_frozen(policy, pr.policy_key(identity, personality)); n += 1\n"
                "print('loaded', n)\n")
        proc = self.run(["-c", code], timeout=600)
        self.record("every policy loads (7 policies x 2 identities x 4 personalities)", proc.returncode == 0,
                    (proc.stdout.strip().splitlines() or [""])[-1] if proc.returncode == 0 else proc.stderr[-600:])

    def check_fingerprint(self) -> None:
        proc = self.run(["-c", "from workshop.scripts.tournament import code_fingerprint; print(code_fingerprint())"])
        got = proc.stdout.strip()
        expected = (self.provenance.get("source") or {}).get("game_code_fingerprint")
        self.record("game code fingerprint", proc.returncode == 0 and got == expected,
                    f"{got} (private source {expected})")

    # -- runtime ----------------------------------------------------------------------------------------------------
    def check_configs(self) -> None:
        bad = []
        configs = sorted((self.root / "configs").glob("*.yaml"))
        for cfg in configs:
            proc = self.run(["-m", "spy_vs_ai.play", "--config", str(cfg.relative_to(self.root)), "--check"], timeout=120)
            if proc.returncode != 0 or "Configuration problems" in proc.stdout:
                bad.append(cfg.name)
        self.record("example configs parse and validate", bool(configs) and not bad,
                    f"{len(configs) - len(bad)}/{len(configs)}" + (f"; failing {bad}" if bad else ""))

    def check_cli(self) -> None:
        checks = {"--help": ["--help"], "--list-policies --list-providers": ["--list-policies", "--list-providers"],
                  "--list-models fake": ["--list-models", "fake"]}
        bad = []
        for label, argv in checks.items():
            proc = self.run(["-m", "spy_vs_ai.play", *argv], timeout=120)
            if proc.returncode != 0:
                bad.append(label)
        proc = self.run(["-m", "spy_vs_ai.play", "--list-policies"], timeout=120)
        labels_ok = ("CONFIRMED / GENERALIZED RESEARCH BASELINE" in proc.stdout
                     and "NOT SAFELY PROMOTABLE" in proc.stdout)
        self.record("CLI", not bad and labels_ok, f"{len(checks) - len(bad)}/{len(checks)} commands exit 0; frozen "
                    f"Hybrid-v1 / Hybrid-v2 labels {'shown' if labels_ok else 'MISSING'}" + (f"; failing {bad}" if bad else ""))

    def check_setup_ui(self) -> None:
        code = ("import functools\n"
                "from spy_vs_ai import play\n"
                "import spy_vs_ai.ui.setup as setup\n"
                "frames = []\n"
                "original = setup.SetupScreen.run\n"
                "def run(self, *, max_frames=None):\n"
                "    out = original(self, max_frames=5); frames.append(len(self.fields)); return out\n"
                "setup.SetupScreen.run = run\n"
                "setup.provider_status = lambda *a, **k: {}\n"
                "play.prepare_display = lambda headless: None\n"
                "code = play.main([])\n"
                "assert code == 0 and frames, (code, frames)\n"
                "print('setup screen drawn, fields', frames[0])\n")
        proc = self.run(["-c", code], timeout=300)
        self.record("no-option launch opens the visual setup", proc.returncode == 0,
                    (proc.stdout.strip().splitlines() or [""])[-1] if proc.returncode == 0 else proc.stderr[-600:])

    def _match(self, name: str, argv: list[str], *, block_strands: bool = False) -> Path | None:
        out = self.work / name
        args = ["--headless", "--time-limit", "30", "--output-dir", str(out), *argv]
        if block_strands:
            code = ("import sys\n"
                    "class Block:\n"
                    "    def find_spec(self, name, path=None, target=None):\n"
                    "        if name == 'strands' or name.startswith('strands.'):\n"
                    "            raise ImportError('the Strands SDK is blocked for this check')\n"
                    "sys.meta_path.insert(0, Block())\n"
                    "from spy_vs_ai import play\n"
                    f"code = play.main({args!r})\n"
                    "assert 'strands' not in sys.modules\n"
                    "sys.exit(code)\n")
            proc = self.run(["-c", code], timeout=900)
        else:
            proc = self.run(["-m", "spy_vs_ai.play", *args], timeout=900)
        replays = sorted(out.rglob("replay_match01.json"))
        if proc.returncode != 0 or not replays:
            self.record(f"match: {name}", False, f"exit {proc.returncode}; {proc.stderr[-500:] or proc.stdout[-500:]}")
            return None
        return replays[0]

    def _replay(self, path: Path) -> tuple[bool, str]:
        proc = self.run(["-m", "spy_vs_ai.replay", str(path), "--headless", "--verify"], timeout=900)
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("[REPLAY]")]
        verdict = next((ln for ln in lines if "REPLAY REPRODUCED" in ln or "DIVERGED" in ln), lines[-1] if lines else "")
        warned = [ln for ln in lines if "warning" in ln]
        return proc.returncode == 0 and "REPLAY REPRODUCED" in proc.stdout and not warned, verdict.replace("[REPLAY] ", "")

    def check_matches(self) -> None:
        runs = {
            "call-free B2-v1 vs B2-v2": (["--black-policy", "b2_v1", "--white-policy", "b2_v2"], False),
            "fake-provider Hybrid-v1 vs Hybrid-v2 (Strands SDK blocked)": (
                ["--black-policy", "hybrid_v1", "--black-provider", "fake", "--black-model", "fake-legal",
                 "--white-policy", "hybrid_v2", "--white-provider", "fake", "--white-model", "fake-legal"], True),
        }
        for name, (argv, block) in runs.items():
            started = time.monotonic()
            replay = self._match(name.split(" (")[0].replace(" ", "_").replace("/", "_"), argv, block_strands=block)
            if replay is None:
                continue
            ok, verdict = self._replay(replay)
            self.record(f"match + saved replay + offline replay: {name}", ok,
                        f"{verdict} ({time.monotonic() - started:.0f} s)")
        secrets, _ = self.scan_tree(self.work, local_paths=False)
        self.record("secret scan of the generated traces", not secrets,
                    "no credential in the run outputs" if not secrets else f"{secrets[:5]}")

    def check_examples(self) -> None:
        examples = sorted((self.root / "examples/replays").glob("*.json"))
        bad = []
        for path in examples:
            ok, verdict = self._replay(path)
            if not ok:
                bad.append(f"{path.name}: {verdict}")
        self.record("example replays reproduce offline (identical game code)", bool(examples) and not bad,
                    f"{len(examples) - len(bad)}/{len(examples)} REPLAY REPRODUCED" + (f"; {bad}" if bad else ""))

    # -- imports ----------------------------------------------------------------------------------------------------
    def _module_file(self, name: str) -> Path | None:
        p = self.root.joinpath(*name.split("."))
        if (p / "__init__.py").is_file():
            return p / "__init__.py"
        if p.with_suffix(".py").is_file():
            return p.with_suffix(".py")
        return p if p.is_dir() else None  # a namespace package

    def _resolves(self, module: str, attribute: str | None = None) -> bool:
        parts = module.split(".")
        if any(self._module_file(".".join(parts[:i])) is None for i in range(1, len(parts) + 1)):
            return False
        if attribute is None or self._module_file(f"{module}.{attribute}") is not None:
            return True
        source = self._module_file(module)  # `from module import attribute`: defined (or re-exported) there
        return source.is_file() and re.search(rf"\b{re.escape(attribute)}\b", source.read_text(encoding="utf-8")) is not None

    def check_imports(self) -> None:
        allowed = {f: set(v) for f, v in (self.provenance.get("lazy_research_imports") or {}).items()}
        unresolved, tests_lazy = [], []
        for path, rel in self.files():
            if path.suffix != ".py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            package = list(Path(rel).with_suffix("").parts[:-1])
            for node in ast.walk(tree):
                wanted: list[tuple[str, str | None]] = []
                if isinstance(node, ast.Import):
                    wanted = [(a.name, None) for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        base = package[: len(package) - node.level + 1]
                        module = ".".join(base + ([node.module] if node.module else []))
                    else:
                        module = node.module or ""
                    wanted = [(module, a.name) for a in node.names if a.name != "*"]
                for module, attribute in wanted:
                    if module.split(".")[0] not in PROJECT_PACKAGES or self._resolves(module, attribute):
                        continue
                    full = f"{module}.{attribute}" if attribute else module
                    if full in allowed.get(rel, set()) or module in allowed.get(rel, set()):
                        continue
                    (tests_lazy if rel.startswith("tests/") else unresolved).append(f"{rel}: {full}")
        self.record("every project import resolves", not unresolved,
                    f"{sum(len(v) for v in allowed.values())} documented research-only lazy imports"
                    + (f"; unresolved: {sorted(set(unresolved))[:6]}" if unresolved else "")
                    + (f"; {len(set(tests_lazy))} lazy archive import(s) inside archive-only tests" if tests_lazy else ""))

    # -- tests ------------------------------------------------------------------------------------------------------
    def check_tests(self) -> None:
        if self.skip_tests:
            self.record("tests", True, "skipped (--skip-tests)", level="SKIP")
            return
        started = time.monotonic()
        before = {n for n in LOCAL_OUTPUTS if (self.root / n).exists()}
        proc = self.run(["-m", "pytest", "-q", "-p", "no:cacheprovider", "--no-header", "-rfEs", "tests"], timeout=5400)
        import shutil
        created = [n for n in LOCAL_OUTPUTS if (self.root / n).exists() and n not in before]
        for name in created:  # outputs the test matches wrote into the tree: not the user's
            shutil.rmtree(self.root / name, ignore_errors=True)
        summary = next((ln for ln in reversed(proc.stdout.splitlines()) if " in " in ln and ("passed" in ln or "failed" in ln)), "")
        failed = [ln for ln in proc.stdout.splitlines() if ln.startswith(("FAILED", "ERROR"))]
        skips = {}
        for ln in proc.stdout.splitlines():
            m = re.match(r"SKIPPED \[(\d+)\] [^:]+(?::\d+)?: (.*)", ln)
            if m:
                reason = "research archive" if "research archive" in m.group(2) else m.group(2)[:60]
                skips[reason] = skips.get(reason, 0) + int(m.group(1))
        self.record("tests", proc.returncode == 0, f"{summary.strip('= ')} ({time.monotonic() - started:.0f} s)"
                    + (f"; skipped: {skips}" if skips else "") + (f"; {failed[:8]}" if failed else "")
                    + (f"; removed test outputs {created}" if created else ""))

    # -- all --------------------------------------------------------------------------------------------------------
    def run_all(self) -> bool:
        print(f"validating {self.root.name}/ with {Path(self.python).name} - 0 provider calls", flush=True)
        for check in (self.check_required, self.check_forbidden, self.check_scans, self.check_links,
                      self.check_artifacts, self.check_fingerprint, self.check_configs, self.check_cli,
                      self.check_setup_ui, self.check_matches, self.check_examples, self.check_imports,
                      self.check_tests):
            try:
                check()
            except Exception as exc:  # noqa: BLE001 - a crashed check is a failed check
                self.record(check.__name__.replace("check_", ""), False, f"{type(exc).__name__}: {exc}")
        import shutil
        shutil.rmtree(self.work, ignore_errors=True)  # the traces of the matches run here
        failed = [r for r in self.results if r["status"] == "FAIL"]
        print(f"public repository validation: {'PASS' if not failed else 'FAIL'} - "
              f"{sum(r['status'] == 'PASS' for r in self.results)} passed, {len(failed)} failed, "
              f"{sum(r['status'] in ('WARN', 'SKIP') for r in self.results)} warning / skipped", flush=True)
        return not failed


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--skip-tests", action="store_true")
    p.add_argument("--python", default=sys.executable, help="the interpreter to validate with (default: this one)")
    p.add_argument("--json", metavar="PATH", help="also write the results as JSON")
    args = p.parse_args(argv)
    v = Validator(Path(args.root).resolve(), skip_tests=args.skip_tests, python=args.python)
    ok = v.run_all()
    if args.json:
        Path(args.json).write_text(json.dumps(v.results, indent=1) + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
