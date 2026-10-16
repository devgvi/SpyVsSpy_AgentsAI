"""Export the public Spy vs AI repository from the private research archive (spec KIRO_PUBLIC_PRIVATE_REPOSITORY_SPLIT.md).

    python -m tools.export_public_repo --destination ../spy-vs-ai-public-build
    python -m tools.export_public_repo --destination ../spy-vs-ai-public-build --skip-tests   # faster validation
    python -m tools.export_public_repo --destination DIR --no-validate                       # export only

Runs in the private archive. Reads PUBLIC_EXPORT_MANIFEST.yaml (a whitelist) and writes a clean tree:
  * only the listed files, at the same relative path (or the manifest's `rename`), plus the public-only files of
    `public_overlay/`;
  * scientific policy artifacts byte for byte, checked against the frozen manifests before and after the copy;
  * shared pages lose their archive-only `<!-- private:begin -->` ... `<!-- private:end -->` blocks;
  * markdown links rewritten for renamed files; a link to a file that is not exported becomes its archive path as text;
  * PUBLIC_EXPORT_PROVENANCE.json (source identity, manifest hash, artifact hashes; no credential, no local path);
then scans the export for secrets and local paths and runs tools/validate_public_repo.py inside it.

The source tree is never modified. The destination must be outside it, and it is wiped first only if it is empty or
holds a previous export (it has a PUBLIC_EXPORT_PROVENANCE.json). Nothing is pushed or published.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True  # the export never writes into the source tree, not even bytecode caches
ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "PUBLIC_EXPORT_MANIFEST.yaml"
PROVENANCE = "PUBLIC_EXPORT_PROVENANCE.json"
SCHEMA = "spy-vs-ai-public-export/1"
PROVENANCE_SCHEMA = "spy-vs-ai-public-export-provenance/1"
STAGE3 = "workshop/policies/phase-a-v2/stage3-release/freeze_manifest.json"
B2V2 = "workshop/policies/stage5/b2-v2-candidate/manifest.json"
FINAL1 = "workshop/policies/final-release/freeze_manifest.json"
FINAL1_SHA256 = "sha256:0eceb995c42bb588cb49f7c8f9d94fc76b276325bdfe98d9e821cb5f0458c229"  # FINAL-1 release record
LINK_RE = re.compile(r"(!?)\[([^\]]*)\]\(([^)\s]+)\)")
SRC_RE = re.compile(r"(<(?:img|video|source)\b[^>]*\bsrc=\")([^\"]+)(\")", re.I)


class ExportError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _json(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def load_manifest(path: Path = MANIFEST) -> dict:
    try:
        import yaml
    except ImportError as exc:  # the exporter is a private-archive tool; PyYAML is in its environment
        raise ExportError("PyYAML is needed to read PUBLIC_EXPORT_MANIFEST.yaml (pip install PyYAML)") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise ExportError(f"{path.name}: schema {data.get('schema')!r}, expected {SCHEMA!r}")
    for key in ("include", "policy_artifacts", "forbidden", "overlay"):
        if key not in data:
            raise ExportError(f"{path.name}: no '{key}' section")
    return data


# -- the plan: destination path -> (source path, kind) -------------------------------------------------------------
def _excluded(rel: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch("/" + rel, p) for p in patterns) or \
        any(part in ("__pycache__",) for part in Path(rel).parts) or Path(rel).name == ".DS_Store"


def expected_artifact_hashes(manifest: dict) -> list[dict]:
    """Every scientific artifact file with its frozen hash, from the frozen manifests (never recomputed)."""
    spec = manifest["policy_artifacts"]
    s3, b2, f1 = _json(STAGE3), _json(B2V2), _json(FINAL1)
    out = []

    def add(art: dict, rel: str, frozen: str | None, note: str = "") -> None:
        out.append({"id": art["id"], "version": art["version"], "status": art["status"], "used_by": art["used_by"],
                    "frozen_in": art["frozen_in"], "source_path": rel, "public_path": rel, "frozen_sha256": frozen,
                    "note": note})

    for art in spec["artifacts"]:
        kind = art["frozen_in"]
        if "path" in art:
            for identity in spec["identities"]:
                for personality in spec["personalities"]:
                    base = art["path"].format(identity=identity, personality=personality)
                    if kind == "stage3":
                        emulator = Path(base).name
                        files = s3["policy_artifacts"][f"{identity}:{personality}"][emulator]["files"]
                    elif kind == "b2v2":
                        prefix = f"{identity}/{personality}/{Path(base).name}/"
                        files = {k[len(prefix):]: v for k, v in b2["files"].items() if k.startswith(prefix)}
                    else:
                        raise ExportError(f"artifact {art['id']}: unknown frozen manifest {kind!r}")
                    if not files:
                        raise ExportError(f"artifact {art['id']}: no frozen file list for {base}")
                    on_disk = sorted(p.name for p in (ROOT / base).iterdir() if p.is_file() and p.name != ".DS_Store")
                    if sorted(files) != on_disk:
                        raise ExportError(f"{base}: files on disk {on_disk} differ from the frozen list {sorted(files)}")
                    for name, digest in sorted(files.items()):
                        add(art, f"{base}/{name}", digest)
            continue
        for rel in art["files"]:
            if kind == "final1":
                by_path = {f1["artifacts"]["b2_v2"]["candidate"]["path"]: f1["artifacts"]["b2_v2"]["candidate"]["sha256"],
                           f1["artifacts"]["b2_v2"]["manifest"]["path"]: f1["artifacts"]["b2_v2"]["manifest"]["sha256"],
                           f1["artifacts"]["r2_1"]["router_file"]["path"]: f1["artifacts"]["r2_1"]["router_file"]["sha256"]}
                if rel not in by_path:
                    raise ExportError(f"{rel}: not hashed in the FINAL-1 freeze manifest")
                add(art, rel, by_path[rel])
            elif kind == "final1-self":
                add(art, rel, FINAL1_SHA256, "the FINAL-1 release record (its file hash, as recorded at the freeze)")
            elif kind == "stage3-dataset":
                data = _json(rel)
                if data.get("dataset_hash") != s3["datasets"]["training_dataset_hash"]:
                    raise ExportError(f"{rel}: dataset hash {data.get('dataset_hash')} is not the Stage-3 training "
                                      f"dataset hash {s3['datasets']['training_dataset_hash']}")
                add(art, rel, None, "no file hash is frozen; its dataset_hash equals the Stage-3 training dataset hash")
            else:
                raise ExportError(f"artifact {art['id']}: unknown frozen manifest {kind!r}")
    return out


def build_plan(manifest: dict) -> tuple[dict[str, tuple[Path, str]], list[dict], list[str]]:
    exclude = list(manifest.get("exclude") or [])
    rename = dict(manifest.get("rename") or {})
    plan: dict[str, tuple[Path, str]] = {}
    for group, patterns in manifest["include"].items():
        for pattern in patterns:
            matches = [p for p in sorted(ROOT.glob(pattern)) if p.is_file()]
            matches = [p for p in matches if not _excluded(p.relative_to(ROOT).as_posix(), exclude)]
            if not matches:
                raise ExportError(f"include.{group}: '{pattern}' matches no file")
            for src in matches:
                rel = src.relative_to(ROOT).as_posix()
                plan[rename.get(rel, rel)] = (src, "markdown" if src.suffix == ".md" else "copy")
    missing_renames = [r for r in rename if not (ROOT / r).is_file()]
    if missing_renames:
        raise ExportError(f"rename: no such file {missing_renames}")
    artifacts = expected_artifact_hashes(manifest)
    for a in artifacts:
        plan[a["public_path"]] = (ROOT / a["source_path"], "artifact")
    overlay_dir = ROOT / manifest["overlay"]
    overlays = []
    for src in sorted(overlay_dir.rglob("*")):
        rel = src.relative_to(overlay_dir).as_posix()
        if src.is_file() and not _excluded(rel, exclude):
            if rel in plan and plan[rel][1] == "artifact":
                raise ExportError(f"overlay {rel} would replace a scientific artifact")
            plan[rel] = (src, "overlay")
            overlays.append(rel)
    forbidden = list(manifest["forbidden"])
    clash = sorted(d for d in plan for f in forbidden if d == f or d.startswith(f.rstrip("/") + "/"))
    if clash:
        raise ExportError(f"forbidden paths in the plan: {clash[:10]}")
    return plan, artifacts, overlays


# -- markdown ------------------------------------------------------------------------------------------------------
def transform_markdown(text: str, src_rel: str, dest_rel: str, src_to_dest: dict[str, str],
                       dest_dirs: set[str], dest_paths: set[str]) -> tuple[str, list[str]]:
    """Rewrites relative links for the public tree; a link to a file that is not exported becomes its label."""
    unresolved: list[str] = []
    src_dir = (ROOT / src_rel).parent
    dest_dir = Path(dest_rel).parent

    archive_path: dict[str, str] = {}  # unresolved target -> its project-relative archive path

    def target_for(target: str) -> str | None:
        file_part, hash_, fragment = target.partition("#")
        if not file_part:
            return target
        resolved = Path(os.path.normpath(src_dir / file_part))
        try:
            rel = resolved.relative_to(ROOT).as_posix()
        except ValueError:
            return None  # outside the project (the archive's own changelog, specs, ...)
        slash = "/" if file_part.endswith("/") else ""
        if rel in src_to_dest:
            new = src_to_dest[rel]
        elif rel in dest_dirs or rel in dest_paths:  # a folder, or a page the public overlay provides
            new = rel
        else:
            archive_path[target] = rel + slash
            return None
        return os.path.relpath(new, dest_dir or ".") + slash + hash_ + fragment

    def link(m: re.Match) -> str:
        bang, label, target = m.groups()
        if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
            return m.group(0)
        new = target_for(target)
        if new is None:
            if bang:
                raise ExportError(f"{src_rel}: image {target} is not exported")
            unresolved.append(target)
            rel = archive_path.get(target)
            bare = label.strip("`")
            if rel is None:
                return label  # outside the project: the label alone
            if bare == rel.rstrip("/") or bare == Path(rel).name or bare == target:
                return f"`{rel}`"  # the label was the file name: show where it lives in the archive
            return f"{label} (`{rel}`)"
        return f"{bang}[{label}]({new})"

    def src(m: re.Match) -> str:
        new = target_for(m.group(2))
        if new is None:
            raise ExportError(f"{src_rel}: media {m.group(2)} is not exported")
        return m.group(1) + new + m.group(3)

    out, in_code = [], False
    for line in text.split("\n"):
        if line.startswith("```"):
            in_code = not in_code
        out.append(line if in_code else SRC_RE.sub(src, LINK_RE.sub(link, line)))
    return "\n".join(out), unresolved


PRIVATE_BEGIN, PRIVATE_END = "<!-- private:begin -->", "<!-- private:end -->"


def strip_private(text: str, src_rel: str) -> tuple[str, int]:
    """Removes the archive-only blocks of a shared page (markers included). Unbalanced markers stop the export."""
    out, inside, blocks = [], False, 0
    for n, line in enumerate(text.split("\n"), 1):
        marker = line.strip()
        if marker == PRIVATE_BEGIN:
            if inside:
                raise ExportError(f"{src_rel}:{n}: nested {PRIVATE_BEGIN}")
            inside, blocks = True, blocks + 1
            continue
        if marker == PRIVATE_END:
            if not inside:
                raise ExportError(f"{src_rel}:{n}: {PRIVATE_END} without {PRIVATE_BEGIN}")
            inside = False
            continue
        if not inside:
            out.append(line)
    if inside:
        raise ExportError(f"{src_rel}: {PRIVATE_BEGIN} is never closed")
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).rstrip("\n") + "\n", blocks


def insert_note(text: str, note: str) -> str:
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("# "):
            return "\n".join(lines[: i + 1] + ["", f"> {note.strip()}", ""] + lines[i + 1:])
    return f"> {note.strip()}\n\n{text}"


# -- export --------------------------------------------------------------------------------------------------------
def prepare_destination(dest: Path) -> None:
    if dest == ROOT or ROOT in dest.parents or dest in ROOT.parents:
        raise ExportError(f"the destination must be outside the private archive: {dest}")
    if dest.exists():
        entries = [p for p in dest.iterdir() if p.name != ".DS_Store"]
        if entries and not (dest / PROVENANCE).is_file():
            raise ExportError(f"{dest} is not empty and is not a previous export (no {PROVENANCE}); refusing to wipe it")
        shutil.rmtree(dest)
    dest.mkdir(parents=True)


def source_tree_hash(plan: dict[str, tuple[Path, str]]) -> str:
    digest = hashlib.sha256()
    for dest_rel in sorted(plan):
        src, _ = plan[dest_rel]
        digest.update(dest_rel.encode())
        digest.update(src.read_bytes())
    return "sha256:" + digest.hexdigest()[:16]


def export(dest: Path, manifest: dict) -> dict:
    plan, artifacts, overlays = build_plan(manifest)
    for a in artifacts:  # before the copy: the private source still equals the freeze
        a["source_sha256"] = sha256(ROOT / a["source_path"])
        if a["frozen_sha256"] and a["source_sha256"] != a["frozen_sha256"]:
            raise ExportError(f"{a['source_path']}: {a['source_sha256']} is not the frozen {a['frozen_sha256']}")
    prepare_destination(dest)
    rename = dict(manifest.get("rename") or {})
    src_to_dest = {src.relative_to(ROOT).as_posix(): d for d, (src, kind) in plan.items() if kind != "overlay"}
    dest_dirs = {Path(d).parents[i].as_posix() for d in plan for i in range(len(Path(d).parents) - 1)}
    notes = (manifest.get("markdown") or {}).get("notes") or {}
    unresolved: dict[str, int] = {}
    private_blocks: dict[str, int] = {}
    for dest_rel, (src, kind) in sorted(plan.items()):
        out = dest / dest_rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if kind == "markdown":
            src_rel = src.relative_to(ROOT).as_posix()
            text, blocks = strip_private(src.read_text(encoding="utf-8"), src_rel)
            if blocks:
                private_blocks[dest_rel] = blocks
            text, lost = transform_markdown(text, src_rel, dest_rel, src_to_dest, dest_dirs, set(plan))
            if src_rel in notes:
                text = insert_note(text, notes[src_rel])
            if lost:
                unresolved[dest_rel] = len(lost)
            out.write_text(text, encoding="utf-8")
        else:
            shutil.copyfile(src, out)  # artifacts, overlays, code, data and media: byte for byte
    for a in artifacts:  # after the copy
        a["exported_sha256"] = sha256(dest / a["public_path"])
        if a["exported_sha256"] != a["source_sha256"]:
            raise ExportError(f"{a['public_path']}: the copy differs from the source")
        a["verified"] = True
    from workshop.scripts.tournament import code_fingerprint
    files = [p for p in dest.rglob("*") if p.is_file()]
    by_top: dict[str, int] = {}
    for p in files:
        top = p.relative_to(dest).parts[0]
        by_top[top] = by_top.get(top, 0) + 1
    tree_hash = source_tree_hash(plan)
    paper_src = next((s for s, d in rename.items() if d.startswith("paper/")), None)
    provenance = {
        "schema": PROVENANCE_SCHEMA,
        "export_schema": manifest["schema"],
        "public_repository": manifest.get("public_repository"),
        "exported_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "git_commit": None,
            "git_note": "the private research archive is not a git repository; source_tree_hash identifies the exported "
                        "source files (their public paths and bytes, before the markdown link rewrite)",
            "source_tree_hash": tree_hash,
            "game_code_fingerprint": code_fingerprint(),
        },
        "manifest": {"path": MANIFEST.name, "sha256": sha256(MANIFEST)},
        "documentation_version": manifest.get("documentation_version"),
        "paper": {"version": manifest.get("paper_version"), "public_path": rename.get(paper_src),
                  "sha256": sha256(ROOT / paper_src) if paper_src else None},
        "research_artifact_version": manifest.get("research_artifact_version"),
        "version_mapping": {"private_source_tree": tree_hash, "public_release": "not tagged yet (planned "
                            f"{manifest.get('planned_public_tag')})", "research_artifact": manifest.get("research_artifact_version"),
                            "paper": manifest.get("paper_version")},
        "files": {"count": len(files) + 1, "bytes": sum(p.stat().st_size for p in files), "by_top_level": by_top},
        "renamed": rename,
        "overlays": overlays,
        "markdown_links_made_text": unresolved,
        "archive_only_blocks_removed": private_blocks,
        "scientific_artifacts": artifacts,
        "policy_composition": manifest["policy_artifacts"].get("composition"),
        "lazy_research_imports": manifest.get("lazy_research_imports") or {},
        "tests_archive_only": (manifest.get("tests") or {}).get("archive_only") or {},
        "tests_environment_dependent": (manifest.get("tests") or {}).get("environment_dependent") or {},
        "forbidden": manifest["forbidden"],
        "provider_calls": 0,
    }
    (dest / PROVENANCE).write_text(json.dumps(provenance, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    return provenance


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--destination", required=True, help="a clean directory outside this archive")
    p.add_argument("--manifest", default=str(MANIFEST))
    p.add_argument("--no-validate", action="store_true", help="export only")
    p.add_argument("--skip-tests", action="store_true", help="validate everything but the test suite")
    args = p.parse_args(argv)
    os.chdir(ROOT)
    dest = Path(args.destination).expanduser().resolve()
    try:
        manifest = load_manifest(Path(args.manifest))
        prov = export(dest, manifest)
    except ExportError as exc:
        print(f"[EXPORT] STOPPED - {exc}", file=sys.stderr)
        return 2
    arts = prov["scientific_artifacts"]
    print(f"[EXPORT] {prov['files']['count']} files, {prov['files']['bytes'] / 1e6:.1f} MB -> {dest.name}/")
    print(f"[EXPORT] scientific artifacts: {len(arts)} files byte-identical to their frozen hashes")
    print(f"[EXPORT] markdown links to archive-only files made plain text: {sum(prov['markdown_links_made_text'].values())} "
          f"in {len(prov['markdown_links_made_text'])} page(s)")
    print(f"[EXPORT] archive-only blocks removed: {sum(prov['archive_only_blocks_removed'].values())} "
          f"in {len(prov['archive_only_blocks_removed'])} page(s)")
    print(f"[EXPORT] source tree {prov['source']['source_tree_hash']}, game code {prov['source']['game_code_fingerprint']}")
    if args.no_validate:
        return 0
    cmd = [sys.executable, "-m", "tools.validate_public_repo", "--root", str(dest)] + (["--skip-tests"] if args.skip_tests else [])
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(cmd, cwd=dest, env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
