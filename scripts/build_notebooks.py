"""Build Mathematica notebooks from the Wolfram scripts in Mathematica/, and check that they are current.

Every notebook is generated from its script: the script stays the source of truth, and a notebook is never
edited by hand. What is built, and how, is declared in ``Mathematica/notebooks.yml``; what each notebook was
built from is recorded in ``Mathematica/notebooks.lock.json``.

Usage (from the repository root, in the conda environment)::

    python scripts/build_notebooks.py check [--release]
    python scripts/build_notebooks.py reads SCRIPT.wl
    python scripts/build_notebooks.py build SCRIPT.wl ... [--cold-cache]
    python scripts/build_notebooks.py build --release --cold-cache

``check`` lists every notebook that is stale (its script, a file the script reads, its configuration entry or
the build tools changed since it was built). It exits 0 with warnings; with ``--release`` it exits 1 on any
stale notebook or any notebook that was not built from a cold moment cache.

``reads`` prints the files a script reads, followed through every file it loads.

``build`` converts each named script in a scratch directory holding only the files it reads, evaluates the
notebook there when its entry says so, checks the result and copies the notebook back. Files the script
writes stay in the scratch directory; any that differ from the committed copies are reported.
"""

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import NoReturn

import yaml

ROOT = Path(__file__).resolve().parent.parent
MATHEMATICA = "Mathematica"
CONFIG_NAME = "notebooks.yml"
LOCK_NAME = "notebooks.lock.json"
TOOLS = ("Mathematica/wl2nb.wls", "Mathematica/runnb.wls")

KINDS = ("script", "library", "data", "tool")
BUILT_KINDS = ("script", "library")
SCRIPT_SUFFIXES = {".wl", ".m", ".wls"}
READ_SUFFIXES = SCRIPT_SUFFIXES | {
    ".json",
    ".jsonl",
    ".mx",
    ".csv",
    ".txt",
    ".dat",
    ".tsv",
}

ENTRY_EXAMPLE = """\
notebooks:
  ContinuumLimit_Reference.wl:
    kind: script            # script | library | data | tool
    release: true           # listed in the reproduction manifest
    evaluate: true          # save the outputs in the notebook
    args: []                # command-line arguments the notebook run passes
    reads: {}               # "File.wl: call" -> files, for reads the builder cannot resolve
    allowed_messages: []    # kernel messages that may appear, e.g. NIntegrate::slwcon
    extra_cells: []         # code appended as input cells after the script
    intro: null             # text placed under the title
    timeout_seconds: 3600
    skip: null              # or a reason the script gets no notebook
  CubeScalarMoments.m:
    kind: data
    release: true"""


class ConfigError(ValueError):
    """A configuration error, with what is wrong, where, what it should look like and how to fix it."""


def _fail(what: str, where: Path, key: str, should: str, fix: str) -> NoReturn:
    raise ConfigError(
        f"{what}\n"
        f"  What: {what}\n"
        f"  Where: {where}, key {key}\n"
        f"  Should look like:\n{_indent(should)}\n"
        f"  Fix: {fix}"
    )


def _indent(text: str, by: int = 6) -> str:
    return "\n".join(" " * by + line for line in text.splitlines())


# --- configuration -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Entry:
    """One file of Mathematica/ and how its notebook is built."""

    name: str
    kind: str
    release: bool
    evaluate: bool = False
    args: tuple[str, ...] = ()
    reads: dict[str, tuple[str, ...]] = field(default_factory=dict)
    allowed_messages: tuple[str, ...] = ()
    extra_cells: tuple[str, ...] = ()
    intro: str | None = None
    timeout_seconds: int = 0
    skip: str | None = None

    @property
    def builds_notebook(self) -> bool:
        return self.kind in BUILT_KINDS and self.skip is None

    @property
    def notebook(self) -> str:
        return str(Path(self.name).with_suffix(".nb"))

    def digest(self) -> str:
        """SHA-256 of the entry's settings, so a change to any of them makes the notebook stale."""
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()


@dataclass(frozen=True)
class Config:
    path: Path
    max_notebook_bytes: int
    require_every_script_listed: bool
    entries: dict[str, Entry]


_BUILT_KEYS = {
    "kind": str,
    "release": bool,
    "evaluate": bool,
    "args": list,
    "reads": dict,
    "allowed_messages": list,
    "extra_cells": list,
    "intro": (str, type(None)),
    "timeout_seconds": int,
    "skip": (str, type(None)),
}
_PLAIN_KEYS = {"kind": str, "release": bool}
_TOP_KEYS = {
    "max_notebook_bytes": int,
    "require_every_script_listed": bool,
    "notebooks": dict,
}


def _type_name(t: type | tuple[type, ...]) -> str:
    names = [x.__name__ for x in (t if isinstance(t, tuple) else (t,))]
    return " or ".join("null" if n == "NoneType" else n for n in names)


def _check_keys(
    data: dict, spec: dict, path: Path, prefix: str, fix_where: str
) -> None:
    missing = [k for k in spec if k not in data]
    unknown = [k for k in data if k not in spec]
    if missing:
        _fail(
            f"missing key(s) {', '.join(missing)}",
            path,
            f"{prefix}{missing[0]}",
            ENTRY_EXAMPLE,
            f"add {', '.join(missing)} {fix_where}; every key is required, an empty value is written explicitly",
        )
    if unknown:
        _fail(
            f"unknown key(s) {', '.join(map(str, unknown))}",
            path,
            f"{prefix}{unknown[0]}",
            ENTRY_EXAMPLE,
            f"remove or correct {', '.join(map(str, unknown))} {fix_where}; allowed keys: {', '.join(spec)}",
        )
    for k, t in spec.items():
        v = data[k]
        # a YAML bool is an int in Python: refuse it where a count is wanted
        if not isinstance(v, t) or (t is int and isinstance(v, bool)):
            _fail(
                f"{prefix}{k} is {v!r} ({type(v).__name__}), expected {_type_name(t)}",
                path,
                f"{prefix}{k}",
                ENTRY_EXAMPLE,
                f"set {k} to a value of type {_type_name(t)} {fix_where}",
            )


def _string_list(value: list, path: Path, key: str) -> tuple[str, ...]:
    if not all(isinstance(x, str) for x in value):
        _fail(
            f"{key} must be a list of strings, got {value!r}",
            path,
            key,
            ENTRY_EXAMPLE,
            f"quote each item of {key}",
        )
    return tuple(value)


def load_config(path: Path, root: Path = ROOT) -> Config:
    """Read and validate the notebook configuration; raise ConfigError with a full diagnostic on any fault."""
    if not path.exists():
        _fail(
            "the notebook configuration file does not exist",
            path,
            "(the whole file)",
            "max_notebook_bytes: 5000000\nrequire_every_script_listed: false\n"
            + ENTRY_EXAMPLE,
            f"create {path} listing every Wolfram file of the reproduction manifest",
        )
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as err:
        _fail(
            f"the file is not valid YAML: {err}",
            path,
            "(the whole file)",
            ENTRY_EXAMPLE,
            "fix the YAML syntax",
        )
    if not isinstance(data, dict):
        _fail(
            "the file must be a mapping at the top level",
            path,
            "(top level)",
            ENTRY_EXAMPLE,
            "rewrite as a mapping",
        )
    _check_keys(data, _TOP_KEYS, path, "", "at the top level of the file")
    if data["max_notebook_bytes"] <= 0:
        _fail(
            f"max_notebook_bytes must be positive, got {data['max_notebook_bytes']}",
            path,
            "max_notebook_bytes",
            "max_notebook_bytes: 5000000",
            "set a positive byte count",
        )
    folder = root / MATHEMATICA
    entries: dict[str, Entry] = {}
    for name, raw in data["notebooks"].items():
        prefix = f"notebooks.{name}."
        fix_where = f"under notebooks: -> {name}: in {path}"
        if not isinstance(raw, dict) or "kind" not in raw:
            _fail(
                f"entry {name} has no kind",
                path,
                f"notebooks.{name}.kind",
                ENTRY_EXAMPLE,
                f"add kind {fix_where}",
            )
        if raw["kind"] not in KINDS:
            _fail(
                f"{prefix}kind is {raw['kind']!r}; allowed: {', '.join(KINDS)}",
                path,
                f"{prefix}kind",
                ENTRY_EXAMPLE,
                f"set kind to one of {', '.join(KINDS)} {fix_where}",
            )
        if not (folder / name).is_file():
            _fail(
                f"entry {name} names a file that does not exist: {folder / name}",
                path,
                f"notebooks.{name}",
                ENTRY_EXAMPLE,
                f"correct the file name or remove the entry from {path}",
            )
        if raw["kind"] in BUILT_KINDS:
            _check_keys(raw, _BUILT_KEYS, path, prefix, fix_where)
            reads: dict[str, tuple[str, ...]] = {}
            for call, files in raw["reads"].items():
                if not isinstance(files, list):
                    _fail(
                        f"{prefix}reads[{call!r}] must be a list of files, got {files!r}",
                        path,
                        f"{prefix}reads",
                        'reads:\n  "CubeMomentStore.wl: Get[$storeFile]": [CubeScalarMoments.m]',
                        "give the files the call reads as a list, [] when it reads nothing tracked",
                    )
                reads[call] = _string_list(files, path, f"{prefix}reads")
            if raw["timeout_seconds"] <= 0:
                _fail(
                    f"{prefix}timeout_seconds must be positive",
                    path,
                    f"{prefix}timeout_seconds",
                    "timeout_seconds: 3600",
                    f"set a positive number of seconds {fix_where}",
                )
            entries[name] = Entry(
                name=name,
                kind=raw["kind"],
                release=raw["release"],
                evaluate=raw["evaluate"],
                args=_string_list(raw["args"], path, f"{prefix}args"),
                reads=reads,
                allowed_messages=_string_list(
                    raw["allowed_messages"], path, f"{prefix}allowed_messages"
                ),
                extra_cells=_string_list(
                    raw["extra_cells"], path, f"{prefix}extra_cells"
                ),
                intro=raw["intro"],
                timeout_seconds=raw["timeout_seconds"],
                skip=raw["skip"],
            )
        else:
            _check_keys(raw, _PLAIN_KEYS, path, prefix, fix_where)
            entries[name] = Entry(name=name, kind=raw["kind"], release=raw["release"])
    if data["require_every_script_listed"]:
        listed = set(entries)
        unlisted = sorted(
            p.name
            for p in folder.iterdir()
            if p.suffix in SCRIPT_SUFFIXES and p.is_file() and p.name not in listed
        )
        if unlisted:
            _fail(
                f"{len(unlisted)} Wolfram file(s) are not listed: {', '.join(unlisted[:10])}"
                + (" ..." if len(unlisted) > 10 else ""),
                path,
                "notebooks",
                ENTRY_EXAMPLE,
                "add an entry for each (kind data or tool for files that get no notebook, or skip: with a reason)",
            )
    return Config(
        path, data["max_notebook_bytes"], data["require_every_script_listed"], entries
    )


def check_against_manifest(config: Config, manifest: list[str]) -> None:
    """The entries marked release must be exactly the Wolfram files of the reproduction manifest."""
    wolfram = {n for n in manifest if Path(n).suffix in SCRIPT_SUFFIXES}
    release = {n for n, e in config.entries.items() if e.release}
    if wolfram != release:
        only_manifest = sorted(wolfram - release)
        only_config = sorted(release - wolfram)
        _fail(
            "the release entries and the reproduction manifest disagree: "
            f"in the manifest only: {only_manifest or 'none'}; marked release only: {only_config or 'none'}",
            config.path,
            "notebooks.<name>.release",
            ENTRY_EXAMPLE,
            "mark release: true exactly the Wolfram files in NOTEBOOKS of scripts/assemble_paper_repo.py, "
            "or correct that list",
        )


def reproduction_manifest() -> list[str]:
    """The Mathematica/ files the reproduction repository ships (imported here so the lists cannot drift)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import assemble_paper_repo
    finally:
        sys.path.pop(0)
    return list(assemble_paper_repo.NOTEBOOKS)


# --- reads ---------------------------------------------------------------------------------------------

_CALL = re.compile(r"(?<![A-Za-z0-9$`])(Get|Import|ReadString|ReadList|OpenRead)\s*\[")
_SPECIAL = re.compile(r'\(\*|"')


@dataclass(frozen=True)
class Call:
    """A call that reads a file: its source text and the string literals of its first argument."""

    file: Path
    line: int
    source: str
    literals: tuple[str, ...]

    @property
    def key(self) -> str:
        """How the call is named in a reads: mapping: its file and its source, whitespace collapsed."""
        return f"{self.file.name}: {' '.join(self.source.split())}"


def mask_source(text: str, where: str = "") -> tuple[str, list[tuple[int, int, str]]]:
    """Blank comments and string contents (newlines kept); return the mask and the strings outside comments.

    Each string is (start, end, content) with text[start] and text[end] the quotes.
    """
    out: list[str] = []
    strings: list[tuple[int, int, str]] = []
    i, n = 0, len(text)
    while i < n:
        m = _SPECIAL.search(text, i)
        if m is None:
            out.append(text[i:])
            break
        out.append(text[i : m.start()])
        j = m.start()
        if text.startswith("(*", j):
            depth, k = 0, j
            while k < n:
                if text.startswith("(*", k):
                    depth += 1
                    k += 2
                elif text.startswith("*)", k):
                    depth -= 1
                    k += 2
                    if depth == 0:
                        break
                else:
                    k += 1
            if depth:
                raise ValueError(
                    f"unterminated comment starting at line {text.count(chr(10), 0, j) + 1} {where}"
                )
            out.append(re.sub(r"[^\n]", " ", text[j:k]))
            i = k
        else:
            k = j + 1
            while k < n and text[k] != '"':
                k += 2 if text[k] == "\\" else 1
            if k >= n:
                raise ValueError(
                    f"unterminated string starting at line {text.count(chr(10), 0, j) + 1} {where}"
                )
            strings.append((j, k, text[j + 1 : k]))
            out.append('"' + re.sub(r"[^\n]", " ", text[j + 1 : k]) + '"')
            i = k + 1
    return "".join(out), strings


def _close(mask: str, open_at: int) -> int:
    depth = 0
    for k in range(open_at, len(mask)):
        c = mask[k]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                return k
    raise ValueError(f"unbalanced brackets from offset {open_at}")


def find_read_calls(path: Path) -> list[Call]:
    """Every Get/Import/ReadString/ReadList/OpenRead call in the file, outside comments and strings."""
    text = path.read_text(errors="replace")
    mask, strings = mask_source(text, f"in {path}")
    calls = []
    for m in _CALL.finditer(mask):
        open_at = m.end() - 1
        close = _close(mask, open_at)
        # the first argument ends at the first comma at the call's own depth
        depth, end = 0, close
        for k in range(open_at + 1, close):
            c = mask[k]
            if c in "([{":
                depth += 1
            elif c in ")]}":
                depth -= 1
            elif c == "," and depth == 0:
                end = k
                break
        lits = tuple(s for a, b, s in strings if a > open_at and b < end)
        calls.append(
            Call(
                path,
                text.count("\n", 0, m.start()) + 1,
                text[m.start() : close + 1],
                lits,
            )
        )
    return calls


def resolve_call(call: Call) -> Path | None:
    """The file a call reads, when its first argument names it in string literals; else None.

    The literals are joined as path components relative to the directory of the file holding the call
    (the scripts build every path from DirectoryName[$InputFileName]); the last must carry a file suffix.
    """
    if not call.literals or Path(call.literals[-1]).suffix not in READ_SUFFIXES:
        return None
    joined = Path(*call.literals)
    return joined if joined.is_absolute() else (call.file.parent / joined)


@dataclass
class ReadSet:
    """The files a script reads, followed through every file it loads, and the problems found on the way."""

    files: list[Path]
    problems: list[str]


def read_set(entry: Entry, root: Path = ROOT) -> ReadSet:
    """The script and every file it reads, transitively; relative paths from the repository root."""
    folder = (root / MATHEMATICA).resolve()
    start = folder / entry.name
    seen: list[Path] = [start]
    problems: list[str] = []
    used_keys: set[str] = set()
    queue = [start]
    while queue:
        current = queue.pop(0)
        if current.suffix not in SCRIPT_SUFFIXES:
            continue
        for call in find_read_calls(current):
            if call.key in entry.reads:
                used_keys.add(call.key)
                targets = [(current.parent / f) for f in entry.reads[call.key]]
            else:
                target = resolve_call(call)
                if target is None:
                    problems.append(
                        f"cannot tell which file this call reads ({current.name} line {call.line}): {call.source}\n"
                        f"    declare it under notebooks: -> {entry.name}: -> reads: as\n"
                        f'    "{call.key}": [FILE, ...]   ([] when it reads nothing tracked, e.g. the moment cache)'
                    )
                    continue
                targets = [target]
            for t in targets:
                t = t.resolve()
                if not t.is_relative_to(root.resolve()):
                    problems.append(
                        f"{current.name} line {call.line} reads outside the repository: {t}"
                    )
                elif not t.is_file():
                    problems.append(
                        f"{current.name} line {call.line} reads a file that does not exist: {t.relative_to(root.resolve())}"
                        f"\n    if the script writes it before reading it, declare the call under reads: with []"
                    )
                elif t not in seen:
                    seen.append(t)
                    queue.append(t)
    for key in sorted(set(entry.reads) - used_keys):
        problems.append(f"reads: key matches no call (stale entry?): {key!r}")
    rel = [p.relative_to(root.resolve()) for p in seen]
    return ReadSet(rel, problems)


# --- provenance ----------------------------------------------------------------------------------------


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def provenance(entry: Entry, root: Path = ROOT) -> dict:
    """What a notebook depends on: its read set, its configuration entry and the build tools, hashed."""
    rs = read_set(entry, root)
    if rs.problems:
        raise ConfigError(
            f"the read set of {entry.name} is incomplete\n"
            f"  What: {len(rs.problems)} problem(s):\n"
            + "\n".join(f"    - {p}" for p in rs.problems)
            + "\n"
            f"  Where: {root / MATHEMATICA / CONFIG_NAME}, key notebooks.{entry.name}.reads\n"
            f"  Should look like:\n"
            f'      reads:\n        "CubeMomentStore.wl: Get[$storeFile]": [CubeScalarMoments.m]\n'
            f"  Fix: declare each unresolved call with the files it reads, relative to the calling file"
        )
    return {
        "inputs": {str(p): sha256(root / p) for p in rs.files},
        "entry_sha256": entry.digest(),
        "tools": {t: sha256(root / t) for t in TOOLS if (root / t).exists()},
    }


def load_lock(root: Path = ROOT) -> dict:
    path = root / MATHEMATICA / LOCK_NAME
    return json.loads(path.read_text()) if path.exists() else {}


def save_lock(lock: dict, root: Path = ROOT) -> None:
    path = root / MATHEMATICA / LOCK_NAME
    path.write_text(
        json.dumps(dict(sorted(lock.items())), indent=1, sort_keys=True) + "\n"
    )


def staleness(
    config: Config, lock: dict, root: Path = ROOT, release: bool = False
) -> list[str]:
    """Why each notebook is not current; empty when every notebook matches what it was built from."""
    out = []
    folder = root / MATHEMATICA
    for name, entry in sorted(config.entries.items()):
        if not entry.builds_notebook or (release and not entry.release):
            continue
        rec = lock.get(name)
        nb = folder / entry.notebook
        if rec is None:
            out.append(f"{entry.notebook}: never built")
            continue
        if not nb.exists():
            out.append(f"{entry.notebook}: missing (the lock records a build)")
            continue
        if sha256(nb) != rec["notebook_sha256"]:
            out.append(
                f"{entry.notebook}: changed since it was built (edited by hand, or the lock is out of date)"
            )
        if rec["evaluated"] != entry.evaluate:
            out.append(
                f"{entry.notebook}: built {'with' if rec['evaluated'] else 'without'} outputs, entry says otherwise"
            )
        if release and entry.evaluate and not rec["cold_cache"]:
            out.append(
                f"{entry.notebook}: built from a warm moment cache; a release needs --cold-cache"
            )
        try:
            now = provenance(entry, root)
        except ConfigError as err:
            out.append(f"{entry.notebook}: cannot check: {str(err).splitlines()[0]}")
            continue
        if now["entry_sha256"] != rec["entry_sha256"]:
            out.append(
                f"{entry.notebook}: its notebooks.yml entry changed since it was built"
            )
        for kind in ("inputs", "tools"):
            old, new = rec[kind], now[kind]
            for f in sorted(set(old) | set(new)):
                if f not in new:
                    out.append(f"{entry.notebook}: no longer reads {f}")
                elif f not in old:
                    out.append(
                        f"{entry.notebook}: now reads {f}, which it did not when built"
                    )
                elif old[f] != new[f]:
                    out.append(f"{entry.notebook}: {f} changed since it was built")
    return out


# --- checking a built notebook ---------------------------------------------------------------------------

_FAIL = re.compile(r"\bFAIL\b")
_MESSAGE_NAME = re.compile(r"\s*([A-Za-z$][\w$`]*::[\w$]+)")
_ABSOLUTE = re.compile(
    r"(?<![\w.~])/(?:Users|home|private|tmp|var|opt|Volumes|Applications)/"
)


def check_outputs(report: dict, entry: Entry) -> list[str]:
    """Problems in an evaluation report: an exit or abort, a FAIL, a message not allowed, an absolute path.

    The report is what runnb.wls writes: {"status": "finished" | "exited" | "aborted", "exit_code": int or
    null, "cells": [{"style": ..., "text": ...}, ...]} with one item per output cell.
    """
    problems = []
    if report["status"] == "aborted":
        problems.append("the evaluation was aborted")
    elif report["status"] == "exited" and report["exit_code"] != 0:
        problems.append(f"the script exited with code {report['exit_code']}")
    for cell in report["cells"]:
        text, style = cell["text"], cell["style"]
        short = " ".join(text.split())[:240]
        if _FAIL.search(text):
            problems.append(f"FAIL in a {style} cell: {short}")
        if style == "Message":
            m = _MESSAGE_NAME.match(text)
            if m is None:
                problems.append(f"a message whose name cannot be read: {short}")
            elif m.group(1) not in entry.allowed_messages:
                problems.append(
                    f"message {m.group(1)} (not in allowed_messages): {short}"
                )
        if _ABSOLUTE.search(text):
            problems.append(f"an absolute path in a {style} cell: {short}")
    return problems


def check_notebook_file(path: Path, max_bytes: int, forbidden: list[str]) -> list[str]:
    """Problems in the saved notebook itself: its size, and any of the forbidden strings (scratch, home)."""
    problems = []
    size = path.stat().st_size
    if size > max_bytes:
        problems.append(
            f"{path.name} is {size} bytes, over max_notebook_bytes = {max_bytes}"
        )
    text = path.read_text(errors="replace")
    for s in forbidden:
        if s in text:
            problems.append(f"{path.name} contains {s!r}")
    return problems


# --- command line --------------------------------------------------------------------------------------


def _cmd_check(args: argparse.Namespace) -> int:
    config = load_config(ROOT / MATHEMATICA / CONFIG_NAME)
    check_against_manifest(config, reproduction_manifest())
    problems = staleness(config, load_lock(), release=args.release)
    for p in problems:
        print(("STALE " if args.release else "warning: stale ") + p)
    if not problems:
        print("every notebook is current")
    return 1 if (args.release and problems) else 0


def _cmd_reads(args: argparse.Namespace) -> int:
    config = load_config(ROOT / MATHEMATICA / CONFIG_NAME)
    status = 0
    for name in args.scripts:
        if name not in config.entries:
            print(f"{name}: not in {CONFIG_NAME}")
            status = 1
            continue
        rs = read_set(config.entries[name])
        print(f"{name}:")
        for f in rs.files:
            print(f"  {f}")
        for p in rs.problems:
            print(f"  PROBLEM {p}")
            status = 1
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("check", help="list stale notebooks")
    p.add_argument(
        "--release",
        action="store_true",
        help="exit 1 on any stale or warm-cache notebook",
    )
    p.set_defaults(func=_cmd_check)
    p = sub.add_parser("reads", help="print the files a script reads")
    p.add_argument("scripts", nargs="+")
    p.set_defaults(func=_cmd_reads)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as err:
        print(f"build_notebooks: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
