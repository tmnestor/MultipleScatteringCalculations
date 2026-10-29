"""The notebook builder's configuration, read discovery, provenance and output checks (no Mathematica needed)."""

import importlib.util
import json
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "build_notebooks", ROOT / "scripts" / "build_notebooks.py"
)
assert _spec is not None and _spec.loader is not None
bn = importlib.util.module_from_spec(_spec)
sys.modules["build_notebooks"] = bn
_spec.loader.exec_module(bn)

FULL = """\
    kind: script
    release: true
    evaluate: true
    args: []
    reads: {}
    allowed_messages: []
    extra_cells: []
    intro: null
    timeout_seconds: 60
    skip: null
"""


def assert_diagnostic_error(message: str, where: str) -> None:
    """The four elements: what is wrong, where to fix it, a valid example, how to recover."""
    assert "What:" in message
    assert where in message and "Where:" in message
    example = message.split("Should look like:")[1].split("Fix:")[0]
    assert ": " in example or ":\n" in example  # a YAML example, not an empty section
    assert "Fix:" in message


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "Mathematica").mkdir()
    return tmp_path


def write(root: Path, name: str, text: str) -> Path:
    p = root / "Mathematica" / name
    p.write_text(text)
    return p


def config_text(entries: dict[str, str], top: str = "") -> str:
    head = top or "max_notebook_bytes: 1000\nrequire_every_script_listed: false\n"
    return (
        head
        + "notebooks:\n"
        + "".join(f"  {n}:\n{body}" for n, body in entries.items())
    )


def load(repo: Path, text: str):
    path = write(repo, "notebooks.yml", text)
    return bn.load_config(path, repo)


# --- configuration -------------------------------------------------------------------------------------


def test_valid_config_loads(repo: Path) -> None:
    write(repo, "A.wl", "1 + 1\n")
    write(repo, "D.m", "{}\n")
    cfg = load(
        repo, config_text({"A.wl": FULL, "D.m": "    kind: data\n    release: false\n"})
    )
    assert (
        cfg.entries["A.wl"].builds_notebook and cfg.entries["A.wl"].notebook == "A.nb"
    )
    assert not cfg.entries["D.m"].builds_notebook


def test_missing_file(repo: Path) -> None:
    path = repo / "Mathematica" / "notebooks.yml"
    with pytest.raises(bn.ConfigError) as err:
        bn.load_config(path, repo)
    assert_diagnostic_error(str(err.value), str(path))


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        (FULL.replace("    skip: null\n", ""), "missing key(s) skip"),
        (FULL + "    colour: red\n", "unknown key(s) colour"),
        (FULL.replace("timeout_seconds: 60", "timeout_seconds: true"), "expected int"),
        (FULL.replace("timeout_seconds: 60", "timeout_seconds: 0"), "must be positive"),
        (
            FULL.replace("kind: script", "kind: notebook"),
            "allowed: script, library, data, tool",
        ),
        (FULL.replace("args: []", "args: [1]"), "list of strings"),
        (FULL.replace("reads: {}", 'reads: {"A.wl: Get[x]": B.wl}'), "list of files"),
    ],
)
def test_bad_entry(repo: Path, body: str, fragment: str) -> None:
    write(repo, "A.wl", "1\n")
    with pytest.raises(bn.ConfigError) as err:
        load(repo, config_text({"A.wl": body}))
    assert fragment in str(err.value)
    assert_diagnostic_error(str(err.value), str(repo / "Mathematica" / "notebooks.yml"))


def test_entry_for_missing_file(repo: Path) -> None:
    with pytest.raises(bn.ConfigError) as err:
        load(repo, config_text({"Nope.wl": FULL}))
    assert "does not exist" in str(err.value)
    assert_diagnostic_error(str(err.value), str(repo / "Mathematica" / "notebooks.yml"))


def test_every_script_listed(repo: Path) -> None:
    write(repo, "A.wl", "1\n")
    write(repo, "B.wl", "1\n")
    top = "max_notebook_bytes: 1000\nrequire_every_script_listed: true\n"
    with pytest.raises(bn.ConfigError) as err:
        load(repo, config_text({"A.wl": FULL}, top))
    assert (
        "B.wl" in str(err.value)
        and "A.wl" not in str(err.value).split("listed:")[1].split("\n")[0]
    )


def test_manifest_cross_check(repo: Path) -> None:
    write(repo, "A.wl", "1\n")
    write(repo, "B.wl", "1\n")
    cfg = load(
        repo,
        config_text(
            {"A.wl": FULL, "B.wl": FULL.replace("release: true", "release: false")}
        ),
    )
    bn.check_against_manifest(cfg, ["A.wl", "data.json"])
    with pytest.raises(bn.ConfigError) as err:
        bn.check_against_manifest(cfg, ["A.wl", "B.wl"])
    assert "in the manifest only: ['B.wl']" in str(err.value)


# --- reads ---------------------------------------------------------------------------------------------


def test_mask_source_comments_and_strings() -> None:
    text = 'a = "x(*y" (* outer (* inner "q *) still *) b = "s\\"t";'
    mask, strings = bn.mask_source(text)
    assert [s for _, _, s in strings] == ["x(*y", 's\\"t']
    assert "outer" not in mask and "inner" not in mask and "still" not in mask
    assert mask.index("b =") == text.index("b =")


def test_unterminated_comment() -> None:
    with pytest.raises(ValueError, match="unterminated comment"):
        bn.mask_source("a (* b")


def test_find_read_calls(repo: Path) -> None:
    p = write(
        repo,
        "S.wl",
        '(* Get["Commented.wl"] *)\n'
        'x = ImportString["{}", "RawJSON"]; nb = NotebookGet[];\n'
        'r = Import[DirectoryName[$InputFileName] <> "ref.json", "RawJSON"];\n'
        'Get[FileNameJoin[{DirectoryName[$InputFileName], "sub", "Lib.wl"}]];\n'
        "v = Import[f];\n",
    )
    calls = bn.find_read_calls(p)
    assert [c.literals for c in calls] == [("ref.json",), ("sub", "Lib.wl"), ()]
    assert [c.line for c in calls] == [3, 4, 5]
    assert calls[2].key == "S.wl: Import[f]"
    assert bn.resolve_call(calls[0]) == p.parent / "ref.json"
    assert bn.resolve_call(calls[1]) == p.parent / "sub" / "Lib.wl"
    assert bn.resolve_call(calls[2]) is None


def _entry(**kw):
    base = dict(
        name="S.wl", kind="script", release=True, evaluate=True, timeout_seconds=60
    )
    base.update(kw)
    return bn.Entry(**base)


def test_read_set_transitive_and_declared(repo: Path) -> None:
    write(
        repo,
        "S.wl",
        'Get[FileNameJoin[{DirectoryName[$InputFileName], "Lib.wl"}]];\nImport[base <> "x"];\n',
    )
    write(
        repo,
        "Lib.wl",
        'ref = Import[DirectoryName[$InputFileName] <> "ref.json", "RawJSON"];\n',
    )
    write(repo, "ref.json", "{}")
    write(repo, "data.json", "{}")
    rs = bn.read_set(_entry(), repo)
    assert len(rs.problems) == 1 and 'S.wl: Import[base <> "x"]' in rs.problems[0]
    rs = bn.read_set(_entry(reads={'S.wl: Import[base <> "x"]': ("data.json",)}), repo)
    assert rs.problems == []
    assert [str(f) for f in rs.files] == [
        "Mathematica/S.wl",
        "Mathematica/Lib.wl",
        "Mathematica/data.json",
        "Mathematica/ref.json",
    ]


def test_read_set_problems(repo: Path) -> None:
    write(repo, "S.wl", 'Get["Missing.wl"];\nGet["/etc/hosts.txt"];\n')
    rs = bn.read_set(_entry(reads={"S.wl: Get[nothing]": ()}), repo)
    text = "\n".join(rs.problems)
    assert "does not exist: Mathematica/Missing.wl" in text
    assert "outside the repository" in text
    assert "stale entry" in text


# --- provenance and staleness ----------------------------------------------------------------------------


def _built(repo: Path) -> tuple:
    write(
        repo,
        "A.wl",
        'Import[DirectoryName[$InputFileName] <> "ref.json", "RawJSON"];\n',
    )
    write(repo, "ref.json", "{}")
    cfg = load(repo, config_text({"A.wl": FULL}))
    write(repo, "A.nb", "Notebook[{}]")
    entry = cfg.entries["A.wl"]
    rec = bn.provenance(entry, repo) | {
        "notebook_sha256": bn.sha256(repo / "Mathematica" / "A.nb"),
        "evaluated": True,
        "cold_cache": True,
    }
    return cfg, {"A.wl": rec}


def test_current_notebook_is_not_stale(repo: Path) -> None:
    cfg, lock = _built(repo)
    assert bn.staleness(cfg, lock, repo) == []
    assert bn.staleness(cfg, lock, repo, release=True) == []


def test_never_built(repo: Path) -> None:
    cfg, _ = _built(repo)
    assert bn.staleness(cfg, {}, repo) == ["A.nb: never built"]


def test_changed_input_names_the_file(repo: Path) -> None:
    cfg, lock = _built(repo)
    write(repo, "ref.json", '{"changed": 1}')
    assert bn.staleness(cfg, lock, repo) == [
        "A.nb: Mathematica/ref.json changed since it was built"
    ]


def test_hand_edit_and_entry_change(repo: Path) -> None:
    cfg, lock = _built(repo)
    write(repo, "A.nb", 'Notebook[{Cell["edited"]}]')
    assert (
        "changed since it was built (edited by hand" in bn.staleness(cfg, lock, repo)[0]
    )
    cfg2 = load(
        repo,
        config_text(
            {"A.wl": FULL.replace("timeout_seconds: 60", "timeout_seconds: 61")}
        ),
    )
    write(repo, "A.nb", "Notebook[{}]")
    assert bn.staleness(cfg2, lock, repo) == [
        "A.nb: its notebooks.yml entry changed since it was built"
    ]


def test_release_refuses_warm_cache(repo: Path) -> None:
    cfg, lock = _built(repo)
    lock["A.wl"]["cold_cache"] = False
    assert bn.staleness(cfg, lock, repo) == []
    assert bn.staleness(cfg, lock, repo, release=True) == [
        "A.nb: built from a warm moment cache; a release needs --cold-cache"
    ]


def test_lock_round_trip(repo: Path) -> None:
    _, lock = _built(repo)
    bn.save_lock(lock, repo)
    assert bn.load_lock(repo) == json.loads(json.dumps(lock))


# --- output checks -------------------------------------------------------------------------------------


def _report(status="finished", code=None, cells=()):
    return {
        "status": status,
        "exit_code": code,
        "cells": [{"style": s, "text": t} for s, t in cells],
    }


def test_clean_report_passes() -> None:
    rep = _report(
        cells=[("Print", "  [a] identity: rel 1.2e-16  PASS"), ("Output", "3.14159")]
    )
    assert bn.check_outputs(rep, _entry()) == []
    assert bn.check_outputs(_report("exited", 0, [("Print", "done")]), _entry()) == []


def test_script_without_outputs_fails_but_library_may_be_silent() -> None:
    assert bn.check_outputs(_report(), _entry()) == [
        "no output cells: an evaluated script must show its outputs"
    ]
    assert bn.check_outputs(_report(), _entry(kind="library")) == []


@pytest.mark.parametrize(
    ("report", "fragment"),
    [
        (_report("aborted", None, [("Print", "started")]), "aborted"),
        (_report("exited", 1, [("Print", "started")]), "exited with code 1"),
        (_report(cells=[("Print", "check ****FAIL****")]), "FAIL in a Print cell"),
        (
            _report(
                cells=[("Message", "Power::infy: Infinite expression 1/0 encountered.")]
            ),
            "message Power::infy",
        ),
        (_report(cells=[("Message", "garbled")]), "cannot be read"),
        (_report(cells=[("Print", "wrote /Users/someone/x.json")]), "absolute path"),
        (
            _report(
                cells=[
                    ("Print", "wrote /private/var/folders/x/T/tmpab/Mathematica/x.json")
                ]
            ),
            "absolute path",
        ),
    ],
)
def test_report_problems(report: dict, fragment: str) -> None:
    problems = bn.check_outputs(report, _entry())
    assert len(problems) == 1 and fragment in problems[0]


def test_allowed_message_and_no_false_positives() -> None:
    rep = _report(
        cells=[
            (
                "Message",
                "NIntegrate::slwcon: Numerical integration converging too slowly.",
            ),
            ("Print", "FAILURE is not the marker; nor is PASSFAIL, nor a ratio a/b/c"),
        ]
    )
    assert bn.check_outputs(rep, _entry(allowed_messages=("NIntegrate::slwcon",))) == []


def test_notebook_file_checks(repo: Path) -> None:
    nb = write(repo, "A.nb", 'Notebook[{Cell["/scratch/abc/Mathematica"]}]')
    assert bn.check_notebook_file(nb, 10_000, []) == []
    problems = bn.check_notebook_file(nb, 10, ["/scratch/abc"])
    assert (
        len(problems) == 2
        and "over max_notebook_bytes" in problems[0]
        and "/scratch/abc" in problems[1]
    )


# --- the repository's own configuration ----------------------------------------------------------------


def test_repository_config_is_valid_and_complete() -> None:
    cfg = bn.load_config(ROOT / "Mathematica" / "notebooks.yml")
    bn.check_against_manifest(cfg, bn.reproduction_manifest())
    for entry in cfg.entries.values():
        if entry.builds_notebook:
            assert bn.read_set(entry).problems == [], entry.name


def test_repository_notebooks_current() -> None:
    """The commit gate only warns about stale notebooks; the release check refuses them."""
    cfg = bn.load_config(ROOT / "Mathematica" / "notebooks.yml")
    for problem in bn.staleness(cfg, bn.load_lock()):
        warnings.warn(f"stale notebook: {problem}", stacklevel=1)


# --- build helpers (no Mathematica) -------------------------------------------------------------------


def test_run_stops_at_timeout(tmp_path: Path) -> None:
    code, _ = bn._run(
        [sys.executable, "-c", "import time; time.sleep(30)"], tmp_path, 1
    )
    assert code is None
    code, out = bn._run([sys.executable, "-c", "print(42)"], tmp_path, 30)
    assert code == 0 and out.strip() == "42"


def test_written_files_compares_with_committed(tmp_path: Path) -> None:
    root, scratch = tmp_path / "repo", tmp_path / "scratch"
    for d in (root / "Mathematica", scratch / "Mathematica" / "cache"):
        d.mkdir(parents=True)
    (root / "Mathematica" / "same.json").write_text("1")
    (root / "Mathematica" / "diff.json").write_text("1")
    (scratch / "Mathematica" / "input.wl").write_text("x")
    before = bn._snapshot(scratch)
    (scratch / "Mathematica" / "same.json").write_text("1")
    (scratch / "Mathematica" / "diff.json").write_text("2")
    (scratch / "Mathematica" / "new.json").write_text("3")
    (scratch / "Mathematica" / "cache" / "E_1.mx").write_text("c")
    (scratch / "Mathematica" / "A.nb").write_text("nb")
    lines = bn.written_files(before, bn._snapshot(scratch), root, {"Mathematica/A.nb"})
    assert lines == [
        "Mathematica/diff.json: DIFFERS from the committed copy",
        "Mathematica/new.json: not in the repository",
        "Mathematica/same.json: identical to the committed copy",
    ]
