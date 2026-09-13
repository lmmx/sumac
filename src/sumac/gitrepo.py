"""Every git subprocess call in sumac lives here — nowhere else. See docs/journal
2026-09-11-branch-per-user-design.md §2, §3, §5.
"""

from __future__ import annotations

import subprocess
from pathlib import Path, PurePosixPath

from sumac.errors import GitError

_GPG_OFF = ["-c", "commit.gpgsign=false"]


def _run(path: Path, args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    cmd = ["git", "-C", str(path), *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise GitError(f"{' '.join(cmd)}: {result.stderr.strip()}")
    return result


def is_repo(path: Path) -> bool:
    result = _run(path, ["rev-parse", "--is-inside-work-tree"], check=False)
    return result.returncode == 0 and result.stdout.strip() == "true"


def toplevel(path: Path) -> Path | None:
    if not is_repo(path):
        return None
    result = _run(path, ["rev-parse", "--show-toplevel"])
    return Path(result.stdout.strip())


def current_branch(path: Path) -> str | None:
    result = _run(path, ["symbolic-ref", "-q", "--short", "HEAD"], check=False)
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def rel_to_toplevel(path: Path) -> PurePosixPath:
    top = toplevel(path)
    if top is None:
        raise GitError(f"{path} is not inside a git repo")
    rel = path.resolve().relative_to(top.resolve())
    return PurePosixPath(rel.as_posix())


def list_writer_refs(path: Path) -> dict[str, str]:
    """{writer_id: ref}, local `refs/heads/writer/*` plus `refs/remotes/*/writer/*`;
    local wins on a collision."""
    result = _run(path, ["for-each-ref", "--format=%(refname)", "refs/remotes", "refs/heads"])
    remotes: dict[str, str] = {}
    locals_: dict[str, str] = {}
    for line in result.stdout.splitlines():
        ref = line.strip()
        if not ref:
            continue
        if ref.startswith("refs/heads/writer/"):
            writer_id = ref.removeprefix("refs/heads/writer/")
            locals_[writer_id] = ref
        elif ref.startswith("refs/remotes/"):
            tail = ref.removeprefix("refs/remotes/")
            parts = tail.split("/", 1)
            if len(parts) == 2 and parts[1].startswith("writer/"):
                writer_id = parts[1].removeprefix("writer/")
                remotes.setdefault(writer_id, ref)
    return {**remotes, **locals_}


def read_blob(path: Path, ref: str, rel: str) -> str:
    result = _run(path, ["cat-file", "blob", f"{ref}:{rel}"], check=False)
    if result.returncode != 0:
        return ""
    return result.stdout


def init_repo(path: Path, *, initial_branch: str | None = None) -> None:
    args = ["init"] if initial_branch is None else ["init", "-b", initial_branch]
    _run(path, args)


def orphan_checkout(path: Path, branch: str) -> None:
    _run(path, ["checkout", "--orphan", branch])


def root_commit(path: Path, ref: str = "HEAD") -> str:
    result = _run(path, ["rev-list", "--max-parents=0", ref])
    return result.stdout.strip().splitlines()[0]


def create_branch_from(path: Path, branch: str, commit: str, *, checkout: bool) -> None:
    args = ["checkout", "-B", branch, commit] if checkout else ["branch", branch, commit]
    _run(path, args)


def commit_paths(path: Path, rels: list[str], message: str) -> None:
    _run(path, ["add", *rels])
    _run(path, [*_GPG_OFF, "commit", "-m", message])


def fetch_writers(path: Path, remote: str = "origin") -> None:
    refspec = f"refs/heads/writer/*:refs/remotes/{remote}/writer/*"
    _run(path, ["fetch", remote, refspec])


def commits_touching(path: Path, ref: str, rels: list[str]) -> list[str]:
    result = _run(path, ["log", "--reverse", "--format=%H", ref, "--", *rels])
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def read_blobs_batch(path: Path, specs: list[str]) -> dict[str, str]:
    """One `git cat-file --batch` subprocess for every `<commit>:<rel>` in `specs`,
    so a history walk (docs/journal 2026-09-11-branch-per-user-design.md §4) costs
    one subprocess per writer, not one per commit. Missing specs map to `""`."""
    if not specs:
        return {}
    cmd = ["git", "-C", str(path), "cat-file", "--batch"]
    stdin = ("\n".join(specs) + "\n").encode("utf-8")
    result = subprocess.run(cmd, input=stdin, capture_output=True)
    if result.returncode != 0:
        raise GitError(f"{' '.join(cmd)}: {result.stderr.decode('utf-8', 'replace').strip()}")
    out = result.stdout
    blobs: dict[str, str] = {}
    pos = 0
    for spec in specs:
        nl = out.index(b"\n", pos)
        header = out[pos:nl].decode("utf-8")
        pos = nl + 1
        parts = header.split()
        if parts[-1] == "missing":
            blobs[spec] = ""
            continue
        size = int(parts[2])
        blobs[spec] = out[pos : pos + size].decode("utf-8")
        pos += size + 1  # trailing newline after the payload
    return blobs
