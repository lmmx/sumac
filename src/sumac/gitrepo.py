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


def remote_writer_ids(path: Path, remote: str) -> set[str]:
    """Writer ids with a `writer/*` ref under `refs/remotes/<remote>/` — scoped
    to one remote, unlike `list_writer_refs` (which merges every remote's
    tracking refs together, picking one arbitrarily on a name collision).
    Used by `sumac sources`'s per-branch view (docs/journal
    2026-09-16-sources-per-branch-visibility-gap.md) to know which writer
    branches a specific remote actually has, regardless of what's local."""
    prefix = f"refs/remotes/{remote}/writer/"
    result = _run(path, ["for-each-ref", "--format=%(refname)", f"refs/remotes/{remote}"])
    return {
        line.strip().removeprefix(prefix)
        for line in result.stdout.splitlines()
        if line.strip().startswith(prefix)
    }


def local_writer_ids(path: Path) -> set[str]:
    """Writer ids with a local `refs/heads/writer/*` branch."""
    result = _run(path, ["for-each-ref", "--format=%(refname)", "refs/heads/writer"])
    return {
        line.strip().removeprefix("refs/heads/writer/")
        for line in result.stdout.splitlines()
        if line.strip().startswith("refs/heads/writer/")
    }


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


def remote_names(path: Path) -> list[str]:
    """Names of every remote configured in this repo. See docs/journal
    2026-09-13-sumac-sources-design.md §3."""
    result = _run(path, ["remote"])
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def remote_url(path: Path, name: str) -> str | None:
    """The fetch URL for a configured remote, or `None` if it has none (shouldn't
    happen for a remote `remote_names` just listed, but `git remote get-url` can
    still fail on a malformed config). See docs/journal
    2026-09-13-sumac-sources-design.md §3."""
    result = _run(path, ["remote", "get-url", name], check=False)
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def fast_forward_to(path: Path, ref: str) -> None:
    """Fast-forward the checked-out branch to `ref` (a remote-tracking ref) and
    update the worktree to match, or raise `GitError` if that isn't possible
    (e.g. local has diverged commits `ref` doesn't). `--ff-only` guarantees this
    never invents a merge commit or silently drops local work — it either
    succeeds cleanly or fails. See docs/journal
    2026-09-15-read-path-freshness.md §2 (own-branch resync before a read)."""
    _run(path, ["merge", "--ff-only", ref])


def fast_forward_branch(path: Path, branch: str, ref: str) -> None:
    """Fast-forward `branch` (any local branch, not necessarily checked out) to
    `ref`, or raise `GitError` if that isn't a clean fast-forward. `git merge
    --ff-only` only ever operates on `HEAD`, so a branch that isn't currently
    checked out needs a different mechanism: `git fetch . <ref>:<branch>` — a
    plain (non-forced) refspec fetched from the repo itself, which git rejects
    outright as `[rejected] ... (non-fast-forward)` unless `branch` is a strict
    ancestor of `ref`. Confirmed by hand in both directions before relying on
    this. `branch` must not be the currently checked-out branch — git itself
    refuses that ("refusing to fetch into branch ... checked out"); use
    `fast_forward_to` for the checked-out branch instead. See docs/journal
    2026-09-16-mirror-other-writer-branch-convergence.md §4 (resyncing every
    writer's branch on a mirror, not just the current writer's own)."""
    _run(path, ["fetch", ".", f"{ref}:{branch}"])


def ref_summary(path: Path, ref: str) -> str | None:
    """`"<short hash> <relative time>"` for `ref`'s tip commit, or `None` if
    `ref` doesn't resolve. Used by `sumac sources`'s per-branch view (docs/journal
    2026-09-16-sources-per-branch-visibility-gap.md) to show what a fetch
    actually pulled for a writer branch, without inventing any health verdict
    about it."""
    result = _run(path, ["log", "-1", "--format=%h %cr", ref], check=False)
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


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
