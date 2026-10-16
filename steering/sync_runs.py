"""Upload run recordings and analysis outputs to the group's private Hugging Face dataset repo.

Data stays out of git (decision of 16 Oct): code and docs live in the fork, recordings and results in the
private dataset repo ``pinhofst10d/so-101-steering`` (members of the pinhofst10d organisation only).

Layout in the repo, mirroring ``steering/results/``:

    runs/<stamp>_<tag>/          rollout recordings (ticks.npz, chunks.npz, frames/, meta.json, plots/, replay*.mp4)
    server_runs/<stamp>_<tag>/   MolmoAct2 server recordings (the colleague's interface)
    analyses/<name>/             outputs of analysis scripts (new ones go here, not into git)

Uploads are resumable and skip files whose content is already in the repo (``upload_large_folder`` hashes
every file, so changed files are sent too), so running it after every session is cheap. ``--download`` only
fetches files missing locally unless ``--overwrite`` is given. Needs ``uv run hf auth login`` once, with a token that can write to the organisation.

    uv run python steering/sync_runs.py --dry-run   # what would go up
    uv run python steering/sync_runs.py             # upload runs/, server_runs/, analyses/
    uv run python steering/sync_runs.py --download  # fetch everything into steering/results/ (a new machine)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO_ID = "pinhofst10d/so-101-steering"
RESULTS = Path(__file__).resolve().parent / "results"
FOLDERS = ("runs", "server_runs", "analyses")
IGNORE = ["**/__pycache__/**", "**/.cache/**", "**/*.tmp", "**/.DS_Store"]


def local_files() -> list[Path]:
    """Every file under the synced folders, relative to results/."""
    out = []
    for name in FOLDERS:
        root = RESULTS / name
        if root.is_dir():
            out += [
                p.relative_to(RESULTS)
                for p in root.rglob("*")
                if p.is_file()
                and "__pycache__" not in p.parts
                and ".cache" not in p.parts
                and p.suffix != ".tmp"
                and p.name != ".DS_Store"
            ]
    return sorted(out)


def main() -> None:
    """Upload (default), list (--dry-run) or download (--download) the synced folders."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", default=REPO_ID, help="dataset repo id")
    ap.add_argument(
        "--dry-run", action="store_true", help="list local files missing from the repo; upload nothing"
    )
    ap.add_argument(
        "--download", action="store_true", help="download repo files missing locally (never overwrites)"
    )
    ap.add_argument(
        "--overwrite", action="store_true", help="with --download: also replace local files that differ"
    )
    args = ap.parse_args()
    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception:  # noqa: BLE001
        sys.exit("not logged in: run `uv run hf auth login` with a token that can write to the organisation")
    info = api.repo_info(args.repo, repo_type="dataset")
    if not info.private:
        sys.exit(f"{args.repo} is public; refusing to sync lab recordings to it")

    remote = set(api.list_repo_files(args.repo, repo_type="dataset"))
    files = local_files()
    if args.download:
        synced = [r for r in remote if r.split("/", 1)[0] in FOLDERS]
        local = {p.as_posix() for p in files}
        wanted = synced if args.overwrite else [r for r in synced if r not in local]
        if not wanted:
            print("nothing to download: every repo file is already here")
            return
        snapshot_download(args.repo, repo_type="dataset", local_dir=RESULTS, allow_patterns=wanted)
        print(
            f"downloaded {len(wanted)} files into {RESULTS}"
            + ("" if args.overwrite else " (local files kept)")
        )
        return

    missing = [p for p in files if p.as_posix() not in remote]
    size = sum((RESULTS / p).stat().st_size for p in missing)
    runs = sorted({"/".join(p.parts[:2]) for p in missing})
    print(
        f"{args.repo} (private, as {user}): {len(files)} local files, {len(missing)} not in the repo ({size / 1e6:.0f} MB)"
    )
    for r in runs:
        print(f"  {r}")
    if args.dry_run:
        print(
            "(files changed locally under an existing path are not listed; the upload hashes and sends them)"
        )
        return
    api.upload_large_folder(
        repo_id=args.repo,
        repo_type="dataset",
        folder_path=RESULTS,
        allow_patterns=[f"{f}/**" for f in FOLDERS],
        ignore_patterns=IGNORE,
        private=True,
    )
    print(f"uploaded; browse at https://huggingface.co/datasets/{args.repo}/tree/main")


if __name__ == "__main__":
    main()
