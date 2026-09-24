#!/usr/bin/env python3
"""Upload the YCBMultiTrack release to the Hugging Face Hub.

Dry-run by default; nothing reaches the Hub without --execute.

    python scripts/hf_release/upload_release.py                  # plan only
    python scripts/hf_release/upload_release.py --execute        # create + upload (private)
    python scripts/hf_release/upload_release.py --make-public    # flip to public afterwards

Requires `hf auth login` first. The repo is created private; run --make-public only
after verifying the round trip.
"""

import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HUB_DIR = os.path.join(HERE, "hub")
REPO_ID = "tzuyuanlin/YCBMultiTrack"
SMALL_FILES = ["README.md", "sequences.json", "download.py"]
ARCHIVES = [
    "YCBMultiTrack-Models.zip",  # small, upload first
    "YCBMultiTrack-Sim.zip",
    "YCBMultiTrack-Real.zip",
]


def human(n):
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", default=REPO_ID)
    p.add_argument("--zip-dir", default="/home/justin/data")
    p.add_argument("--execute", action="store_true", help="actually upload")
    p.add_argument("--make-public", action="store_true",
                   help="flip an existing repo to public")
    p.add_argument("--no-xet", action="store_true",
                   help="disable Xet and use plain LFS multipart upload; Xet keeps a "
                        "local chunk cache under HF_XET_CACHE that can fill a nearly "
                        "full disk mid-upload")
    args = p.parse_args()

    if args.no_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"

    small = [(os.path.join(HUB_DIR, f), f) for f in SMALL_FILES]
    archives = [(os.path.join(args.zip_dir, f), f) for f in ARCHIVES]

    missing = [q for q, _ in small + archives if not os.path.exists(q)]
    if missing:
        sys.exit("missing files:\n  " + "\n  ".join(missing))

    print(f"repo:  {args.repo_id}  (dataset, private)")
    total = 0
    for path, name in small + archives:
        size = os.path.getsize(path)
        total += size
        print(f"  {name:<28} {human(size):>10}   <- {path}")
    print(f"  {'total':<28} {human(total):>10}")

    free = shutil.disk_usage(args.zip_dir).free
    print(f"\nfree disk: {human(free)}")
    if free < 15e9 and not args.no_xet:
        cache = os.environ.get("HF_XET_CACHE",
                               os.path.expanduser("~/.cache/huggingface/xet"))
        print(f"  WARNING: Xet buffers chunks under {cache} while uploading and can")
        print("           exhaust a nearly full disk. Free space, or pass --no-xet.")

    if args.make_public:
        from huggingface_hub import HfApi

        HfApi().update_repo_settings(args.repo_id, repo_type="dataset", private=False)
        print(f"\n{args.repo_id} is now PUBLIC")
        return

    if not args.execute:
        print("\ndry run — nothing uploaded. re-run with --execute")
        return

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    print(f"\nauthenticated as {api.whoami()['name']}")
    api.create_repo(args.repo_id, repo_type="dataset", private=True, exist_ok=True)
    print(f"created/found {args.repo_id}")

    # Card + index + helper in one commit, so the repo is never half-described.
    api.create_commit(
        repo_id=args.repo_id, repo_type="dataset",
        operations=[CommitOperationAdd(path_in_repo=n, path_or_fileobj=p)
                    for p, n in small],
        commit_message="Add dataset card, sequence index and download helper",
    )
    print("uploaded card + index + helper")

    # One commit per archive: a 9.3 GB upload should not be able to invalidate the
    # others if it fails, and Xet dedup makes a retry resume cheaply.
    for path, name in archives:
        print(f"uploading {name} ({human(os.path.getsize(path))})...")
        api.upload_file(path_or_fileobj=path, path_in_repo=name,
                        repo_id=args.repo_id, repo_type="dataset",
                        commit_message=f"Add {name}")
        print(f"  done: {name}")

    print(f"\nhttps://huggingface.co/datasets/{args.repo_id}  (private)")
    print("verify, then re-run with --make-public")


if __name__ == "__main__":
    main()
