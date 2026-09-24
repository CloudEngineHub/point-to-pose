#!/usr/bin/env python3
"""Download and extract the YCBMultiTrack dataset from the Hugging Face Hub.

    pip install huggingface_hub

    python download.py --split all    --out ./YCBMultiTrack   # meshes + both splits (10.6 GB)
    python download.py --split sim    --out ./YCBMultiTrack   # synthetic only (1.1 GB)
    python download.py --split models --out ./YCBMultiTrack   # YCB meshes only (85 MB)
    python download.py --split real   --out ./YCBMultiTrack --keep-archive

Downloads are verified against the sha256 recorded in sequences.json. Re-running is
safe: an already-extracted split is skipped unless --force is given.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile

REPO_ID = "tzuyuanlin/YCBMultiTrack"
ARCHIVES = {
    "real": "YCBMultiTrack-Real.zip",
    "sim": "YCBMultiTrack-Sim.zip",
    "models": "YCBMultiTrack-Models.zip",
}
DIR_NAMES = {
    "real": "YCBMultiTrack-Real",
    "sim": "YCBMultiTrack-Sim",
    "models": "YCBMultiTrack-Models",
}


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    size = os.path.getsize(path)
    done = 0
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
            done += len(block)
            print(f"\r  verifying... {100 * done / size:5.1f}%", end="", flush=True)
    print("\r  verifying... done   ")
    return h.hexdigest()


def download_split(split, out_dir, keep_archive, force, skip_verify):
    from huggingface_hub import hf_hub_download

    target = os.path.join(out_dir, DIR_NAMES[split])
    if os.path.isdir(target) and not force:
        print(f"[{split}] {target} already exists, skipping (use --force to redo)")
        return

    index_path = hf_hub_download(REPO_ID, "sequences.json", repo_type="dataset")
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    expected = index["models"] if split == "models" else index["splits"][split]["archive"]

    print(f"[{split}] downloading {expected['file_name']} "
          f"({expected['size_bytes'] / 1e9:.2f} GB)")
    # local_dir keeps the archive out of ~/.cache/huggingface, so it is a real file we
    # can delete after extraction rather than a symlink into the cache blob store,
    # which would otherwise keep a second full copy on disk forever.
    os.makedirs(out_dir, exist_ok=True)
    archive = hf_hub_download(REPO_ID, ARCHIVES[split], repo_type="dataset",
                             local_dir=out_dir)

    if not skip_verify:
        digest = sha256(archive)
        if digest != expected["sha256"]:
            sys.exit(f"[{split}] CHECKSUM MISMATCH\n"
                     f"  expected {expected['sha256']}\n  got      {digest}\n"
                     f"  delete {archive} and retry")

    print(f"[{split}] extracting to {out_dir}")
    if force and os.path.isdir(target):
        shutil.rmtree(target)
    with zipfile.ZipFile(archive) as zf:
        members = zf.infolist()
        for i, m in enumerate(members):
            zf.extract(m, out_dir)
            if i % 2000 == 0 or i == len(members) - 1:
                print(f"\r  {i + 1}/{len(members)} entries", end="", flush=True)
    print()

    if not keep_archive:
        os.remove(archive)
        print(f"[{split}] removed {os.path.basename(archive)} "
              f"(use --keep-archive to retain)")

    if split == "models":
        print(f"[models] ready: {len(index['models']['objects'])} meshes in {target}")
    else:
        n = index["splits"][split]["num_sequences"]
        print(f"[{split}] ready: {n} sequences in {target}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=["real", "sim", "models", "all"], default="all")
    p.add_argument("--out", default="./YCBMultiTrack", help="extraction directory")
    p.add_argument("--keep-archive", action="store_true",
                   help="keep the downloaded .zip next to the extracted tree")
    p.add_argument("--force", action="store_true",
                   help="re-extract even if the split directory exists")
    p.add_argument("--skip-verify", action="store_true",
                   help="skip the sha256 check (not recommended)")
    args = p.parse_args()

    splits = ["models", "sim", "real"] if args.split == "all" else [args.split]
    for split in splits:
        download_split(split, args.out, args.keep_archive, args.force, args.skip_verify)


if __name__ == "__main__":
    main()
