#!/usr/bin/env python3
"""Build the release zip for a YCBMultiTrack split.

Matches the layout of the original archives: every file under a single top-level
`YCBMultiTrack-<Split>/` directory, deflate-compressed.

    python scripts/hf_release/pack_split.py --split Sim
    python scripts/hf_release/pack_split.py --split Real
"""

import argparse
import hashlib
import os
import sys
import zipfile


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["Real", "Sim"], required=True)
    p.add_argument("--data-root", default="/home/justin/data")
    p.add_argument("--keep-old", action="store_true",
                   help="keep any existing archive as <name>.prev.zip")
    args = p.parse_args()

    name = f"YCBMultiTrack-{args.split}"
    src = os.path.join(args.data_root, name)
    out = os.path.join(args.data_root, f"{name}.zip")
    if not os.path.isdir(src):
        sys.exit(f"no such directory: {src}")

    files = []
    for dirpath, _, filenames in os.walk(src):
        for fn in sorted(filenames):
            full = os.path.join(dirpath, fn)
            files.append((full, os.path.join(name, os.path.relpath(full, src))))
    files.sort(key=lambda t: t[1])
    total = sum(os.path.getsize(f) for f, _ in files)
    print(f"{name}: {len(files)} files, {total / 1e9:.2f} GB")

    if os.path.exists(out):
        if args.keep_old:
            prev = out.replace(".zip", ".prev.zip")
            os.replace(out, prev)
            print(f"  previous archive kept at {prev}")
        else:
            os.remove(out)
            print("  removed previous archive")

    tmp = out + ".partial"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6,
                         allowZip64=True) as zf:
        for i, (full, arc) in enumerate(files):
            zf.write(full, arc)
            if i % 5000 == 0:
                print(f"\r  {i}/{len(files)}", end="", flush=True)
    os.replace(tmp, out)
    print(f"\r  {len(files)}/{len(files)}")
    print(f"wrote {out}  ({os.path.getsize(out) / 1e9:.2f} GB)")
    print(f"sha256 {sha256(out)}")


if __name__ == "__main__":
    main()
