#!/usr/bin/env python3
"""Build sequences.json, the machine-readable index shipped with the YCBMultiTrack
release on the Hugging Face Hub.

Scans both split directories, records per-sequence frame counts, objects, intrinsics
and visibility statistics, and checksums the three release archives.

    python scripts/hf_release/build_index.py
"""

import argparse
import glob
import hashlib
import json
import os
import zipfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def scan_sequence(seq_dir):
    rgb = sorted(glob.glob(os.path.join(seq_dir, "rgb", "*.png")))
    width, height = Image.open(rgb[0]).size
    K = np.loadtxt(os.path.join(seq_dir, "cam_K.txt")).reshape(3, 3)

    objects = []
    for obj in sorted(os.listdir(os.path.join(seq_dir, "masks"))):
        entry = {"name": obj}
        in_image = os.path.join(seq_dir, "is_obj_in_image_labels", obj,
                                "is_obj_in_image.npy")
        visible = os.path.join(seq_dir, "is_mask_visible", obj, "is_mask_visible.npy")
        if os.path.exists(in_image):
            entry["frames_out_of_frame"] = int((np.load(in_image) == 0).sum())
        if os.path.exists(visible):
            entry["frames_mask_not_visible"] = int((np.load(visible) == 0).sum())
        objects.append(entry)

    out = {
        "num_frames": len(rgb),
        "resolution": [width, height],
        "first_frame_id": os.path.splitext(os.path.basename(rgb[0]))[0],
        "cam_K": K.tolist(),
        "objects": objects,
    }
    prov = os.path.join(seq_dir, "MASK_PROVENANCE.json")
    if os.path.exists(prov):
        with open(prov, encoding="utf-8") as f:
            out["mask_provenance"] = json.load(f)
    return out


def describe_zip(path):
    with zipfile.ZipFile(path) as zf:
        num_files = sum(1 for i in zf.infolist() if not i.is_dir())
    return {
        "file_name": os.path.basename(path),
        "size_bytes": os.path.getsize(path),
        "num_files": num_files,
        "sha256": sha256(path),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real", default="/home/justin/data/YCBMultiTrack-Real")
    p.add_argument("--sim", default="/home/justin/data/YCBMultiTrack-Sim")
    p.add_argument("--zip-dir", default="/home/justin/data")
    p.add_argument("--out", default=os.path.join(HERE, "hub", "sequences.json"))
    args = p.parse_args()

    index = {"name": "YCBMultiTrack", "splits": {}}

    models_zip = os.path.join(args.zip_dir, "YCBMultiTrack-Models.zip")
    if os.path.exists(models_zip):
        with zipfile.ZipFile(models_zip) as zf:
            objects = sorted({n.split("/")[1] for n in zf.namelist()
                              if n.count("/") > 1 and not n.endswith("/")})
        index["models"] = describe_zip(models_zip)
        index["models"]["objects"] = objects
        index["models"]["source"] = "YCB Object and Model Set (CC BY 4.0), via HO3D-v3"

    for split, root in [("real", args.real), ("sim", args.sim)]:
        sequences = {}
        for seq in sorted(os.listdir(root)):
            if os.path.isdir(os.path.join(root, seq)):
                sequences[seq] = scan_sequence(os.path.join(root, seq))
        zip_path = os.path.join(
            args.zip_dir, f"YCBMultiTrack-{'Real' if split == 'real' else 'Sim'}.zip")
        index["splits"][split] = {
            "archive": describe_zip(zip_path),
            "num_sequences": len(sequences),
            "num_frames": sum(s["num_frames"] for s in sequences.values()),
            "sequences": sequences,
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    if "models" in index:
        m = index["models"]
        print(f"models: {len(m['objects'])} objects, "
              f"{m['size_bytes'] / 1e6:.1f} MB, sha256 {m['sha256'][:16]}...")
    for split, info in index["splits"].items():
        a = info["archive"]
        print(f"{split}: {info['num_sequences']} sequences, {info['num_frames']} frames, "
              f"{a['size_bytes'] / 1e9:.2f} GB, sha256 {a['sha256'][:16]}...")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
