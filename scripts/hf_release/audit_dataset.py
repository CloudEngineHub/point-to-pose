#!/usr/bin/env python3
"""Full integrity pass over every YCBMultiTrack sequence. Read-only.

Complements audit_masks.py (which checks mask *content*). This checks that the
dataset is structurally sound and that every file is readable and plausible:

  structure   required subdirectories present; per-object mask/pose dirs present
  counts      rgb / depth / normalized_depth / masks / poses all the same length
  ids         frame ids align across subdirectories
  decode      every PNG decodes, with the expected size and dtype
  depth       uint16, plausible range, fraction of invalid (zero) pixels
  rgb         not blank; consecutive identical frames (a stuck capture)
  poses       parse as 4x4, rotation orthonormal with det +1, object in front of
              the camera at a plausible distance
  continuity  frame-to-frame translation and rotation jumps
  labels      visibility arrays are uint8 with one entry per frame

    python scripts/hf_release/audit_dataset.py --out dataset_audit.json
"""

import argparse
import glob
import hashlib
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

ROOTS = {
    "real": "/home/justin/data/YCBMultiTrack-Real",
    "sim": "/home/justin/data/YCBMultiTrack-Sim",
}
REQUIRED = ["rgb", "depth", "normalized_depth", "masks", "annotated_poses",
            "is_mask_visible", "is_obj_in_image_labels"]


def stem(p):
    return os.path.splitext(os.path.basename(p))[0]


def check_image(path, expect_shape, kind):
    """Decode one image and return a problem string, or None."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return f"unreadable: {path}"
    if img.shape[:2] != expect_shape:
        return f"wrong size {img.shape[:2]} != {expect_shape}: {path}"
    if kind == "depth" and img.dtype != np.uint16:
        return f"depth dtype {img.dtype} != uint16: {path}"
    if kind == "rgb" and img.ndim != 3:
        return f"rgb not 3-channel: {path}"
    return None


def audit_sequence(split, seq, deep):
    d = os.path.join(ROOTS[split], seq)
    prob = []
    info = {"split": split, "sequence": seq}

    for sub in REQUIRED:
        if not os.path.isdir(os.path.join(d, sub)):
            prob.append(f"missing directory: {sub}")
    if not os.path.exists(os.path.join(d, "cam_K.txt")):
        prob.append("missing cam_K.txt")
    if prob:
        return info, prob

    K = np.loadtxt(os.path.join(d, "cam_K.txt")).reshape(3, 3)
    if not (K[0, 0] > 0 and K[1, 1] > 0 and K[2, 2] == 1):
        prob.append(f"implausible intrinsics: {K.tolist()}")

    rgb = sorted(glob.glob(f"{d}/rgb/*.png"))
    dep = sorted(glob.glob(f"{d}/depth/*.png"))
    nrm = sorted(glob.glob(f"{d}/normalized_depth/*.png"))
    n = len(rgb)
    info["num_frames"] = n
    if not (n == len(dep) == len(nrm)):
        prob.append(f"count mismatch rgb={n} depth={len(dep)} normalized={len(nrm)}")
    if [stem(p) for p in rgb] != [stem(p) for p in dep]:
        prob.append("rgb and depth frame ids differ")
    if [stem(p) for p in rgb] != [stem(p) for p in nrm]:
        prob.append("rgb and normalized_depth frame ids differ")

    first = cv2.imread(rgb[0])
    h, w = first.shape[:2]
    info["resolution"] = [w, h]

    objects = sorted(os.listdir(f"{d}/masks"))
    info["objects"] = objects
    ids = [stem(p) for p in rgb]

    for o in objects:
        mk = sorted(glob.glob(f"{d}/masks/{o}/*.png"))
        po = sorted(glob.glob(f"{d}/annotated_poses/{o}/*.txt"))
        if len(mk) != n:
            prob.append(f"{o}: {len(mk)} masks for {n} frames")
        if len(po) != n:
            prob.append(f"{o}: {len(po)} pose files for {n} frames")
        for lbl, fn in (("is_mask_visible", "is_mask_visible.npy"),
                        ("is_obj_in_image_labels", "is_obj_in_image.npy")):
            p = f"{d}/{lbl}/{o}/{fn}"
            if not os.path.exists(p):
                prob.append(f"{o}: missing {lbl}")
                continue
            a = np.load(p)
            if len(a) != n:
                prob.append(f"{o}: {lbl} has {len(a)} entries for {n} frames")
            if a.dtype != np.uint8 or not set(np.unique(a)) <= {0, 1}:
                prob.append(f"{o}: {lbl} dtype {a.dtype} values {np.unique(a)[:5]}")

        # poses: valid SE(3), in front of camera, plausible distance
        bad_se3 = bad_z = 0
        trans, rots = [], []
        prev = None
        for i, pid in enumerate(ids):
            pp = f"{d}/annotated_poses/{o}/{pid}.txt"
            if not os.path.exists(pp):
                prob.append(f"{o}: missing pose for frame {pid}")
                break
            T = np.loadtxt(pp)
            if T.shape != (4, 4):
                prob.append(f"{o}: pose {pid} is {T.shape}, not 4x4")
                break
            R = T[:3, :3]
            if (np.abs(R @ R.T - np.eye(3)).max() > 1e-3
                    or abs(np.linalg.det(R) - 1) > 1e-3):
                bad_se3 += 1
            z = T[2, 3]
            if not 0.05 < z < 5.0:
                bad_z += 1
            if prev is not None:
                trans.append(float(np.linalg.norm(T[:3, 3] - prev[:3, 3])))
                dR = R @ prev[:3, :3].T
                rots.append(float(np.degrees(np.arccos(
                    np.clip((np.trace(dR) - 1) / 2, -1, 1)))))
            prev = T
        if bad_se3:
            prob.append(f"{o}: {bad_se3} poses are not valid SE(3)")
        if bad_z:
            prob.append(f"{o}: {bad_z} poses place the object outside 5 cm - 5 m")
        if trans:
            info.setdefault("motion", {})[o] = {
                "translation_p50_mm": round(float(np.percentile(trans, 50)) * 1e3, 2),
                "translation_p99_mm": round(float(np.percentile(trans, 99)) * 1e3, 2),
                "translation_max_mm": round(float(np.max(trans)) * 1e3, 2),
                "rotation_p99_deg": round(float(np.percentile(rots, 99)), 2),
                "rotation_max_deg": round(float(np.max(rots)), 2),
            }
            # a jump far beyond the sequence's own 99th percentile is suspicious
            thr = max(0.05, 6 * np.percentile(trans, 99))
            jumps = int((np.array(trans) > thr).sum())
            if jumps:
                prob.append(f"{o}: {jumps} pose jumps over {thr * 1e3:.0f} mm "
                            f"between consecutive frames")

    if not deep:
        return info, prob

    # decode every image
    tasks = ([(p, (h, w), "rgb") for p in rgb]
             + [(p, (h, w), "depth") for p in dep]
             + [(p, (h, w), "gray") for p in nrm])
    for o in objects:
        tasks += [(p, (h, w), "gray") for p in sorted(glob.glob(f"{d}/masks/{o}/*.png"))]
    with ThreadPoolExecutor(16) as ex:
        for r in ex.map(lambda t: check_image(*t), tasks):
            if r:
                prob.append(r)
    info["files_decoded"] = len(tasks)

    # depth sanity + blank/stuck rgb, sampled
    zero_frac, dmin, dmax = [], [], []
    hashes, blank = [], 0
    for i in range(0, n, max(1, n // 60)):
        z = cv2.imread(dep[i], cv2.IMREAD_UNCHANGED)
        zero_frac.append(float((z == 0).mean()))
        nz = z[z > 0]
        if nz.size:
            dmin.append(int(nz.min()))
            dmax.append(int(nz.max()))
        c = cv2.imread(rgb[i])
        if c.max() == 0:
            blank += 1
        hashes.append(hashlib.md5(c.tobytes()).hexdigest())
    info["depth_zero_fraction_mean"] = round(float(np.mean(zero_frac)), 4)
    if dmin:
        info["depth_mm_range"] = [int(np.min(dmin)), int(np.max(dmax))]
        if np.min(dmin) < 50 or np.max(dmax) > 20000:
            prob.append(f"depth out of plausible range: {np.min(dmin)}-{np.max(dmax)} mm")
    if blank:
        prob.append(f"{blank} sampled rgb frames are entirely black")
    dup = sum(1 for a, b in zip(hashes, hashes[1:]) if a == b)
    if dup:
        prob.append(f"{dup} consecutive sampled rgb frames are byte-identical")
    return info, prob


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/home/justin/data/ycbmultitrack_mask_rerender/dataset_audit.json")
    p.add_argument("--shallow", action="store_true",
                   help="skip decoding every image (structure and poses only)")
    args = p.parse_args()

    report, all_problems = {}, defaultdict(list)
    for split, root in ROOTS.items():
        for seq in sorted(os.listdir(root)):
            if not os.path.isdir(os.path.join(root, seq)):
                continue
            info, prob = audit_sequence(split, seq, not args.shallow)
            info["problems"] = prob
            report[f"{split}/{seq}"] = info
            status = "OK" if not prob else f"{len(prob)} PROBLEM(S)"
            print(f"[{split}] {seq:<58} {status}", flush=True)
            for q in prob:
                print(f"    - {q}", flush=True)
                all_problems[f"{split}/{seq}"].append(q)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    n_seq = len(report)
    n_bad = len(all_problems)
    print(f"\n{n_seq} sequences audited, {n_seq - n_bad} clean, {n_bad} with findings")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
