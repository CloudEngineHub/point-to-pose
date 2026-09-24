#!/usr/bin/env python3
"""Build YCBMultiTrack-Models.zip — the object meshes the poses are defined against,
so users do not have to source them separately.

Ships only what evaluation and rendering need: the decimated `textured_simple.obj`,
the material it references, its texture map, and the sampled point cloud. The
full-resolution `textured.obj` (~51 MB/object) is deliberately excluded.

Files are copied verbatim from the HO3D-v3 model set.

    python scripts/hf_release/pack_models.py
"""

import argparse
import os
import re
import sys
import zipfile

OBJECTS = [
    "002_master_chef_can", "004_sugar_box", "005_tomato_soup_can",
    "006_mustard_bottle", "008_pudding_box", "010_potted_meat_can",
    "019_pitcher_base", "021_bleach_cleanser", "037_scissors",
    "072a_toy_airplane",
]

ATTRIBUTION = """# Object meshes — attribution

The meshes in this archive are from the **YCB Object and Model Set**, redistributed
here under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) so that
YCBMultiTrack is usable without sourcing them separately.

> Calli, B., Singh, A., Bruce, J., Walsman, A., Konolige, K., Srinivasa, S.,
> Abbeel, P., Dollar, A. M. "Yale-CMU-Berkeley dataset for robotic manipulation
> research." *The International Journal of Robotics Research*, 2017.
>
> Calli, B., Walsman, A., Singh, A., Srinivasa, S., Abbeel, P., Dollar, A. M.
> "Benchmarking in Manipulation Research: The YCB Object and Model Set and
> Benchmarking Protocols." *IEEE Robotics and Automation Magazine*, 2015.

Project page: <https://www.ycbbenchmarks.com/>

**Immediate source.** These files are taken verbatim from the model set shipped with
[HO3D-v3](https://www.tugraz.at/index.php?id=40231), i.e. `HO3D_V3/models/<object>/`.
That set is HO3D's packaging of the YCB meshes, and `textured_simple.obj` is the
decimated variant also distributed with YCB-Video. If you already have HO3D-v3, these
files are byte-identical to your copy and this archive is redundant.

**Changes made:** no mesh, material or texture file has been modified — all files are
byte-for-byte copies. The only change is *selection*: this archive contains the ten
objects used by YCBMultiTrack, and for each only `textured_simple.obj`, the material
it references, `texture_map.png`, and `points.xyz` where available. The
full-resolution `textured.obj` and the other format conversions shipped with the
original set are omitted for size.

YCBMultiTrack ground-truth poses are defined against **`textured_simple.obj`**, in
metres — a different mesh origin or scale will not align.
"""


def referenced_material(obj_path):
    """Return the .mtl an .obj declares. The objects do not agree on this: most point
    at textured_simple.obj.mtl, 072a_toy_airplane at textured.mtl."""
    with open(obj_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("mtllib"):
                return os.path.basename(line.split(maxsplit=1)[1].strip())
            if line.startswith(("v ", "f ")):
                break
    return None


def referenced_texture(mtl_path):
    with open(mtl_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.match(r"\s*map_Kd\s+(.+)", line)
            if m:
                return os.path.basename(m.group(1).strip())
    return None


def collect(models_root, obj_name):
    d = os.path.join(models_root, obj_name)
    obj = os.path.join(d, "textured_simple.obj")
    if not os.path.exists(obj):
        sys.exit(f"{obj_name}: missing textured_simple.obj")
    files = [obj]

    mtl_name = referenced_material(obj)
    if mtl_name is None:
        sys.exit(f"{obj_name}: textured_simple.obj declares no mtllib")
    mtl = os.path.join(d, mtl_name)
    if not os.path.exists(mtl):
        sys.exit(f"{obj_name}: obj references {mtl_name}, which is missing")
    files.append(mtl)

    tex_name = referenced_texture(mtl)
    if tex_name is None:
        sys.exit(f"{obj_name}: {mtl_name} declares no map_Kd")
    tex = os.path.join(d, tex_name)
    if not os.path.exists(tex):
        sys.exit(f"{obj_name}: material references {tex_name}, which is missing")
    files.append(tex)

    points = os.path.join(d, "points.xyz")
    if os.path.exists(points):
        files.append(points)
    return [(p, os.path.basename(p)) for p in files]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models-root", default="/home/justin/data/HO3D_V3/models")
    p.add_argument("--out", default="/home/justin/data/YCBMultiTrack-Models.zip")
    args = p.parse_args()

    plan = {obj: collect(args.models_root, obj) for obj in OBJECTS}
    total = sum(os.path.getsize(src) for fs in plan.values() for src, _ in fs)
    print(f"packing {len(OBJECTS)} objects, {total / 1e6:.1f} MB raw")

    root = "YCBMultiTrack-Models"
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr(f"{root}/ATTRIBUTION.md", ATTRIBUTION)
        for obj, files in plan.items():
            for src, name in files:
                zf.write(src, f"{root}/{obj}/{name}")
            print(f"  {obj:<24} {', '.join(n for _, n in files)}")

    print(f"wrote {args.out}  ({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
