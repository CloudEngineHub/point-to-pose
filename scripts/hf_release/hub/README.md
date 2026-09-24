---
license: cc-by-4.0
pretty_name: YCBMultiTrack
size_categories:
- 10K<n<100K
task_categories:
- robotics
tags:
- 6d-pose-estimation
- pose-tracking
- rgbd
- object-tracking
- occlusion
- ycb
- motion-capture
viewer: false
---

# YCBMultiTrack

A dynamic **multi-object RGB-D benchmark for 6D pose tracking under heavy occlusion**,
with motion-capture ground truth. Released with
[Point2Pose](https://point2pose.github.io/) (ECCV 2026).

Objects are picked up, moved, rotated, mutually occluded, and repeatedly carried
**fully out of frame** by a human hand — the regime that separates methods that
re-localize from methods that merely track. Every frame carries per-object 6D pose,
an instance mask, and two visibility labels that make the evaluation protocol
unambiguous.

| Split | Sequences | Frames | Objects | Camera | Archive |
|---|---|---|---|---|---|
| **Real** | 11 | 16,314 | 5 YCB | RealSense D435 + OptiTrack | 9.35 GB |
| **Sim** | 5 | 2,705 | 7 YCB | Isaac Sim, 4 environments | 1.13 GB |
| **Models** | — | — | 10 YCB meshes | — | 85 MB |

## Download

```bash
pip install huggingface_hub
wget https://huggingface.co/datasets/tzuyuanlin/YCBMultiTrack/raw/main/download.py

python download.py --split all    --out ./YCBMultiTrack   # meshes + both splits, 10.6 GB
python download.py --split sim    --out ./YCBMultiTrack   # synthetic only, 1.1 GB
python download.py --split models --out ./YCBMultiTrack   # YCB meshes only, 85 MB
```

The helper downloads, verifies the sha256 recorded in `sequences.json`, and extracts.
Or do it by hand:

```python
from huggingface_hub import hf_hub_download
p = hf_hub_download("tzuyuanlin/YCBMultiTrack", "YCBMultiTrack-Sim.zip", repo_type="dataset")
```

```
sha256  0cd5b0bc8e194c3dd5673fe585c1cd28ef76e3ed1637e4a0ed0327fcfa3036cb  YCBMultiTrack-Real.zip
sha256  59f06f2ce57cafd892ff4f29f63932be272d42f8a4de248188fd3bba4d0c2317  YCBMultiTrack-Sim.zip
sha256  ef77921d3f5b5616dfb9fc58d0f31ae07ec924c5b50316b2bbb400779d4064f6  YCBMultiTrack-Models.zip
```

## Layout

Each split archive extracts to one directory per sequence:

```
YCBMultiTrack-Real/<sequence>/
├── rgb/000000.png ...                  uint8 RGB, 640x480
├── depth/000000.png ...                uint16 PNG, millimetres
├── normalized_depth/000000.png ...     8-bit depth, for visualization only
├── cam_K.txt                           3x3 intrinsics
├── masks/<object>/000000.png ...       binary instance mask
├── annotated_poses/<object>/000000.txt 4x4 T_mesh2cam, row-major
├── is_mask_visible/<object>/
│   ├── is_mask_visible.npy             uint8 [N], per-frame
│   └── video_with_mask_labels.mp4      label overlay, for inspection
└── is_obj_in_image_labels/<object>/
    └── is_obj_in_image.npy             uint8 [N], per-frame
```

Frame ids are consistent across `rgb/`, `depth/`, `normalized_depth/`, `masks/` and
`annotated_poses/`. Sequences are named after the objects they contain, so the object
set is readable from the directory name.

A few directories hold small non-frame files alongside the per-frame data — glob
`annotated_poses/<object>/*.txt` rather than `*` so they are not picked up as poses.

### Conventions

- **Depth** is uint16 millimetres; divide by 1000 for metres. Eleven Real frames contain
  the RealSense saturation sentinel `65535` on a few hundred background pixels; none fall
  inside an object mask. Treat depth above ~10 m as invalid if you back-project the full
  frame.
- **Poses are held constant while an object is occluded or out of frame**, and each Real
  sequence ends with 53–89 frames in which the objects are at rest and the pose is
  therefore identical frame to frame. Both are expected; use `is_mask_visible` and
  `is_obj_in_image` rather than pose motion to decide what to score.
- **Poses** are `T_mesh2cam`: a 4×4 row-major matrix mapping the object mesh frame
  into the camera frame, in metres. They are defined against the **`textured_simple.obj`**
  meshes shipped in `YCBMultiTrack-Models.zip` — a different mesh origin will not align.
- **`is_obj_in_image`** is 0 when the object is entirely outside the image.
- **`is_mask_visible`** is 0 when no pixel of the object is visible (fully occluded or
  out of frame). Both are 1 only when the object is genuinely observable.
- **Intrinsics** are constant within each split:

  | Split | fx | fy | cx | cy |
  |---|---|---|---|---|
  | Real | 614.429 | 615.125 | 323.473 | 239.058 |
  | Sim | 380.0 | 380.0 | 320.0 | 240.0 |

`sequences.json` carries the same facts machine-readably: per-sequence frame counts,
objects, intrinsics, and per-object counts of out-of-frame and fully-occluded frames.

### Object meshes

**The meshes are included** — `YCBMultiTrack-Models.zip` (85 MB), so nothing has to be
sourced separately:

```
YCBMultiTrack-Models/
├── ATTRIBUTION.md
└── <object>/
    ├── textured_simple.obj        the mesh poses are defined against, in metres
    ├── textured_simple.obj.mtl    (072a_toy_airplane: textured.mtl)
    ├── texture_map.png            4096x4096
    └── points.xyz                 sampled point cloud, where available
```

Ten objects: `002_master_chef_can`, `004_sugar_box`, `005_tomato_soup_can`,
`006_mustard_bottle`, `008_pudding_box`, `010_potted_meat_can`, `019_pitcher_base`,
`021_bleach_cleanser`, `037_scissors`, `072a_toy_airplane`.

These are byte-for-byte copies of the model set shipped with
[HO3D-v3](https://www.tugraz.at/index.php?id=40231) (`HO3D_V3/models/<object>/`), which
is HO3D's packaging of the [YCB Object and Model Set](https://www.ycbbenchmarks.com/);
`textured_simple.obj` is the decimated variant also distributed with YCB-Video. **If you
already have HO3D-v3, this archive is redundant** — point your model root at
`HO3D_V3/models` instead. The full-resolution `textured.obj` (~51 MB/object) is omitted
for size.

Redistributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — see
`ATTRIBUTION.md` in the archive for citations.

## Sequences

### Real — RealSense D435, OptiTrack ground truth

Single-object sequences exercise out-of-frame disappearance; multi-object sequences add
mutual occlusion, with `easy`/`hard` denoting the amount of occlusion and motion.

Percentages are the share of frames in which an object is **occluded** — hidden behind
another object or a hand, or carried out of the image — i.e. where
`is_mask_visible` or `is_obj_in_image` is 0. Those frames are excluded from scoring.

| Sequence | Frames | Size | Objects (% frames occluded) |
|---|---|---|---|
| `005_tomato_soup_can` | 1,420 | 818 MB | tomato_soup_can (14%) |
| `006_mustard_bottle` | 1,816 | 1,032 MB | mustard_bottle (29%) |
| `008_pudding_box` | 1,316 | 743 MB | pudding_box (26%) |
| `010_potted_meat_can` | 1,419 | 810 MB | potted_meat_can (19%) |
| `021_bleach_cleanser` | 1,561 | 900 MB | bleach_cleanser (10%) |
| `005_tomato_soup_can_008_pudding_box_easy` | 1,181 | 698 MB | tomato_soup_can (11%), pudding_box (7%) |
| `005_tomato_soup_can_008_pudding_box_hard` | 1,587 | 938 MB | tomato_soup_can (11%), pudding_box (15%) |
| `006_mustard_bottle_010_potted_meat_can_easy` | 1,342 | 816 MB | mustard_bottle (2%), potted_meat_can (1%) |
| `006_mustard_bottle_010_potted_meat_can_hard` | 1,769 | 1,063 MB | mustard_bottle (15%), potted_meat_can (14%) |
| `006_mustard_bottle_010_potted_meat_can_005_tomato_soup_can` | 1,366 | 889 MB | tomato_soup_can (8%), mustard_bottle (9%), potted_meat_can (13%) |
| `021_bleach_cleanser_005_tomato_soup_can_008_pudding_box` | 1,537 | 980 MB | tomato_soup_can (4%), pudding_box (3%), bleach_cleanser (<1%) |

### Sim — Isaac Sim

Objects are never occluded or out of frame in the synthetic split; it isolates pose
accuracy without the occlusion-recovery challenge.

| Sequence | Frames | Size | Objects |
|---|---|---|---|
| `env0_019_pitcher_base` | 501 | 207 MB | pitcher_base |
| `env0_037_scissors_002_master_chef_can` | 501 | 212 MB | scissors, master_chef_can |
| `env1_004_sugar_box` | 501 | 205 MB | sugar_box |
| `env2_072a_toy_airplane_005_tomato_soup_can` | 701 | 314 MB | toy_airplane, tomato_soup_can |
| `env3_010_potted_meat_can_004_sugar_box` | 501 | 213 MB | potted_meat_can, sugar_box |

## Ground truth

**Real.** Object poses come from an OptiTrack motion-capture system, transferred into
the camera frame through a per-sequence camera-to-world extrinsic and a per-object
mesh-to-rigid-body transform. The camera was repositioned between takes, so the
extrinsic is genuinely per-sequence.

**Sim.** Poses are exact by construction.

## Evaluation protocol

The protocol used in the Point2Pose paper:

- Score a frame only where the object is observable **and** ground truth exists **and**
  the method produced a prediction — i.e. `is_obj_in_image & is_mask_visible`.
- Align predictions to ground truth at the first valid frame (methods are tracked from
  an arbitrary initial frame, not from a canonical pose).
- Report **ADD** and **ADD-S**, and their AUC with a 0.1 m threshold, against the
  `textured_simple.obj` meshes.

Reference implementation: [`experiments/ycbinisaac/`](https://github.com/tzuyuan/point-to-pose/tree/main/experiments/ycbinisaac)
in the Point2Pose repository. `YCBInIsaacReader` in
[`point2pose/io/sources/dataset/datareader.py`](https://github.com/tzuyuan/point-to-pose/blob/main/point2pose/io/sources/dataset/datareader.py)
reads this layout directly:

```python
from point2pose.io.sources.dataset.datareader import YCBInIsaacReader

reader = YCBInIsaacReader("YCBMultiTrack/YCBMultiTrack-Real/005_tomato_soup_can")
rgb = reader.get_color(0)              # HxWx3 uint8
depth = reader.get_depth(0)            # HxW float, metres
poses = reader.get_gt_poses(0)         # {object_name: 4x4 T_mesh2cam}
```

## License

Released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The bundled
YCB object meshes carry their own attribution from the
[YCB Object and Model Set](https://www.ycbbenchmarks.com/), also CC BY 4.0 — see
`ATTRIBUTION.md` inside `YCBMultiTrack-Models.zip`.

## Citation

```bibtex
@inproceedings{lin2026point2pose,
  title     = {Occlusion-Recovering 6D Pose Tracking and 3D Reconstruction for
               Multiple Unknown Objects via 2D Point Trackers},
  author    = {Lin, Tzu-Yuan and Lee, Ho Jae and Doherty, Kevin and
               Lee, Yonghyeon and Kim, Sangbae},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```
