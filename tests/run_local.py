"""Run the projection outside ComfyUI: python tests/run_local.py MODEL.glb PHOTO MASK OUT.glb [--texture 4096] [--label-free CLEAN.png]
  [--view NAME:PHOTO:MASK:CLEAN:OFFSET ...]
MASK: a greyscale image, white = object (for --view, "-" for none; CLEAN "-" for none; OFFSET degrees from the main
photo's camera, e.g. 180 back, -90 left, 90 right, "-" to search all round). Writes OUT.glb and OUT_fit.png."""
import argparse, json, os, sys, time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from projection_core import project_photo  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("glb"); ap.add_argument("photo"); ap.add_argument("mask"); ap.add_argument("out")
ap.add_argument("--texture", type=int, default=4096); ap.add_argument("--label-free", default="")
ap.add_argument("--edge-zone", type=float, default=0.03); ap.add_argument("--fine", type=int, default=896)
ap.add_argument("--view", action="append", default=[])
a = ap.parse_args()
photo = np.asarray(Image.open(a.photo).convert("RGB"))
mask = np.asarray(Image.open(a.mask).convert("L").resize(photo.shape[1::-1]), np.float32) / 255
t = time.time()
lf = np.asarray(Image.open(a.label_free).convert("RGB")) if a.label_free else None
extra = []
for spec in a.view:
    name, vp, vm, vc, off = spec.split(":")
    ph = np.asarray(Image.open(vp).convert("RGB"))
    extra.append(dict(name=name, photo=ph,
                      mask=None if vm == "-" else np.asarray(Image.open(vm).convert("L").resize(ph.shape[1::-1]), np.float32) / 255,
                      label_free=None if vc == "-" else np.asarray(Image.open(vc).convert("RGB")),
                      yaw_offset=None if off == "-" else float(off)))
glb, preview, info = project_photo(open(a.glb, "rb").read(), photo, mask, texture_size=a.texture, label_free_rgb=lf, edge_zone=a.edge_zone,
                                   fine_work=a.fine, extra_views=extra)
open(a.out, "wb").write(glb)
Image.fromarray(preview).save(os.path.splitext(a.out)[0] + "_fit.png")
info.update(seconds=round(time.time() - t, 1), mb=round(len(glb) / 1e6, 1))
print(json.dumps(info))
