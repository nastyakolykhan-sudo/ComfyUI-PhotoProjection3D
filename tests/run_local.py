"""Run the projection outside ComfyUI: python tests/run_local.py MODEL.glb PHOTO MASK OUT.glb [--texture 4096]
MASK: a greyscale image, white = object. Writes OUT.glb and OUT_fit.png (fitted outline over the photo)."""
import argparse, json, os, sys, time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from projection_core import project_photo  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("glb"); ap.add_argument("photo"); ap.add_argument("mask"); ap.add_argument("out")
ap.add_argument("--texture", type=int, default=4096)
a = ap.parse_args()
photo = np.asarray(Image.open(a.photo).convert("RGB"))
mask = np.asarray(Image.open(a.mask).convert("L").resize(photo.shape[1::-1]), np.float32) / 255
t = time.time()
glb, preview, info = project_photo(open(a.glb, "rb").read(), photo, mask, texture_size=a.texture)
open(a.out, "wb").write(glb)
Image.fromarray(preview).save(os.path.splitext(a.out)[0] + "_fit.png")
info.update(seconds=round(time.time() - t, 1), mb=round(len(glb) / 1e6, 1))
print(json.dumps(info))
