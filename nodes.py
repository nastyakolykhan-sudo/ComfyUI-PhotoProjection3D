"""ComfyUI node: Project Photo onto 3D Model."""
from __future__ import annotations

import io

import numpy as np
import torch

from .projection_core import project_photo


def _file3d(data: bytes):
    from comfy_api.latest import Types

    return Types.File3D(io.BytesIO(data), "glb")


class PhotoProjection3D:
    """Puts the real photo onto a textured 3D model from the photo's own camera, so print, logos and graphics
    come out as photographed. The model keeps its own texture wherever the photo can't see."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_3d": ("FILE_3D,FILE_3D_GLB", {"tooltip": "A textured GLB (UV-mapped base colour texture), e.g. from an image-to-3D node."}),
                "image": ("IMAGE", {"tooltip": "The photo of the same object. Any angle: the camera is found automatically."}),
                "texture_size": ("INT", {"default": 4096, "min": 1024, "max": 8192, "step": 512,
                                         "tooltip": "Size of the new base colour texture. Higher keeps small print sharper; the file gets bigger."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "The object in the photo (white = object), e.g. from Remove Background. Strongly recommended: the camera is fitted to it and nothing outside it is projected."}),
                "label_free_image": ("IMAGE", {"tooltip": "The same photo with its text and labels removed (the copy the model was built from). Optional: makes the camera fit more exact and turns on the edge fix, which stops doubled or fringed edges where a part hangs in front of another (tags, charms, laces). Print is never moved."}),
                "facing_start": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 0.95, "step": 0.01, "advanced": True,
                                           "tooltip": "Surfaces turned further from the camera than this (cosine) keep the model's texture."}),
                "facing_end": ("FLOAT", {"default": 0.55, "min": 0.05, "max": 1.0, "step": 0.01, "advanced": True,
                                         "tooltip": "Surfaces facing the camera more than this (cosine) take the photo fully; between start and end they blend."}),
                "match_colours": ("BOOLEAN", {"default": True, "advanced": True,
                                              "tooltip": "Pull the model's own texture towards the photo's colours so the two meet without a seam."}),
                "flatten_normals": ("BOOLEAN", {"default": True, "advanced": True,
                                                "tooltip": "Flatten the model's normal map where the photo goes on, so invented relief doesn't show under real print."}),
                "edge_zone": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 0.1, "step": 0.005, "advanced": True,
                                        "tooltip": "Edge fix reach, as a fraction of the object size around parts that stand in front of others (needs label_free_image). 0 turns it off."}),
            },
        }

    RETURN_TYPES = ("FILE_3D_GLB", "IMAGE")
    RETURN_NAMES = ("model_3d", "camera_fit")
    FUNCTION = "run"
    CATEGORY = "3d"

    def run(self, model_3d, image, texture_size, mask=None, label_free_image=None, facing_start=0.20, facing_end=0.55,
            match_colours=True, flatten_normals=True, edge_zone=0.03):
        fmt = (getattr(model_3d, "format", "") or "").lower()
        if fmt and fmt not in ("glb",):
            raise ValueError(f"Project Photo onto 3D Model: needs a GLB, got {fmt!r}")
        data = model_3d.get_bytes() if hasattr(model_3d, "get_bytes") else model_3d.get_data().getvalue()
        photo = (image[0, ..., :3].clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)
        m = mask[0].clamp(0, 1).cpu().numpy() if mask is not None else None
        if facing_end <= facing_start:
            facing_end = min(1.0, facing_start + 0.05)
        lf = (label_free_image[0, ..., :3].clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8) if label_free_image is not None else None
        glb, preview, info = project_photo(data, photo, m, texture_size=texture_size, facing=(facing_start, facing_end),
                                           match_colours=match_colours, flatten_normals=flatten_normals,
                                           label_free_rgb=lf, edge_zone=edge_zone)
        print(f"[Project Photo onto 3D Model] {info}")
        return (_file3d(glb), torch.from_numpy(preview.astype(np.float32) / 255.0)[None])


NODE_CLASS_MAPPINGS = {"PhotoProjection3D": PhotoProjection3D}
NODE_DISPLAY_NAME_MAPPINGS = {"PhotoProjection3D": "Project Photo onto 3D Model"}
