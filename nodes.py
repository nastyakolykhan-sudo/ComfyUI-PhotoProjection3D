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
                "extra_views": ("PHOTO_VIEWS", {"tooltip": "More photos of the same object (back, sides) from Collect Extra Views. Each is fitted and projected onto the surfaces it sees best."}),
            },
        }

    RETURN_TYPES = ("FILE_3D_GLB", "IMAGE")
    RETURN_NAMES = ("model_3d", "camera_fit")
    FUNCTION = "run"
    CATEGORY = "3d"

    def run(self, model_3d, image, texture_size, mask=None, label_free_image=None, facing_start=0.20, facing_end=0.55,
            match_colours=True, flatten_normals=True, edge_zone=0.03, extra_views=None):
        fmt = (getattr(model_3d, "format", "") or "").lower()
        if fmt and fmt not in ("glb",):
            raise ValueError(f"Project Photo onto 3D Model: needs a GLB, got {fmt!r}")
        data = model_3d.get_bytes() if hasattr(model_3d, "get_bytes") else model_3d.get_data().getvalue()
        photo = (image[0, ..., :3].clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)
        m = mask[0].clamp(0, 1).cpu().numpy() if mask is not None else None
        if facing_end <= facing_start:
            facing_end = min(1.0, facing_start + 0.05)
        lf = _u8(label_free_image)
        glb, preview, info = project_photo(data, photo, m, texture_size=texture_size, facing=(facing_start, facing_end),
                                           match_colours=match_colours, flatten_normals=flatten_normals,
                                           label_free_rgb=lf, edge_zone=edge_zone, extra_views=extra_views or None)
        print(f"[Project Photo onto 3D Model] {info}")
        return (_file3d(glb), torch.from_numpy(preview.astype(np.float32) / 255.0)[None])


def _u8(img):
    return None if img is None else (img[0, ..., :3].clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)


# where each extra view is expected, in degrees around the object from the main photo's camera
_VIEWS = (("back", 180.0), ("left", -90.0), ("right", 90.0))
# upload nodes whose empty value means "not given"
_UPLOAD_NODES = {"WorkflowIOImageInput": "value", "LoadImage": "image"}


class ExtraViews3D:
    """Optional extra photos of the object (back, left, right). Only the views actually given run: an empty upload
    is skipped before its upstream nodes (label removal, cut-out) execute, so it costs nothing and never fails.
    Gives the label-free copies to the 3D generator and all of it to Project Photo onto 3D Model."""

    @classmethod
    def INPUT_TYPES(cls):
        opt = {}
        for v, _ in _VIEWS:
            opt[f"{v}_photo"] = ("IMAGE", {"lazy": True, "tooltip": f"The {v} of the object, as photographed (labels included)."})
            opt[f"{v}_label_free"] = ("IMAGE", {"lazy": True, "tooltip": f"The same {v} photo with its text and labels removed."})
            opt[f"{v}_mask"] = ("MASK", {"lazy": True, "tooltip": f"The object in the {v} photo (white = object)."})
        return {"required": {}, "optional": opt, "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "PHOTO_VIEWS")
    RETURN_NAMES = ("back_label_free", "left_label_free", "right_label_free", "extra_views")
    FUNCTION = "run"
    CATEGORY = "3d"

    @staticmethod
    def _given(prompt, unique_id, name):
        """Is this input connected to something that will deliver an image? An upload node with an empty value
        isn't; any other source is."""
        node = (prompt or {}).get(str(unique_id), {})
        link = node.get("inputs", {}).get(name)
        if not isinstance(link, (list, tuple)) or not link:
            return False
        src = (prompt or {}).get(str(link[0]), {})
        key = _UPLOAD_NODES.get(src.get("class_type"))
        if key is None:
            return True
        return bool(str(src.get("inputs", {}).get(key, "") or "").strip())

    def check_lazy_status(self, prompt=None, unique_id=None, **kw):
        need = []
        for v, _ in _VIEWS:
            if not self._given(prompt, unique_id, f"{v}_photo"):
                continue
            for part in ("photo", "label_free", "mask"):
                n = f"{v}_{part}"
                if kw.get(n) is None and self._given(prompt, unique_id, n):
                    need.append(n)
        return need

    def run(self, prompt=None, unique_id=None, **kw):
        views, outs = [], []
        for v, offset in _VIEWS:
            photo = kw.get(f"{v}_photo")
            lf = kw.get(f"{v}_label_free")
            if photo is None:
                outs.append(None)
                continue
            m = kw.get(f"{v}_mask")
            views.append(dict(name=v, photo=_u8(photo), label_free=_u8(lf), yaw_offset=offset,
                              mask=m[0].clamp(0, 1).cpu().numpy() if m is not None else None))
            outs.append(lf if lf is not None else photo)
        print(f"[Collect Extra Views] given: {[v['name'] for v in views] or 'none'}")
        return (outs[0], outs[1], outs[2], views)


NODE_CLASS_MAPPINGS = {"PhotoProjection3D": PhotoProjection3D, "ExtraViews3D": ExtraViews3D}
NODE_DISPLAY_NAME_MAPPINGS = {"PhotoProjection3D": "Project Photo onto 3D Model", "ExtraViews3D": "Collect Extra Views"}
