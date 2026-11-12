# ComfyUI-PhotoProjection3D

One node, **Project Photo onto 3D Model**: puts the real photo onto a textured 3D model from the photo's own
camera, so printed text, logos and graphics come out exactly as photographed instead of re-drawn by a 3D
generator. The model keeps its own texture wherever the photo can't see (sides, back, underside).

Works with any textured GLB (a UV-mapped base colour texture), e.g. from an image-to-3D node.

## How it works

1. **Camera fit.** The model, with its own colours, is rendered from many orbit cameras; each render is placed on
   the object mask and scored by silhouette overlap plus brightness correlation; the best few are refined
   (yaw, pitch, roll, field of view, placement). Any photo angle works; no front view is assumed.
2. **Projection.** Every texel facing that camera, visible from it and inside the mask takes the photo's colour,
   feathered by the viewing angle. Everything else keeps the model's texture.
3. **Colour match.** The model's texture is pulled towards the photo's colours, per colour, so photo and model
   meet without a seam. The normal map is flattened under the photo so no invented relief shows through print.
4. **Edge fix (with `label_free_image`).** Near edges where one part stands in front of another (a tag on a bag, a
   charm, laces), the generated shape and the real object rarely line up exactly. There the photo is shifted onto the
   model and, where the two still disagree, the model keeps its own texture: no doubled charms, no fringes. Print
   that the label-free copy shows was removed is never moved. The camera is also fitted against the label-free copy
   (it looks like the model's own texture), in two passes: coarse, then at full detail.

## Inputs

| Input | |
|---|---|
| `model_3d` | Textured GLB. |
| `image` | The photo of the same object. |
| `texture_size` | New base colour texture size (default 4096). |
| `mask` (optional, recommended) | The object in the photo, white = object (e.g. from Remove Background). |
| `label_free_image` (optional, recommended) | The photo with its text and labels removed, the copy the model was built from. Turns on the edge fix. |
| `facing_start`, `facing_end` (advanced) | Viewing-angle feather (cosines, default 0.20 to 0.55). |
| `match_colours`, `flatten_normals` (advanced) | On by default. |
| `edge_zone` (advanced) | Reach of the edge fix around parts in front of others, as a fraction of the object size (default 0.03; 0 = off). |

Outputs: `model_3d` (GLB, base colour as JPEG q92) and `camera_fit` (the fitted outline drawn over the photo).

## Tips

- Build the model from a copy of the photo with the labels removed: generators otherwise sculpt their own
  version of the lettering into the surface, and the real print then sits on top of it.
- Thin, loose parts (laces, chains) that the generator places slightly differently can show a doubled edge.
- The photo can only be as sharp as it is: use a high-resolution photo for small print.

## Requirements

`trimesh`, `moderngl`, `scipy` (numpy and Pillow come with ComfyUI). The label-free input also uses OpenCV (`cv2`),
which almost every ComfyUI install already has; without it the node skips the edge fix and says so. Needs OpenGL 3.3: a desktop GPU, or EGL on
headless Linux (used automatically). All dependencies are MIT/BSD/Apache licensed.

## Licence

MIT.
