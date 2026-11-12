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

## Inputs

| Input | |
|---|---|
| `model_3d` | Textured GLB. |
| `image` | The photo of the same object. |
| `texture_size` | New base colour texture size (default 4096). |
| `mask` (optional, recommended) | The object in the photo, white = object (e.g. from Remove Background). |
| `facing_start`, `facing_end` (advanced) | Viewing-angle feather (cosines, default 0.20 to 0.55). |
| `match_colours`, `flatten_normals` (advanced) | On by default. |

Outputs: `model_3d` (GLB, base colour as JPEG q92) and `camera_fit` (the fitted outline drawn over the photo).

## Tips

- Build the model from a copy of the photo with the labels removed: generators otherwise sculpt their own
  version of the lettering into the surface, and the real print then sits on top of it.
- Thin, loose parts (laces, chains) that the generator places slightly differently can show a doubled edge.
- The photo can only be as sharp as it is: use a high-resolution photo for small print.

## Requirements

`trimesh`, `moderngl`, `scipy` (numpy and Pillow come with ComfyUI). Needs OpenGL 3.3: a desktop GPU, or EGL on
headless Linux (used automatically). All dependencies are MIT/BSD licensed.

## Licence

MIT.
