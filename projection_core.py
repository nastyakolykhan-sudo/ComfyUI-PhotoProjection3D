"""Put a real photo onto a textured 3D model from the photo's own camera.

The model (any textured GLB, e.g. from an image-to-3D service) keeps its own texture everywhere the photo can't
see. Where the photo does see the surface, the photo's pixels replace the texture, so printed text, logos and
graphics come out exactly as photographed instead of re-drawn by a generator.

Steps
1. Camera fit: the model, textured with its own colours, is rendered from many orbit cameras (yaw, pitch, field
   of view); each render is placed on the subject mask (scale + offset) and scored by silhouette overlap plus
   luminance correlation. The best few are refined with Nelder-Mead over yaw, pitch, roll, field of view and
   the 2D placement.
2. Projection: every texel is rasterised in UV space on the GPU. A texel takes the photo's colour when it faces
   that camera (feathered by the angle), is visible from it (distance-map test) and falls inside the subject
   mask; elsewhere the model's texture stays.
3. Colour match: the model's texture is pulled towards the photo's colours, per colour, using the texels both
   see (so a red-only overlap can't tint greys). The normal map is flattened where the photo goes on, so no
   invented relief shows under real print.

Stack: numpy, scipy, Pillow, trimesh (MIT), moderngl (MIT; EGL on headless Linux).
"""
from __future__ import annotations

import io
import logging
import math

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage, optimize

log = logging.getLogger("PhotoProjection3D")

FIT_VS = """#version 330
uniform mat4 mvp; in vec3 in_pos; in vec2 in_uv; out vec2 v_uv;
void main(){ v_uv = in_uv; gl_Position = mvp * vec4(in_pos, 1.0); }"""
FIT_FS = """#version 330
uniform sampler2D albedo; in vec2 v_uv; out vec4 f;
void main(){ f = vec4(texture(albedo, v_uv).rgb, 1.0); }"""
DIST_VS = """#version 330
uniform mat4 mvp; in vec3 in_pos; out vec3 v_pos;
void main(){ v_pos = in_pos; gl_Position = mvp * vec4(in_pos, 1.0); }"""
DIST_FS = """#version 330
uniform vec3 eye; in vec3 v_pos; out float f;
void main(){ f = length(v_pos - eye); }"""
UV_VS = """#version 330
in vec3 in_pos; in vec3 in_nrm; in vec2 in_uv; out vec3 v_pos; out vec3 v_nrm;
void main(){ v_pos = in_pos; v_nrm = in_nrm; gl_Position = vec4(in_uv * 2.0 - 1.0, 0.0, 1.0); }"""
UV_FS = """#version 330
uniform mat4 mvp; uniform vec3 eye; uniform float R; uniform float s; uniform vec2 t; uniform vec2 psize;
uniform sampler2D photo; uniform sampler2D mask; uniform sampler2D distmap; uniform float eps; uniform vec2 face;
in vec3 v_pos; in vec3 v_nrm; layout(location=0) out vec4 o0; layout(location=1) out vec4 o1;
void main(){
  vec4 clip = mvp * vec4(v_pos, 1.0); vec3 ndc = clip.xyz / clip.w;
  vec2 rp = vec2((ndc.x * 0.5 + 0.5) * R, (0.5 - ndc.y * 0.5) * R);
  vec2 puv = (s * rp + t) / psize;
  float inside = step(0.0, puv.x) * step(puv.x, 1.0) * step(0.0, puv.y) * step(puv.y, 1.0) * step(0.0, clip.w);
  float d = length(v_pos - eye);
  float stored = texture(distmap, ndc.xy * 0.5 + 0.5).r;
  float vis = 1.0 - smoothstep(eps, 2.0 * eps, d - stored);
  float facing = dot(normalize(v_nrm), normalize(eye - v_pos));
  float w = smoothstep(face.x, face.y, facing) * vis * texture(mask, puv).r * inside;
  o0 = vec4(texture(photo, puv).rgb, w); o1 = vec4(1.0);
}"""


# ----------------------------------------------------------------------------- GL
def gl_context():
    import moderngl

    try:
        return moderngl.create_standalone_context(require=330)
    except Exception:
        # headless Linux (containers): no X display, EGL is the offscreen path
        return moderngl.create_standalone_context(require=330, backend="egl")


# ----------------------------------------------------------------------------- maths
def perspective(fovy, near, far):
    f = 1.0 / math.tan(math.radians(fovy) / 2)
    return np.array([[f, 0, 0, 0], [0, f, 0, 0], [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)], [0, 0, -1, 0]], np.float32)


def look_at(eye, target, up=(0, 1, 0)):
    eye, target, up = (np.asarray(x, np.float64) for x in (eye, target, up))
    z = eye - target; z /= np.linalg.norm(z); x = np.cross(up, z); x /= np.linalg.norm(x); y = np.cross(z, x)
    m = np.eye(4); m[0, :3], m[1, :3], m[2, :3] = x, y, z; m[:3, 3] = -m[:3, :3] @ eye
    return m


def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(np.clip(c, 0, None), 1 / 2.4) - 0.055)


def to_lab(rgb):
    c = srgb_to_linear(rgb)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ M.T / np.array([0.9505, 1.0, 1.089])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


# ----------------------------------------------------------------------------- model
class Part:
    """One textured mesh of the scene, in world space (node transform applied)."""

    def __init__(self, geom, transform):
        w = geom.copy(); w.apply_transform(transform)
        self.geom = geom
        self.pos = np.asarray(w.vertices, np.float32)
        self.nrm = np.asarray(w.vertex_normals, np.float32)
        self.faces = np.asarray(geom.faces, np.int32)
        vis = geom.visual
        uv = getattr(vis, "uv", None)
        self.mat = getattr(vis, "material", None)
        self.tex_attr = None
        tex = None
        for attr in ("baseColorTexture", "image"):  # PBR material, or trimesh's SimpleMaterial
            if self.mat is not None and getattr(self.mat, attr, None) is not None:
                tex, self.tex_attr = getattr(self.mat, attr), attr
                break
        self.textured = uv is not None and tex is not None
        self.uv = np.asarray(uv, np.float32) if uv is not None else np.zeros((len(self.pos), 2), np.float32)
        if tex is not None:
            self.albedo = tex.convert("RGB")
        else:
            col = getattr(vis, "main_color", None)
            col = tuple(int(x) for x in (np.asarray(col)[:3] if col is not None else (200, 200, 200)))
            self.albedo = Image.new("RGB", (4, 4), col)


def load_model(glb_bytes):
    import trimesh

    scene = trimesh.load(io.BytesIO(glb_bytes), file_type="glb", force="scene")
    parts = []
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph[node]
        g = scene.geometry[gname]
        if isinstance(g, trimesh.Trimesh) and len(g.faces):
            parts.append(Part(g, T))
    if not parts:
        raise ValueError("Photo Projection: the model has no mesh")
    if not any(p.textured for p in parts):
        raise ValueError("Photo Projection: the model has no UV-mapped base colour texture to project onto")
    return scene, parts


def camera(c, r, yaw, pitch, roll, fov):
    d = r * 1.05 / math.sin(math.radians(fov) / 2)
    th, ph = math.radians(yaw), math.radians(pitch)
    eye = c + d * np.array([math.cos(ph) * math.sin(th), math.sin(ph), math.cos(ph) * math.cos(th)])
    V = look_at(eye, c)
    rz = math.radians(roll); Rz = np.eye(4)
    Rz[0, 0], Rz[0, 1], Rz[1, 0], Rz[1, 1] = math.cos(rz), -math.sin(rz), math.sin(rz), math.cos(rz)
    near = d - 1.2 * r if d - 1.2 * r > 1e-3 else d * 0.05
    P = perspective(fov, near, d + 1.2 * r)
    return (P @ Rz @ V).astype(np.float32), eye.astype(np.float32)


# ----------------------------------------------------------------------------- colour
def colour_lut(base, photo, sel, n=17, sigma=10.0, prior=8.0, samples=6000):
    lut = np.ones((n, n, n, 3), np.float32)
    idx = np.flatnonzero(sel.ravel())
    if len(idx) < 500:
        return lut
    rng = np.random.default_rng(0); idx = rng.choice(idx, min(samples, len(idx)), replace=False)
    cb = base.reshape(-1, 3)[idx].astype(np.float64); cp = photo.reshape(-1, 3)[idx].astype(np.float64)
    r = np.clip((srgb_to_linear(cp) + 0.01) / (srgb_to_linear(cb) + 0.01), 0.5, 2.0)
    lab_s = to_lab(cb)
    g = np.linspace(0, 1, n); grid = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    lab_g = to_lab(grid)
    out = np.empty((len(grid), 3))
    for i in range(0, len(grid), 512):
        d2 = ((lab_g[i:i + 512, None, :] - lab_s[None]) ** 2).sum(-1)
        k = np.exp(-d2 / (2 * sigma ** 2))
        out[i:i + 512] = (k @ r + prior) / (k.sum(1, keepdims=True) + prior)
    return np.clip(out, 0.75, 1.33).reshape(n, n, n, 3).astype(np.float32)


def apply_lut(lut, img):
    n = lut.shape[0]; coords = (np.clip(img, 0, 1) * (n - 1)).reshape(-1, 3).T
    res = np.stack([ndimage.map_coordinates(lut[..., c], coords, order=1) for c in range(3)], -1)
    return res.reshape(img.shape).astype(np.float32)


# ----------------------------------------------------------------------------- main
def project_photo(glb_bytes, photo_rgb, mask=None, texture_size=4096, facing=(0.20, 0.55), match_colours=True,
                  flatten_normals=True, jpeg_quality=92, work=320):
    """photo_rgb: HxWx3 uint8. mask: HxW float 0-1 (subject), optional.
    Returns (glb_bytes, preview HxWx3 uint8, info dict)."""
    import moderngl

    scene, parts = load_model(glb_bytes)
    photo = Image.fromarray(photo_rgb); PW, PH = photo.size
    if mask is None:
        mask_img = Image.new("L", photo.size, 255)
    else:
        mask_img = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8)).resize(photo.size, Image.BILINEAR)
    allpos = np.vstack([p.pos for p in parts]); c = (allpos.min(0) + allpos.max(0)) / 2
    r = float(np.linalg.norm(allpos - c, axis=1).max())

    k = work / max(PW, PH); WW, WH = round(PW * k), round(PH * k)
    pm = np.asarray(mask_img.resize((WW, WH), Image.BILINEAR), np.float32) / 255
    pl = ndimage.gaussian_filter(np.asarray(photo.resize((WW, WH), Image.BILINEAR).convert("L"), np.float32) / 255, 1.0)

    ctx = gl_context()
    try:
        fit_prog = ctx.program(vertex_shader=FIT_VS, fragment_shader=FIT_FS)
        gl_parts = []
        for p in parts:
            tb = np.asarray(p.albedo)[::-1].copy()
            tex = ctx.texture(p.albedo.size, 3, tb.tobytes()); tex.build_mipmaps()
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            ibo = ctx.buffer(p.faces.tobytes())
            vao = ctx.vertex_array(fit_prog, [(ctx.buffer(np.hstack([p.pos, p.uv]).astype("f4").tobytes()), "3f 2f", "in_pos", "in_uv")], ibo)
            gl_parts.append((tex, ibo, vao))
        R = work
        fbo = ctx.framebuffer(ctx.renderbuffer((R, R), 4), ctx.depth_renderbuffer((R, R)))

        def render(yaw, pitch, roll, fov):
            mvp, _ = camera(c, r, yaw, pitch, roll, fov)
            fit_prog["mvp"].write(mvp.T.tobytes()); fit_prog["albedo"].value = 0
            fbo.use(); ctx.enable(moderngl.DEPTH_TEST); ctx.clear(0, 0, 0, 0)
            for tex, _, vao in gl_parts:
                tex.use(0); vao.render()
            return np.frombuffer(fbo.read(components=4), np.uint8).reshape(R, R, 4)[::-1].astype(np.float32) / 255

        def warp(arr, s, tx, ty):
            A = (1 / s, 0, -tx / s, 0, 1 / s, -ty / s)
            return np.asarray(Image.fromarray((arr * 255).astype(np.uint8)).transform((WW, WH), Image.AFFINE, A, Image.BILINEAR), np.float32) / 255

        def score(img, s, tx, ty):
            rm = warp(img[..., 3], s, tx, ty)
            iou = np.minimum(rm, pm).sum() / max(np.maximum(rm, pm).sum(), 1e-6)
            rl = ndimage.gaussian_filter(warp(img[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32), s, tx, ty), 1.0)
            sel = (rm > 0.5) & (pm > 0.5)
            if sel.sum() < 50:
                return iou
            x, y = rl[sel] - rl[sel].mean(), pl[sel] - pl[sel].mean()
            return iou + 0.5 * float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-6))

        yy, xx = np.mgrid[0:R, 0:R]; Y, X = np.mgrid[0:WH, 0:WW]; ap_ = pm.sum()
        cp = np.array([(X * pm).sum() / max(ap_, 1e-6), (Y * pm).sum() / max(ap_, 1e-6)])

        def init_st(img):
            al = img[..., 3]; ar = al.sum()
            if ar < 10:
                return None
            s = math.sqrt(ap_ / ar); cr = np.array([(xx * al).sum() / ar, (yy * al).sum() / ar]); t = cp - s * cr
            return s, t[0], t[1]

        cands = []
        for fov in (12.0, 25.0, 40.0):
            for pitch in range(-10, 41, 10):
                for yaw in range(-180, 180, 15):
                    img = render(yaw, pitch, 0, fov); st = init_st(img)
                    if st:
                        cands.append((score(img, *st), yaw, pitch, fov, st))
        cands.sort(key=lambda z: -z[0])

        def f(p):
            yaw, pitch, roll, lfov, ls, tx, ty = p
            fov = float(np.clip(math.exp(lfov), 6, 60))
            return -score(render(yaw, pitch, roll, fov), math.exp(ls), tx, ty)

        best = None
        for _, yaw, pitch, fov, (s, tx, ty) in cands[:6]:
            x0 = np.array([yaw, pitch, 0.0, math.log(fov), math.log(s), tx, ty])
            simplex = np.array([x0] + [x0 + dd for dd in np.diag([8, 6, 4, 0.3, 0.05, 6, 6])])
            res = optimize.minimize(f, x0, method="Nelder-Mead", options={"initial_simplex": simplex, "maxiter": 400, "xatol": 0.05, "fatol": 1e-4})
            if best is None or res.fun < best.fun:
                best = res
        yaw, pitch, roll, lfov, ls, tx, ty = best.x
        fov = float(np.clip(math.exp(lfov), 6, 60)); s = math.exp(ls)
        info = {"yaw": round(yaw, 2), "pitch": round(pitch, 2), "roll": round(roll, 2), "fov": round(fov, 2), "fit_score": round(-best.fun, 3)}
        log.info("Photo Projection: camera fit %s", info)

        # preview: fitted outline over the photo
        img = render(yaw, pitch, roll, fov)
        rm = Image.fromarray((warp(img[..., 3], s, tx, ty) * 255).astype(np.uint8)).resize((PW, PH), Image.BILINEAR)
        edge = np.asarray(rm.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(3))) > 40
        preview = np.asarray(photo).copy(); preview[edge] = (0, 200, 255)

        # projection at full photo resolution
        S_full, T_full = s / k, np.array([tx, ty]) / k
        mvp, eye = camera(c, r, yaw, pitch, roll, fov)
        DR = 2048
        dist_prog = ctx.program(vertex_shader=DIST_VS, fragment_shader=DIST_FS)
        dtex = ctx.texture((DR, DR), 1, dtype="f4"); dfbo = ctx.framebuffer([dtex], ctx.depth_renderbuffer((DR, DR)))
        dist_prog["mvp"].write(mvp.T.tobytes()); dist_prog["eye"].value = tuple(float(v) for v in eye)
        dfbo.use(); ctx.enable(moderngl.DEPTH_TEST); dfbo.clear(1e9, 1e9, 1e9, 1e9)
        for p, (_, ibo, _) in zip(parts, gl_parts):
            ctx.vertex_array(dist_prog, [(ctx.buffer(p.pos.astype("f4").tobytes()), "3f", "in_pos")], ibo).render()
        dtex.filter = (moderngl.NEAREST, moderngl.NEAREST)

        ptex = ctx.texture(photo.size, 3, np.asarray(photo).tobytes()); ptex.build_mipmaps()
        ptex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR); ptex.anisotropy = 8
        mk = mask_img.filter(ImageFilter.MinFilter(7)).filter(ImageFilter.GaussianBlur(3))
        mtex = ctx.texture(photo.size, 1, np.asarray(mk).tobytes()); mtex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        uv_prog = ctx.program(vertex_shader=UV_VS, fragment_shader=UV_FS)
        for kname, val in dict(eye=tuple(float(v) for v in eye), R=float(R), s=float(S_full), t=tuple(float(v) for v in T_full),
                               psize=(float(PW), float(PH)), photo=0, mask=1, distmap=2, eps=0.012 * r,
                               face=(float(facing[0]), float(facing[1]))).items():
            uv_prog[kname].value = val
        uv_prog["mvp"].write(mvp.T.tobytes())
        ptex.use(0); mtex.use(1); dtex.use(2)

        N = int(texture_size); shares = []
        lin = srgb_to_linear
        for p, (_, ibo, _) in zip(parts, gl_parts):
            if not p.textured:
                continue
            vao = ctx.vertex_array(uv_prog, [(ctx.buffer(np.hstack([p.pos, p.nrm, p.uv]).astype("f4").tobytes()), "3f 3f 2f", "in_pos", "in_nrm", "in_uv")], ibo)
            o0 = ctx.texture((N, N), 4, dtype="f4"); o1 = ctx.texture((N, N), 4, dtype="f1")
            ufbo = ctx.framebuffer([o0, o1])
            ufbo.use(); ctx.disable(moderngl.DEPTH_TEST); ctx.disable(moderngl.CULL_FACE); ufbo.clear(0, 0, 0, 0); vao.render()
            proj = np.frombuffer(o0.read(), np.float32).reshape(N, N, 4)[::-1]
            cov = np.frombuffer(o1.read(), np.uint8).reshape(N, N, 4)[::-1][..., 0] > 0
            o0.release(); o1.release(); ufbo.release()
            w = np.clip(proj[..., 3], 0, 1); prgb = proj[..., :3]
            base = np.asarray(p.albedo.resize((N, N), Image.LANCZOS), np.float32) / 255
            base_lin = lin(base)
            if match_colours:
                base_lin = base_lin * apply_lut(colour_lut(base, prgb, (w > 0.7) & cov), base)
            out = linear_to_srgb(base_lin * (1 - w[..., None]) + lin(prgb) * w[..., None])
            dist, (iy, ix) = ndimage.distance_transform_edt(~cov, return_indices=True)
            pad = (~cov) & (dist <= 8)
            out[pad] = out[iy[pad], ix[pad]]; wpad = w.copy(); wpad[pad] = w[iy[pad], ix[pad]]
            shares.append(float((wpad > 0.5).mean()))
            new_tex = Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8))
            setattr(p.mat, p.tex_attr, _jpeg_for_gltf(new_tex, jpeg_quality))
            if flatten_normals and getattr(p.mat, "normalTexture", None) is not None:
                nimg = p.mat.normalTexture.convert("RGB"); nn = np.asarray(nimg, np.float32)
                wn = np.asarray(Image.fromarray((wpad * 255).astype(np.uint8)).resize(nimg.size, Image.BILINEAR), np.float32)[..., None] / 255
                p.mat.normalTexture = Image.fromarray((nn * (1 - wn) + np.array([128, 128, 255], np.float32) * wn + 0.5).astype(np.uint8))
        info["photo_share"] = round(float(np.mean(shares)), 3) if shares else 0.0
    finally:
        ctx.release()
    out_bytes = scene.export(file_type="glb")
    return out_bytes, preview, info


def _jpeg_for_gltf(img, quality):
    """A PIL image that trimesh's glTF exporter writes as JPEG at `quality` (it re-saves JPEGs at the default 75)."""
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=int(quality), subsampling=0); buf.seek(0)
    jpg = Image.open(buf); jpg.load()
    orig_save = jpg.save

    def save(fp, format=None, **kw):  # noqa: A002 - PIL's signature
        if (format or "").upper() == "JPEG":
            kw.setdefault("quality", int(quality)); kw.setdefault("subsampling", 0)
        return orig_save(fp, format=format, **kw)

    jpg.save = save
    return jpg
