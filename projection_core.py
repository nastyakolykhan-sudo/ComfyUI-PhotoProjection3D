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
4. More photos (optional): other views of the same object (back, sides) are fitted the same way, each searched
   around its expected direction from the main photo, and projected too. Where several photos see a surface, the
   one facing it most squarely wins, with a narrow hand-over; the main photo is favoured where it sees well.

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


# ----------------------------------------------------------------------------- label-free photo + zone fix
def register_label_free(label_free, photo):
    """The label-free copy can come back resized or slightly reframed: map it onto the photo (SIFT + RANSAC
    similarity). Returns the registered copy (PIL) or None when it can't be lined up."""
    import cv2

    P = np.asarray(photo); C = np.asarray(label_free)
    k1 = 1600 / max(P.shape[:2]); k2 = 1600 / max(C.shape[:2])
    g1 = cv2.cvtColor(cv2.resize(P, None, fx=k1, fy=k1, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(cv2.resize(C, None, fx=k2, fy=k2, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(4000); kp1, d1 = sift.detectAndCompute(g1, None); kp2, d2 = sift.detectAndCompute(g2, None)
    if d1 is None or d2 is None:
        return None
    good = [m for m, n in (x for x in cv2.BFMatcher().knnMatch(d2, d1, k=2) if len(x) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 30:
        return None
    src = np.float32([kp2[m.queryIdx].pt for m in good]) / k2; dst = np.float32([kp1[m.trainIdx].pt for m in good]) / k1
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    if M is None or inl is None or int(inl.sum()) < 30:
        return None
    return Image.fromarray(cv2.warpAffine(C, M, photo.size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))


def zone_fix(albedo_rgba, dist, valid, r, M_render_to_crop, crop, photo, mask, label_free, zone, mismatch, info):
    """Fix only where the generated shape and the real object part ways: within `zone` (fraction of the object size)
    of the model's depth jumps (a tag or a charm in front of a bag, laces over a tongue), never on removed print
    (fine detail present in the photo but gone in the label-free copy), never on the outer outline. Inside: the photo
    is shifted onto the model's own render (optical flow against the label-free copy; skipped when that copy doesn't
    line up), then where the model and the aligned label-free copy still disagree, the model keeps its own texture."""
    import cv2

    PW, PH = photo.size; x0, y0, x1, y1 = crop; cw, chh = x1 - x0, y1 - y0
    M, (tw, th), kt = M_render_to_crop
    thr = 0.03 * r
    jx = valid[:, 1:] & valid[:, :-1] & (np.abs(dist[:, 1:] - dist[:, :-1]) > thr)
    jy = valid[1:, :] & valid[:-1, :] & (np.abs(dist[1:, :] - dist[:-1, :]) > thr)
    e = np.zeros_like(valid); e[:, 1:] |= jx; e[:, :-1] |= jx; e[1:, :] |= jy; e[:-1, :] |= jy
    vy, vx = np.nonzero(valid); diag = math.hypot(np.ptp(vx), np.ptp(vy)); rad = max(2, int(zone * diag))
    zmap = cv2.dilate(e.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))).astype(np.float32)
    zmap = np.clip(cv2.GaussianBlur(zmap, (0, 0), rad / 3), 0, 1)
    ren_t = cv2.warpAffine(albedo_rgba, M, (tw, th), flags=cv2.INTER_LINEAR)
    zone_t = cv2.warpAffine(zmap, M, (tw, th), flags=cv2.INTER_LINEAR)
    sil_t = ren_t[..., 3] > 127
    P = np.asarray(photo); Mk = np.asarray(mask); C = np.asarray(label_free)
    pho_t = cv2.resize(np.ascontiguousarray(P[y0:y1, x0:x1]), (tw, th), interpolation=cv2.INTER_AREA)
    cln_t = cv2.resize(np.ascontiguousarray(C[y0:y1, x0:x1]), (tw, th), interpolation=cv2.INTER_AREA)

    def detail(img):
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        return cv2.GaussianBlur(np.abs(g - cv2.GaussianBlur(g, (0, 0), 2.0)), (0, 0), 4)
    label = np.clip((detail(pho_t) - 1.8 * detail(cln_t) - 2.0) / 4.0, 0, 1)
    label = np.clip(cv2.GaussianBlur(cv2.dilate(label, np.ones((7, 7), np.uint8)), (0, 0), 3) * 1.5, 0, 1)
    inner = cv2.GaussianBlur(cv2.erode(sil_t.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(np.float32), (0, 0), 3)
    work = zone_t * (1 - label) * inner

    def norm(g):
        g = g.astype(np.float32); hp = g - cv2.GaussianBlur(g, (0, 0), 6)
        return np.clip(128 + 60 * hp / (cv2.GaussianBlur(np.abs(hp), (0, 0), 6) + 2.0), 0, 255).astype(np.uint8)
    g0 = norm(cv2.cvtColor(ren_t[..., :3], cv2.COLOR_RGB2GRAY)); g1 = norm(cv2.cvtColor(cln_t, cv2.COLOR_RGB2GRAY)); g0[~sil_t] = 128
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM); dis.setFinestScale(1)
    flow = dis.calc(g0, g1, None)
    for ch in range(2):
        flow[..., ch] = cv2.GaussianBlur(cv2.medianBlur(np.ascontiguousarray(flow[..., ch]), 5), (0, 0), 4)
    mag = np.linalg.norm(flow, axis=-1)
    sel = sil_t & (label < 0.5)
    reliable = bool(sel.any()) and float(np.median(mag[sel])) < 0.01 * max(tw, th)
    flow *= np.minimum(1.0, (0.035 * max(tw, th)) / (mag + 1e-6))[..., None]
    flow *= work[..., None] if reliable else 0.0
    P2, Mk2 = P, Mk
    if reliable:
        fl = cv2.resize(flow, (cw, chh), interpolation=cv2.INTER_LINEAR) / kt
        X, Y = np.meshgrid(np.arange(x0, x1, dtype=np.float32), np.arange(y0, y1, dtype=np.float32))
        mx, my = X + fl[..., 0], Y + fl[..., 1]
        P2 = P.copy(); P2[y0:y1, x0:x1] = cv2.remap(P, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        Mk2 = Mk.copy(); Mk2[y0:y1, x0:x1] = cv2.remap(Mk, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    Xt, Yt = np.meshgrid(np.arange(tw, dtype=np.float32), np.arange(th, dtype=np.float32))
    cln_al = cv2.remap(cln_t, Xt + flow[..., 0], Yt + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    lab_r = cv2.cvtColor(ren_t[..., :3], cv2.COLOR_RGB2LAB).astype(np.float32); lab_c = cv2.cvtColor(cln_al, cv2.COLOR_RGB2LAB).astype(np.float32)
    for lab in (lab_r, lab_c):
        L = lab[..., 0]; mu, sd = L[sil_t].mean(), L[sil_t].std() + 1e-3; lab[..., 0] = (L - mu) / sd * 40 + 128
        lab[..., 0] = lab[..., 0] - cv2.GaussianBlur(lab[..., 0], (0, 0), 25) + 128
    d = np.abs(lab_r - lab_c); D = cv2.GaussianBlur(d[..., 0] / 45 + d[..., 1] / 18 + d[..., 2] / 18, (0, 0), 3)
    cut = np.clip((D - mismatch) / 0.4, 0, 1) * work
    cut = cv2.GaussianBlur(cv2.dilate(cut, np.ones((5, 5), np.uint8)), (0, 0), 2)
    cut_full = np.zeros((PH, PW), np.float32); cut_full[y0:y1, x0:x1] = cv2.resize(cut, (cw, chh), interpolation=cv2.INTER_LINEAR)
    Mk2 = (Mk2.astype(np.float32) * (1 - cut_full)).astype(np.uint8)
    info.update(zone_shift="on" if reliable else "off", zone_share=round(float((zone_t > 0.5)[sil_t].mean()), 3),
                zone_model_texture=round(float((cut > 0.5)[sil_t].mean()), 3))
    return Image.fromarray(np.ascontiguousarray(P2)), Image.fromarray(np.ascontiguousarray(Mk2))


# ----------------------------------------------------------------------------- main
class _FitPass:
    """Render + score at one working resolution (render frame R x R, photo scaled by k)."""

    def __init__(self, ctx, prog, gl_parts, c, r, R, photo_size, mask_img, fit_ref):
        PW, PH = photo_size
        self.ctx, self.prog, self.gl_parts, self.c, self.r = ctx, prog, gl_parts, c, r
        self.R = R; self.k = R / max(PW, PH); self.WW, self.WH = round(PW * self.k), round(PH * self.k)
        self.pm = np.asarray(mask_img.resize((self.WW, self.WH), Image.BILINEAR), np.float32) / 255
        self.pl = ndimage.gaussian_filter(np.asarray(fit_ref.resize((self.WW, self.WH), Image.BILINEAR).convert("L"), np.float32) / 255, 1.0 * R / 320)
        self.fbo = ctx.framebuffer(ctx.renderbuffer((R, R), 4), ctx.depth_renderbuffer((R, R)))

    def render(self, yaw, pitch, roll, fov):
        import moderngl

        mvp, _ = camera(self.c, self.r, yaw, pitch, roll, fov)
        self.prog["mvp"].write(mvp.T.tobytes()); self.prog["albedo"].value = 0
        self.fbo.use(); self.ctx.enable(moderngl.DEPTH_TEST); self.ctx.clear(0, 0, 0, 0)
        for tex, _, vao in self.gl_parts:
            tex.use(0); vao.render()
        R = self.R
        return np.frombuffer(self.fbo.read(components=4), np.uint8).reshape(R, R, 4)[::-1].astype(np.float32) / 255

    def warp(self, arr, s, tx, ty):
        A = (1 / s, 0, -tx / s, 0, 1 / s, -ty / s)
        return np.asarray(Image.fromarray((arr * 255).astype(np.uint8)).transform((self.WW, self.WH), Image.AFFINE, A, Image.BILINEAR), np.float32) / 255

    def score(self, img, s, tx, ty):
        rm = self.warp(img[..., 3], s, tx, ty)
        iou = np.minimum(rm, self.pm).sum() / max(np.maximum(rm, self.pm).sum(), 1e-6)
        rl = ndimage.gaussian_filter(self.warp(img[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32), s, tx, ty), 1.0 * self.R / 320)
        sel = (rm > 0.5) & (self.pm > 0.5)
        if sel.sum() < 50:
            return iou
        x, y = rl[sel] - rl[sel].mean(), self.pl[sel] - self.pl[sel].mean()
        return iou + 0.5 * float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-6))

    def f(self, p):
        yaw, pitch, roll, lfov, ls, tx, ty = p
        fov = float(np.clip(math.exp(lfov), 6, 60))
        return -self.score(self.render(yaw, pitch, roll, fov), math.exp(ls), tx, ty)


def _angle_diff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _fit_camera(ctx, prog, gl_parts, c, r, photo_size, mask_img, fit_ref, work, fine_work, yaw_window, info):
    """Coarse grid + Nelder-Mead at `work`, then a refinement at `fine_work`. yaw_window: (centre, half-width) in
    degrees to search only around an expected direction (extra views), or None for all round.
    Returns (pass, params[yaw, pitch, roll, log fov, log scale, tx, ty])."""
    A1 = _FitPass(ctx, prog, gl_parts, c, r, work, photo_size, mask_img, fit_ref)
    R = A1.R; yy, xx = np.mgrid[0:R, 0:R]; Y, X = np.mgrid[0:A1.WH, 0:A1.WW]; ap_ = A1.pm.sum()
    cp = np.array([(X * A1.pm).sum() / max(ap_, 1e-6), (Y * A1.pm).sum() / max(ap_, 1e-6)])
    yaws = [y for y in range(-180, 180, 15) if yaw_window is None or _angle_diff(y, yaw_window[0]) <= yaw_window[1]]
    cands = []
    for fov in (12.0, 25.0, 40.0):
        for pitch in range(-10, 41, 10):
            for yaw in yaws:
                img = A1.render(yaw, pitch, 0, fov); al = img[..., 3]; ar = al.sum()
                if ar < 10:
                    continue
                s0 = math.sqrt(ap_ / ar); cr = np.array([(xx * al).sum() / ar, (yy * al).sum() / ar]); t0 = cp - s0 * cr
                cands.append((A1.score(img, s0, t0[0], t0[1]), yaw, pitch, fov, (s0, t0[0], t0[1])))
    cands.sort(key=lambda z: -z[0])
    best = None
    for _, yaw, pitch, fov, (s0, tx0, ty0) in cands[:6]:
        x0 = np.array([yaw, pitch, 0.0, math.log(fov), math.log(s0), tx0, ty0])
        simplex = np.array([x0] + [x0 + dd for dd in np.diag([8, 6, 4, 0.3, 0.05, 6, 6])])
        res = optimize.minimize(A1.f, x0, method="Nelder-Mead", options={"initial_simplex": simplex, "maxiter": 400, "xatol": 0.05, "fatol": 1e-4})
        if best is None or res.fun < best.fun:
            best = res
    fit_pass, x = A1, best.x
    info["fit_score"] = round(float(-best.fun), 3)
    if fine_work and fine_work > work:
        A2 = _FitPass(ctx, prog, gl_parts, c, r, int(fine_work), photo_size, mask_img, fit_ref); q = A2.R / A1.R
        x2 = x.copy(); x2[5:7] *= q
        simplex = np.array([x2] + [x2 + dd for dd in np.diag([1.5, 1.2, 0.8, 0.05, 0.01, 2.0, 2.0])])
        res2 = optimize.minimize(A2.f, x2, method="Nelder-Mead", options={"initial_simplex": simplex, "maxiter": 120, "xatol": 0.02, "fatol": 1e-5})
        info["fine_score_start"] = round(float(-A2.f(x2)), 4); info["fine_score"] = round(float(-res2.fun), 4)
        if res2.fun <= A2.f(x2):
            fit_pass, x = A2, res2.x
    return fit_pass, x


def project_photo(glb_bytes, photo_rgb, mask=None, texture_size=4096, facing=(0.20, 0.55), match_colours=True,
                  flatten_normals=True, jpeg_quality=92, work=320, label_free_rgb=None, edge_zone=0.03, fine_work=896,
                  mismatch=1.0, extra_views=None, main_bias=1.25, sharpness=12.0, view_window=60.0):
    """photo_rgb: HxWx3 uint8. mask: HxW float 0-1 (subject), optional. label_free_rgb: the label-free copy the
    model was built from (optional): fits the camera against it and enables the edge-zone fix.
    extra_views: optional list of more photos of the same object, each a dict with "photo" (HxWx3 uint8), "mask"
    (HxW float or None), "label_free" (HxWx3 uint8 or None), "name" and "yaw_offset" (degrees around the object
    from the main photo's camera: 180 back, -90 left, +90 right; None searches all round). Every view is fitted
    and projected; where several see a surface, the one facing it most squarely wins (the main photo is favoured
    by `main_bias`).
    Returns (glb_bytes, preview HxWx3 uint8, info dict)."""
    import moderngl

    scene, parts = load_model(glb_bytes)
    allpos = np.vstack([p.pos for p in parts]); c = (allpos.min(0) + allpos.max(0)) / 2
    r = float(np.linalg.norm(allpos - c, axis=1).max())
    views = [dict(name="main", photo=photo_rgb, mask=mask, label_free=label_free_rgb, yaw_offset=None)]
    views += [v for v in (extra_views or []) if v is not None and v.get("photo") is not None]
    info = {}

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
        dist_prog = ctx.program(vertex_shader=DIST_VS, fragment_shader=DIST_FS)
        DR = 2048

        prepared, previews, main_yaw = [], [], None
        for vi, v in enumerate(views):
            vinfo = info if vi == 0 else {"name": v.get("name", f"view {vi}")}
            photo = Image.fromarray(np.ascontiguousarray(v["photo"])); PW, PH = photo.size
            if v.get("mask") is None:
                mask_img = Image.new("L", photo.size, 255)
            else:
                mask_img = Image.fromarray((np.clip(v["mask"], 0, 1) * 255).astype(np.uint8)).resize(photo.size, Image.BILINEAR)
            label_free = None
            if v.get("label_free") is not None:
                try:
                    label_free = register_label_free(Image.fromarray(np.ascontiguousarray(v["label_free"])), photo)
                    vinfo["label_free"] = "registered" if label_free is not None else "could not be lined up, not used"
                except ImportError:
                    vinfo["label_free"] = "opencv missing, not used"
            fit_ref = label_free if label_free is not None else photo   # label-free like the model's own texture
            window = None
            if vi > 0 and main_yaw is not None and v.get("yaw_offset") is not None:
                window = (main_yaw + float(v["yaw_offset"]), float(view_window))
            fit_pass, x = _fit_camera(ctx, fit_prog, gl_parts, c, r, (PW, PH), mask_img, fit_ref, work, fine_work, window, vinfo)
            yaw, pitch, roll, lfov, ls, tx, ty = x
            fov = float(np.clip(math.exp(lfov), 6, 60)); s = math.exp(ls); R = fit_pass.R; k = fit_pass.k
            vinfo.update(yaw=round(float(yaw), 2), pitch=round(float(pitch), 2), roll=round(float(roll), 2), fov=round(float(fov), 2))
            if vi == 0:
                main_yaw = float(yaw)
            log.info("Photo Projection: camera fit (%s) %s", v.get("name", "main"), vinfo)

            # preview: fitted outline over the photo
            img = fit_pass.render(yaw, pitch, roll, fov)
            rm = Image.fromarray((fit_pass.warp(img[..., 3], s, tx, ty) * 255).astype(np.uint8)).resize((PW, PH), Image.BILINEAR)
            edge = np.asarray(rm.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(3))) > 40
            pv_img = np.asarray(photo).copy(); pv_img[edge] = (0, 200, 255); previews.append(pv_img)

            # distance map from this camera (visibility)
            S_full, T_full = s / k, np.array([tx, ty]) / k
            mvp, eye = camera(c, r, yaw, pitch, roll, fov)
            dtex = ctx.texture((DR, DR), 1, dtype="f4"); dfbo = ctx.framebuffer([dtex], ctx.depth_renderbuffer((DR, DR)))
            dist_prog["mvp"].write(mvp.T.tobytes()); dist_prog["eye"].value = tuple(float(q) for q in eye)
            dfbo.use(); ctx.enable(moderngl.DEPTH_TEST); dfbo.clear(1e9, 1e9, 1e9, 1e9)
            for p, (_, ibo, _) in zip(parts, gl_parts):
                ctx.vertex_array(dist_prog, [(ctx.buffer(p.pos.astype("f4").tobytes()), "3f", "in_pos")], ibo).render()
            dtex.filter = (moderngl.NEAREST, moderngl.NEAREST)

            proj_photo, proj_mask = photo, mask_img
            if label_free is not None and edge_zone > 0:
                import cv2
                hfbo = ctx.framebuffer(ctx.renderbuffer((DR, DR), 4), ctx.depth_renderbuffer((DR, DR)))
                fit_prog["mvp"].write(mvp.T.tobytes()); fit_prog["albedo"].value = 0
                hfbo.use(); ctx.enable(moderngl.DEPTH_TEST); ctx.clear(0, 0, 0, 0)
                for tex, _, vao in gl_parts:
                    tex.use(0); vao.render()
                alb = np.ascontiguousarray(np.frombuffer(hfbo.read(components=4), np.uint8).reshape(DR, DR, 4)[::-1]); hfbo.release()
                dist = np.ascontiguousarray(np.frombuffer(dtex.read(), np.float32).reshape(DR, DR)[::-1])
                mk_arr = np.asarray(mask_img); ys, xs = np.nonzero(mk_arr > 128)
                if len(xs):
                    bx0, bx1, by0, by1 = xs.min(), xs.max(), ys.min(), ys.max(); mg = 0.06 * max(bx1 - bx0, by1 - by0)
                    crop = (int(max(0, bx0 - mg)), int(max(0, by0 - mg)), int(min(PW, bx1 + mg)), int(min(PH, by1 + mg)))
                    cw, chh = crop[2] - crop[0], crop[3] - crop[1]; kt = 1024 / max(cw, chh); tw, th = round(cw * kt), round(chh * kt)
                    A2s = kt * S_full * R / DR; b2 = kt * (T_full - np.array([crop[0], crop[1]]))
                    Mrc = np.float32([[A2s, 0, b2[0]], [0, A2s, b2[1]]])
                    alb = cv2.GaussianBlur(alb, (0, 0), max(0.5, 0.5 / A2s))
                    proj_photo, proj_mask = zone_fix(alb, dist, dist < 1e8, r, (Mrc, (tw, th), kt), crop, photo, mask_img, label_free,
                                                     edge_zone, mismatch, vinfo)

            ptex = ctx.texture(photo.size, 3, np.asarray(proj_photo).tobytes()); ptex.build_mipmaps()
            ptex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR); ptex.anisotropy = 8
            mk = proj_mask.filter(ImageFilter.MinFilter(7)).filter(ImageFilter.GaussianBlur(3))
            mtex = ctx.texture(photo.size, 1, np.asarray(mk).tobytes()); mtex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            prepared.append(dict(eye=tuple(float(q) for q in eye), R=float(R), s=float(S_full), t=tuple(float(q) for q in T_full),
                                 psize=(float(PW), float(PH)), mvp=mvp, ptex=ptex, mtex=mtex, dtex=dtex,
                                 bias=float(main_bias) if vi == 0 else 1.0, info=vinfo))
            if vi > 0:
                info.setdefault("views", []).append(vinfo)

        uv_prog = ctx.program(vertex_shader=UV_VS, fragment_shader=UV_FS)
        uv_prog["photo"].value = 0; uv_prog["mask"].value = 1; uv_prog["distmap"].value = 2
        uv_prog["eps"].value = 0.012 * r; uv_prog["face"].value = (float(facing[0]), float(facing[1]))

        N = int(texture_size); shares = []; view_shares = np.zeros(len(prepared))
        lin = srgb_to_linear
        for p, (_, ibo, _) in zip(parts, gl_parts):
            if not p.textured:
                continue
            vao = ctx.vertex_array(uv_prog, [(ctx.buffer(np.hstack([p.pos, p.nrm, p.uv]).astype("f4").tobytes()), "3f 3f 2f", "in_pos", "in_nrm", "in_uv")], ibo)
            num = den = wmax = cov = None; best = None
            for j, pv in enumerate(prepared):
                for kname in ("eye", "R", "s", "t", "psize"):
                    uv_prog[kname].value = pv[kname]
                uv_prog["mvp"].write(pv["mvp"].T.tobytes())
                pv["ptex"].use(0); pv["mtex"].use(1); pv["dtex"].use(2)
                o0 = ctx.texture((N, N), 4, dtype="f4"); o1 = ctx.texture((N, N), 4, dtype="f1")
                ufbo = ctx.framebuffer([o0, o1])
                ufbo.use(); ctx.disable(moderngl.DEPTH_TEST); ctx.disable(moderngl.CULL_FACE); ufbo.clear(0, 0, 0, 0); vao.render()
                proj = np.frombuffer(o0.read(), np.float32).reshape(N, N, 4)[::-1]
                if cov is None:
                    cov = np.frombuffer(o1.read(), np.uint8).reshape(N, N, 4)[::-1][..., 0] > 0
                o0.release(); o1.release(); ufbo.release()
                w_j = np.clip(proj[..., 3], 0, 1)
                if len(prepared) == 1:
                    num, den, wmax = proj[..., :3], None, w_j
                    break
                # the view facing a texel most squarely wins; `sharpness` keeps the hand-over narrow (no ghosting)
                wp = (w_j * pv["bias"]) ** sharpness
                if num is None:
                    num = np.zeros((N, N, 3), np.float32); den = np.zeros((N, N), np.float32)
                    wmax = np.zeros((N, N), np.float32); best = np.full((N, N), -1, np.int16); bestw = np.zeros((N, N), np.float32)
                num += wp[..., None] * proj[..., :3]; den += wp; wmax = np.maximum(wmax, w_j)
                upd = wp > bestw; best[upd] = j; bestw[upd] = wp[upd]
                del proj
            if den is None:
                w, prgb = wmax, num
            else:
                prgb = np.divide(num, den[..., None], out=np.zeros_like(num), where=den[..., None] > 0); w = wmax
                for j in range(len(prepared)):
                    view_shares[j] += float(((best == j) & (w > 0.5) & cov).mean())
            base = np.asarray(p.albedo.resize((N, N), Image.LANCZOS), np.float32) / 255
            base_lin = lin(base)
            if match_colours:
                base_lin = base_lin * apply_lut(colour_lut(base, prgb, (w > 0.7) & cov), base)
            out = linear_to_srgb(base_lin * (1 - w[..., None]) + lin(prgb) * w[..., None])
            dist_t, (iy, ix) = ndimage.distance_transform_edt(~cov, return_indices=True)
            pad = (~cov) & (dist_t <= 8)
            out[pad] = out[iy[pad], ix[pad]]; wpad = w.copy(); wpad[pad] = w[iy[pad], ix[pad]]
            shares.append(float((wpad > 0.5).mean()))
            new_tex = Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8))
            setattr(p.mat, p.tex_attr, _jpeg_for_gltf(new_tex, jpeg_quality))
            if flatten_normals and getattr(p.mat, "normalTexture", None) is not None:
                nimg = p.mat.normalTexture.convert("RGB"); nn = np.asarray(nimg, np.float32)
                wn = np.asarray(Image.fromarray((wpad * 255).astype(np.uint8)).resize(nimg.size, Image.BILINEAR), np.float32)[..., None] / 255
                p.mat.normalTexture = Image.fromarray((nn * (1 - wn) + np.array([128, 128, 255], np.float32) * wn + 0.5).astype(np.uint8))
        info["photo_share"] = round(float(np.mean(shares)), 3) if shares else 0.0
        if len(prepared) > 1:
            nt = max(1, sum(1 for p in parts if p.textured))
            for j, pv in enumerate(prepared):
                pv["info"]["share"] = round(float(view_shares[j] / nt), 3)
    finally:
        ctx.release()
    out_bytes = scene.export(file_type="glb")
    if len(previews) > 1:
        H = previews[0].shape[0]
        previews = [pv if pv.shape[0] == H else np.asarray(Image.fromarray(pv).resize((round(pv.shape[1] * H / pv.shape[0]), H), Image.BILINEAR)) for pv in previews]
        preview = np.concatenate(previews, axis=1)
    else:
        preview = previews[0]
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
