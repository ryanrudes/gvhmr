"""GPU mesh renderer (moderngl / OpenGL-over-Metal) — a fast drop-in for the pytorch3d
overlay renderer on Apple Silicon, where pytorch3d has no GPU backend and falls back to a
slow CPU rasterizer.

Renders the SMPL body (and an optional checkerboard ground) with a pinhole camera built
straight from the intrinsics ``K`` the model uses, so the overlay lands exactly where
``perspective_projection`` puts the joints. Flat shading via screen-space derivatives keeps
it dependency-free (no per-vertex normals). ~1–3 ms/frame at 1080p vs. seconds on CPU.
"""

from __future__ import annotations

import numpy as np

from gvhmr.utils.vis.renderer_tools import checkerboard_geometry

_VERT_SHADER = """
#version 330
uniform mat4 u_proj;
uniform mat4 u_view;
in vec3 in_pos;
in vec3 in_col;
out vec3 v_cam;     // camera-space position (for flat normal via derivatives)
out vec3 v_col;
void main() {
    vec4 cam = u_view * vec4(in_pos, 1.0);
    v_cam = cam.xyz;
    v_col = in_col;
    gl_Position = u_proj * cam;
}
"""

_FRAG_SHADER = """
#version 330
in vec3 v_cam;
in vec3 v_col;
out vec4 f_color;
uniform float u_shade;   // 1 = lit, 0 = flat unshaded (ground)
void main() {
    if (u_shade > 0.5) {
        vec3 N = normalize(cross(dFdx(v_cam), dFdy(v_cam)));
        vec3 L = normalize(vec3(0.3, -0.5, -1.0));   // camera-space light (CV: +y down, +z fwd)
        float diff = max(abs(dot(N, L)), 0.0);       // abs: shade both facings of the open mesh
        f_color = vec4(v_col * (0.45 + 0.55 * diff), 1.0);
    } else {
        f_color = vec4(v_col, 1.0);
    }
}
"""


def make_renderer(width, height, device=None, faces=None, K=None, prefer_gpu=True):
    """Pick the overlay renderer: the fast GPU one (moderngl, ~real-time, no pytorch3d) when
    available, else pytorch3d. The GPU path works wherever a GL/Metal context exists — Apple
    Silicon, most desktops, and headless NVIDIA nodes via EGL (see ``_standalone_context``); a
    box with no usable GPU context falls back to pytorch3d (CUDA)."""
    from gvhmr.utils.pylogger import Log

    if prefer_gpu:
        try:
            return ModernGLRenderer(width, height, device=device, faces=faces, K=K)
        except Exception as e:  # no GL context / moderngl missing → pytorch3d
            Log.info(f"GPU (moderngl) renderer unavailable ({type(e).__name__}: {e}); using pytorch3d")
    from gvhmr.utils.vis.renderer import Renderer

    return Renderer(width, height, device=device, faces=faces, K=K)


def projection_from_K(K, width, height, near=0.05, far=1000.0):
    """OpenGL projection matrix from OpenCV pinhole intrinsics (camera: +x right, +y down,
    +z forward). Maps camera-space points to clip space, with the image y-flip folded in."""
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    w, h, n, f = width, height, near, far
    return np.array(
        [
            [2 * fx / w, 0, 2 * cx / w - 1, 0],
            [0, -2 * fy / h, 1 - 2 * cy / h, 0],
            [0, 0, (f + n) / (f - n), -2 * f * n / (f - n)],
            [0, 0, 1, 0],
        ],
        dtype="f4",
    )


def _np(x):
    import torch

    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _standalone_context(moderngl):
    """Create a headless GL context, falling back to EGL on displayless nodes.

    moderngl's default backend uses GLX/X11 and dies with ``XOpenDisplay: cannot open
    display`` on headless HPC/cluster GPU nodes (and containers). NVIDIA GPUs render fine
    through EGL there, so try the platform default first (CGL on macOS, GLX on an X11
    desktop) and fall back to the EGL backend.

    glcontext's EGL backend ``dlopen``\\s the *unversioned* ``libEGL.so``/``libGL.so``,
    which only the ``-dev`` packages ship — a runtime-only image (Debian ``libegl1`` +
    Mesa, e.g. the HF Space) has just ``libEGL.so.1`` and fails with ``libEGL.so not
    loaded``. So retry EGL with the versioned SONAMEs before giving up.
    """
    try:
        return moderngl.create_standalone_context()
    except Exception:
        try:
            return moderngl.create_standalone_context(backend="egl")
        except Exception:
            return moderngl.create_standalone_context(backend="egl", libegl="libEGL.so.1", libgl="libGL.so.1")


class ModernGLRenderer:
    """GPU renderer with the same call surface the demo uses on the pytorch3d ``Renderer``."""

    def __init__(self, width, height, device=None, faces=None, K=None):
        import moderngl

        self.W, self.H = int(width), int(height)
        self.faces = _np(faces).astype("i4") if faces is not None else None
        self.K = _np(K).astype("f4")
        self.proj = projection_from_K(self.K, self.W, self.H)
        self.R = np.eye(3, dtype="f4")
        self.T = np.zeros(3, dtype="f4")
        self.ground = None  # (verts, faces, colors)

        self._mgl = moderngl
        self.ctx = _standalone_context(moderngl)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.prog = self.ctx.program(vertex_shader=_VERT_SHADER, fragment_shader=_FRAG_SHADER)
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((self.W, self.H), 4)],
            depth_attachment=self.ctx.depth_texture((self.W, self.H)),
        )

    # ---- camera / ground (mirror the pytorch3d Renderer API the demo calls) ----
    def create_camera(self, R, T):
        self.R, self.T = _np(R).reshape(3, 3).astype("f4"), _np(T).reshape(3).astype("f4")
        return self

    def set_ground(self, length, center_x, center_z):
        v, f, c, _ = checkerboard_geometry(length=length, c1=center_x, c2=center_z, up="y")
        self.ground = (v.astype("f4"), f.astype("i4"), c[:, :3].astype("f4"))

    # ---- drawing ----
    def _view(self, R, T):
        # The global cameras follow pytorch3d's convention (+x left, +y up); flip x,y to the
        # OpenCV frame (+x right, +y down) our projection assumes. det=+1, so no mirroring.
        flip = np.diag([-1.0, -1.0, 1.0]).astype("f4")
        m = np.eye(4, dtype="f4")
        m[:3, :3] = flip @ R
        m[:3, 3] = flip @ T
        return m

    def _draw_meshes(self, meshes):
        for verts, faces, colors, shaded in meshes:
            self.prog["u_shade"].value = 1.0 if shaded else 0.0
            vbo = self.ctx.buffer(np.ascontiguousarray(verts, "f4").tobytes())
            cbo = self.ctx.buffer(np.ascontiguousarray(colors, "f4").tobytes())
            ibo = self.ctx.buffer(np.ascontiguousarray(faces, "i4").tobytes())
            vao = self.ctx.vertex_array(self.prog, [(vbo, "3f", "in_pos"), (cbo, "3f", "in_col")], ibo)
            vao.render()
            vao.release()
            vbo.release()
            cbo.release()
            ibo.release()

    def _draw(self, meshes, view, background, bg_color, on_top=None):
        """meshes: list of (verts(N,3), faces(M,3), colors(N,3), shaded:bool). ``on_top`` meshes (e.g. a
        skeleton overlaid on the body) draw with depth-test off so they stay fully visible over the
        mesh. Returns (H,W,3) uint8."""
        self.fbo.use()
        self.ctx.clear(*bg_color, 0.0, depth=1.0)
        self.prog["u_proj"].write(np.ascontiguousarray(self.proj.T).tobytes())
        self.prog["u_view"].write(np.ascontiguousarray(view.T).tobytes())
        self._draw_meshes(meshes)
        if on_top:
            self.ctx.disable(self._mgl.DEPTH_TEST)
            self._draw_meshes(on_top)
            self.ctx.enable(self._mgl.DEPTH_TEST)
        # read back RGBA, flip (GL origin is bottom-left)
        data = np.frombuffer(self.fbo.read(components=4), "u1").reshape(self.H, self.W, 4)
        data = np.flipud(data)
        rgb, alpha = data[..., :3].astype(np.float32), (data[..., 3:4].astype(np.float32) / 255.0)
        if background is None:
            return rgb.astype(np.uint8)
        bg = np.asarray(background, np.float32)
        return (rgb * alpha + bg * (1 - alpha)).astype(np.uint8)

    def _prep_extra(self, extra_meshes):
        """Normalize caller-supplied meshes to (verts f4, faces i4, colors f4, shaded). Each item is
        (verts, faces, colors) or (verts, faces, colors, shaded); shaded defaults True. Used to draw
        a skeleton (spheres + cylinders) on top of, or instead of, the body."""
        out = []
        for m in extra_meshes:
            shaded = bool(m[3]) if len(m) > 3 else True
            out.append(
                (
                    _np(m[0]).reshape(-1, 3).astype("f4"),
                    _np(m[1]).astype("i4"),
                    np.asarray(_np(m[2]), "f4").reshape(-1, 3),
                    shaded,
                )  # fmt: skip
            )
        return out

    def _split_extra(self, extra_meshes, extra_on_top):
        """Prep extra meshes and return (base_part, on_top_part) per ``extra_on_top``."""
        prepped = self._prep_extra(extra_meshes)
        return ([], prepped) if extra_on_top else (prepped, None)

    def render_mesh(
        self, verts, background, colors=(0.8, 0.8, 0.8), extra_meshes=None, draw_body=True, extra_on_top=False, **_
    ):
        """In-camera: SMPL verts are already in camera frame → identity view, composite on video.

        ``extra_meshes`` (e.g. a skeleton) draw in the same camera frame; ``extra_on_top`` keeps them
        visible over the body; ``draw_body=False`` renders only those."""
        meshes = []
        if draw_body and verts is not None:
            v = _np(verts).reshape(-1, 3).astype("f4")
            col = np.broadcast_to(np.asarray(colors, "f4"), v.shape).copy()
            meshes.append((v, self.faces, col, True))
        base, on_top = self._split_extra(extra_meshes, extra_on_top) if extra_meshes else ([], None)
        meshes.extend(base)
        return self._draw(meshes, np.eye(4, dtype="f4"), background, (0, 0, 0), on_top=on_top)

    def render_with_ground(
        self, verts=None, colors=None, cameras=None, lights=None, faces=None, extra_meshes=None, draw_body=True,
        extra_on_top=False, **_,
    ):  # fmt: skip
        """World view: body + checkerboard ground, camera = (R, T) from create_camera.

        ``extra_meshes`` overlay a skeleton (``extra_on_top`` to keep it visible over the body);
        ``draw_body=False`` gives skeleton-only (ground + skeleton), so the same call serves the
        mesh, skeleton, and overlay videos."""
        view = self._view(self.R, self.T)
        meshes = []
        if draw_body and verts is not None:
            v = _np(verts).reshape(-1, 3).astype("f4")
            col = np.broadcast_to(np.asarray(_np(colors), "f4").reshape(-1, 3)[:1], v.shape).copy()
            meshes.append((v, self.faces if faces is None else _np(faces).astype("i4"), col, True))
        if self.ground is not None:
            gv, gf, gc = self.ground
            meshes.append((gv, gf, gc, False))
        base, on_top = self._split_extra(extra_meshes, extra_on_top) if extra_meshes else ([], None)
        meshes.extend(base)
        return self._draw(meshes, view, None, (1.0, 1.0, 1.0), on_top=on_top)
