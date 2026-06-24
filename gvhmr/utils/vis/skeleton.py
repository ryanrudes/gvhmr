"""SMPL skeleton geometry for the overlay renderer.

The model regresses the 24-joint SMPL skeleton (``J_regressor`` over the SMPL mesh; the
SMPL-X body is converted to SMPL first). This module turns a frame's joint positions into a
renderable **mesh** — small icospheres at the joints, thin cylinders along the bones — so the
existing moderngl/pytorch3d mesh path can draw it with no special primitives. It also resolves
*subsets* (``"legs"``, ``"left_arm"``, or explicit joint names/indices), so callers can render
just the joints/links they care about; a bone is drawn only when both its endpoints survive
the subset.
"""

from __future__ import annotations

import numpy as np

# SMPL's 24 body joints, in regressor order.
SMPL_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar",
    "right_collar", "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hand", "right_hand",
]  # fmt: skip
SMPL_JOINT_IDX = {name: i for i, name in enumerate(SMPL_JOINT_NAMES)}

# Kinematic tree: parent of each joint (root = -1). Bones are (parent, child) edges.
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
SMPL_BONES = [(SMPL_PARENTS[j], j) for j in range(1, len(SMPL_JOINT_NAMES))]

# Named groups for quick subset selection (`--skeleton-joints legs`, etc.).
JOINT_GROUPS: dict[str, list[str]] = {
    "left_leg": ["left_hip", "left_knee", "left_ankle", "left_foot"],
    "right_leg": ["right_hip", "right_knee", "right_ankle", "right_foot"],
    "left_arm": ["left_collar", "left_shoulder", "left_elbow", "left_wrist", "left_hand"],
    "right_arm": ["right_collar", "right_shoulder", "right_elbow", "right_wrist", "right_hand"],
    "spine": ["pelvis", "spine1", "spine2", "spine3", "neck", "head"],
    "head": ["neck", "head"],
    "hands": ["left_wrist", "left_hand", "right_wrist", "right_hand"],
    "feet": ["left_ankle", "left_foot", "right_ankle", "right_foot"],
}
JOINT_GROUPS["legs"] = ["pelvis", *JOINT_GROUPS["left_leg"], *JOINT_GROUPS["right_leg"]]
JOINT_GROUPS["arms"] = [*JOINT_GROUPS["left_arm"], *JOINT_GROUPS["right_arm"]]
JOINT_GROUPS["torso"] = ["pelvis", "spine1", "spine2", "spine3", "neck", "left_collar", "right_collar"]
JOINT_GROUPS["upper_body"] = sorted({*JOINT_GROUPS["arms"], *JOINT_GROUPS["torso"], *JOINT_GROUPS["head"]})
JOINT_GROUPS["lower_body"] = JOINT_GROUPS["legs"]

# Default per-joint coloring: left side warm, right side cool, center neutral (clear in 3D).
_LEFT, _RIGHT, _CENTER = (0.90, 0.45, 0.25), (0.25, 0.55, 0.90), (0.85, 0.85, 0.30)


def resolve_joint_subset(spec: str | None) -> list[int]:
    """Resolve a subset spec to sorted SMPL joint indices.

    ``spec`` is a comma-separated list of group names (``legs``, ``left_arm``, …), joint names
    (``left_knee``), or integer indices; ``None``/``"all"`` selects every joint. Unknown tokens
    raise ``ValueError`` with the valid options.
    """
    if spec is None or spec.strip().lower() in ("", "all"):
        return list(range(len(SMPL_JOINT_NAMES)))
    out: set[int] = set()
    for tok in (t.strip() for t in spec.split(",") if t.strip()):
        if tok in JOINT_GROUPS:
            out.update(SMPL_JOINT_IDX[n] for n in JOINT_GROUPS[tok])
        elif tok in SMPL_JOINT_IDX:
            out.add(SMPL_JOINT_IDX[tok])
        elif tok.isdigit() and int(tok) < len(SMPL_JOINT_NAMES):
            out.add(int(tok))
        else:
            valid = ", ".join([*JOINT_GROUPS, *SMPL_JOINT_NAMES])
            raise ValueError(f"Unknown skeleton joint/group {tok!r}. Valid: all, {valid}")
    return sorted(out)


def default_joint_colors(indices: list[int]) -> np.ndarray:
    """Left/right/center color per selected joint, (len(indices), 3)."""
    cols = []
    for j in indices:
        name = SMPL_JOINT_NAMES[j]
        cols.append(_LEFT if name.startswith("left_") else _RIGHT if name.startswith("right_") else _CENTER)
    return np.asarray(cols, "f4")


def _icosphere(subdiv: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Unit icosphere (radius 1) by subdividing an icosahedron ``subdiv`` times."""
    t = (1.0 + 5.0**0.5) / 2.0
    verts = np.array(
        [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t),
         (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)], "f8")  # fmt: skip
    faces = np.array(
        [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
         (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
         (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)], "i4")  # fmt: skip
    for _ in range(subdiv):
        mid: dict[tuple[int, int], int] = {}
        vlist = list(verts)
        new_faces = []

        def midpoint(a: int, b: int, mid=mid, vlist=vlist) -> int:  # defaults bind this iteration's state
            key = (a, b) if a < b else (b, a)
            if key not in mid:
                mid[key] = len(vlist)
                vlist.append((vlist[a] + vlist[b]) / 2.0)
            return mid[key]

        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        verts = np.asarray(vlist, "f8")
        faces = np.asarray(new_faces, "i4")
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    return verts.astype("f4"), faces


def _cylinder(segments: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Unit cylinder: radius 1, from z=0 to z=1, with end caps."""
    ang = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    ring = np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], 1)
    bottom, top = ring.copy(), ring + [0, 0, 1]
    cb, ct = [0, 0, 0], [0, 0, 1]
    verts = np.concatenate([bottom, top, [cb], [ct]], 0).astype("f4")
    ib, it, icb, ict = 0, segments, 2 * segments, 2 * segments + 1
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces += [(ib + i, ib + j, it + j), (ib + i, it + j, it + i)]  # side
        faces += [(icb, ib + j, ib + i), (ict, it + i, it + j)]  # caps
    return verts, np.asarray(faces, "i4")


def _rot_z_to(d: np.ndarray) -> np.ndarray:
    """Rotation mapping +z onto unit vector ``d`` (Rodrigues; handles the antiparallel case)."""
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, d)
    c = float(np.dot(z, d))
    s = float(np.linalg.norm(v))
    if s < 1e-8:
        return np.eye(3, dtype="f4") if c > 0 else np.diag([1.0, -1.0, -1.0]).astype("f4")
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return (np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))).astype("f4")


# unit primitives, built once
_SPH_V, _SPH_F = _icosphere(1)
_CYL_V, _CYL_F = _cylinder(12)


def build_skeleton_mesh(
    joints: np.ndarray,
    indices: list[int],
    *,
    joint_radius: float = 0.035,
    bone_radius: float = 0.018,
    joint_colors: np.ndarray | None = None,
    bone_color: tuple[float, float, float] = (0.95, 0.95, 0.95),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a single (verts, faces, colors) mesh for one frame's skeleton subset.

    ``joints``: (J, 3) all SMPL joints; ``indices``: which to draw. A sphere is placed at every
    selected joint and a cylinder along every bone whose both endpoints are selected. Returns a
    combined mesh ready for the renderer (per-vertex colors).
    """
    joints = np.asarray(joints, "f4")
    sel = set(indices)
    jcols = default_joint_colors(indices) if joint_colors is None else np.asarray(joint_colors, "f4")
    vparts, fparts, cparts, off = [], [], [], 0

    def add(v: np.ndarray, f: np.ndarray, color: np.ndarray) -> None:
        nonlocal off
        vparts.append(v)
        fparts.append(f + off)
        cparts.append(np.broadcast_to(color, v.shape).copy())
        off += len(v)

    for k, j in enumerate(indices):  # joints → spheres
        add(_SPH_V * joint_radius + joints[j], _SPH_F, jcols[k])
    bcol = np.asarray(bone_color, "f4")
    for parent, child in SMPL_BONES:  # bones → cylinders (only if both endpoints kept)
        if parent not in sel or child not in sel:
            continue
        a, b = joints[parent], joints[child]
        d = b - a
        length = float(np.linalg.norm(d))
        if length < 1e-6:
            continue
        R = _rot_z_to(d / length)
        v = (_CYL_V * [bone_radius, bone_radius, length]) @ R.T + a
        add(v.astype("f4"), _CYL_F, bcol)

    if not vparts:  # empty subset → degenerate single point (renderer-safe)
        return np.zeros((1, 3), "f4"), np.zeros((1, 3), "i4"), np.zeros((1, 3), "f4")
    return np.concatenate(vparts, 0), np.concatenate(fparts, 0), np.concatenate(cparts, 0)
