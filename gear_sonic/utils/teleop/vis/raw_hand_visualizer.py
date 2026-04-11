"""Real-time raw hand visualizer for XRoboToolkit/OpenXR 26-joint hand states."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    import pyvista as pv

    PYVISTA_AVAILABLE = True
except ImportError:
    pv = None
    PYVISTA_AVAILABLE = False


XRT_HAND_BONE_EDGES = [
    (0, 1),
    (1, 2), (2, 3), (3, 4), (4, 5),
    (1, 6), (6, 7), (7, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13), (13, 14), (14, 15),
    (1, 16), (16, 17), (17, 18), (18, 19), (19, 20),
    (1, 21), (21, 22), (22, 23), (23, 24), (24, 25),
]
XRT_WRIST_INDEX = 1


def extract_xrt_hand_joint_positions(hand_state) -> np.ndarray:
    """Return the raw 26x3 joint positions from an XRoboToolkit hand state."""

    state = np.asarray(hand_state, dtype=np.float64)
    if state.ndim != 2 or state.shape[0] < 26 or state.shape[1] < 7:
        raise ValueError(f"hand_state must have shape (>=26, >=7), got {state.shape}")
    return state[:26, :3].copy()


def extract_xrt_hand_joint_positions_wrist_local(hand_state) -> np.ndarray:
    """Return raw 26x3 joint positions in the wrist-local frame."""

    state = np.asarray(hand_state, dtype=np.float64)
    positions = extract_xrt_hand_joint_positions(state)
    wrist_position = positions[XRT_WRIST_INDEX].copy()
    local_positions = positions - wrist_position

    wrist_quat = state[XRT_WRIST_INDEX, 3:7].copy()
    wrist_quat_norm = np.linalg.norm(wrist_quat)
    if np.isfinite(wrist_quat_norm) and wrist_quat_norm > 1e-8:
        wrist_rot = R.from_quat(wrist_quat / wrist_quat_norm)
        local_positions = wrist_rot.inv().apply(local_positions)

    return local_positions


def _make_bone_polydata(points: np.ndarray):
    poly = pv.PolyData()
    poly.points = np.asarray(points, dtype=np.float64)
    lines = []
    for start, end in XRT_HAND_BONE_EDGES:
        lines.extend((2, start, end))
    poly.lines = np.asarray(lines, dtype=np.int32)
    return poly


class RawHandPoseVisualizer:
    """PyVista window that renders raw left/right hand skeletons from XRoboToolkit."""

    LEFT_COLOR = "#7CFC90"
    RIGHT_COLOR = "#87CEFA"
    LEFT_BONE_COLOR = "#4CAF50"
    RIGHT_BONE_COLOR = "#1E88E5"

    def __init__(
        self,
        point_size: float = 14.0,
        line_width: float = 4.0,
        *,
        left_only: bool = False,
        wrist_local: bool = False,
    ):
        if not PYVISTA_AVAILABLE:
            raise ImportError("PyVista is required. Install with: pip install pyvista")

        self.point_size = point_size
        self.line_width = line_width
        self.left_only = left_only
        self.wrist_local = wrist_local
        self.plotter = None
        self._initialized = False
        self._left_points = None
        self._right_points = None
        self._left_bones = None
        self._right_bones = None

    def create_realtime_plotter(self, window_size=(960, 720)):
        pv.set_plot_theme("dark")
        self.plotter = pv.Plotter(window_size=window_size)
        self.plotter.set_background("black")
        self.plotter.add_axes()
        self.plotter.show_grid(color="gray", font_size=8)
        title = "Raw XRoboToolkit Hand Joints"
        if self.left_only and self.wrist_local:
            title = "Left Raw Hand (Wrist-Local)"
        elif self.left_only:
            title = "Left Raw Hand"
        self.plotter.add_text(
            title + ("\nLeft=green  Right=blue" if not self.left_only else ""),
            position="upper_left",
            font_size=10,
            color="white",
        )
        if self.left_only and self.wrist_local:
            self.plotter.camera_position = [(0.35, -0.42, 0.22), (0.0, 0.0, 0.0), (0, 0, 1)]
        else:
            self.plotter.camera_position = [(0.8, -0.8, 0.6), (0.0, 0.0, 0.0), (0, 0, 1)]

        zeros = np.zeros((26, 3), dtype=np.float64)
        self._left_points = pv.PolyData(zeros.copy())
        self._left_bones = _make_bone_polydata(zeros.copy())

        self.plotter.add_mesh(
            self._left_bones,
            color=self.LEFT_BONE_COLOR,
            line_width=self.line_width,
            render_lines_as_tubes=True,
        )
        self.plotter.add_mesh(
            self._left_points,
            color=self.LEFT_COLOR,
            point_size=self.point_size,
            render_points_as_spheres=True,
        )

        if not self.left_only:
            self._right_points = pv.PolyData(zeros.copy())
            self._right_bones = _make_bone_polydata(zeros.copy())
            self.plotter.add_mesh(
                self._right_bones,
                color=self.RIGHT_BONE_COLOR,
                line_width=self.line_width,
                render_lines_as_tubes=True,
            )
            self.plotter.add_mesh(
                self._right_points,
                color=self.RIGHT_COLOR,
                point_size=self.point_size,
                render_points_as_spheres=True,
            )
        else:
            self._right_points = None
            self._right_bones = None

        self.plotter.show(interactive_update=True, auto_close=False)
        self._initialized = True
        return self.plotter

    def update_hands(self, left_hand_state=None, right_hand_state=None):
        if not self._initialized or self.plotter is None:
            raise RuntimeError("create_realtime_plotter() must be called before update_hands().")

        left_positions = self._prepare_positions(left_hand_state)

        self._left_points.points = left_positions
        self._left_bones.points = left_positions
        if not self.left_only:
            right_positions = self._prepare_positions(right_hand_state)
            self._right_points.points = right_positions
            self._right_bones.points = right_positions

    def render(self):
        if self.plotter is not None and self.is_open:
            self.plotter.render()

    def close(self):
        if self.plotter is not None:
            self.plotter.close()
            self.plotter = None
            self._initialized = False

    @property
    def is_open(self) -> bool:
        if self.plotter is None:
            return False
        try:
            return self.plotter.ren_win is not None and not self.plotter._closed
        except (AttributeError, RuntimeError):
            return False

    def _prepare_positions(self, hand_state: Optional[np.ndarray]) -> np.ndarray:
        if hand_state is None:
            return np.zeros((26, 3), dtype=np.float64)
        if self.wrist_local:
            return extract_xrt_hand_joint_positions_wrist_local(hand_state)
        return extract_xrt_hand_joint_positions(hand_state)
