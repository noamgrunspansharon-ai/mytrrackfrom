"""
pose_engine.py
==============
MediaPipe Pose (BlazePose, 33 landmarks) wrapper + OpenCV drawing/rendering utilities.

* PoseEngine              - frame -> 33 keypoints
* extract_keypoints()     - whole video -> VideoPoseData (cached to .npz for pro reference clips)
* draw_skeleton()         - left/right colour-coded skeleton overlay
* render_annotated_video()- single annotated video (key-frame banners + angle marks)
* build_phase_alignment() - maps every user frame to the pro frame in the SAME movement phase
* render_comparison_video()- user | pro side-by-side video, skeletons drawn, phases synced
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from biomechanics import calculate_angle

# BlazePose body connections (identical to mediapipe's POSE_CONNECTIONS for landmarks 11-32)
POSE_CONNECTIONS: List[Tuple[int, int]] = [
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), (12, 14), (14, 16), (16, 18),
    (16, 20), (16, 22), (18, 20), (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)]

COL_LEFT = (255, 160, 40)     # BGR  (blue-ish)  anatomical LEFT
COL_RIGHT = (60, 120, 255)    # BGR  (orange-ish) anatomical RIGHT
COL_MID = (60, 230, 60)
COL_HILITE = (0, 255, 255)
PROGRESS = Optional[Callable[[float], None]]


# --------------------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------------------
@dataclass
class PoseResult:
    px: np.ndarray            # (33,2) pixel coords in the processed frame
    visibility: np.ndarray    # (33,)


@dataclass
class VideoPoseData:
    keypoints: np.ndarray     # (N,33,3) = x_px, y_px, visibility ; NaN where nothing detected
    fps: float
    size: Tuple[int, int]     # (width, height) of the PROCESSED frames

    @property
    def n(self) -> int:
        return len(self.keypoints)

    @property
    def n_valid(self) -> int:
        return int(np.isfinite(self.keypoints[:, 23, 0]).sum())

    def save(self, path: str) -> None:
        np.savez_compressed(path, keypoints=self.keypoints, fps=self.fps, size=np.array(self.size))

    @staticmethod
    def load(path: str) -> "VideoPoseData":
        d = np.load(path)
        return VideoPoseData(d["keypoints"], float(d["fps"]), (int(d["size"][0]), int(d["size"][1])))


# --------------------------------------------------------------------------------------
# Pose estimation
# --------------------------------------------------------------------------------------
class PoseEngine:
    """Thin context-managed wrapper around mediapipe.solutions.pose.Pose."""

    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5, smooth_landmarks: bool = True):
        try:
            import mediapipe as mp
            pose_mod = mp.solutions.pose
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "This app needs the MediaPipe 'solutions' API (BlazePose). Newer MediaPipe releases removed it - "
                "install the pinned version:  pip install mediapipe==0.10.14   (Python 3.9-3.12)") from exc
        self._pose = pose_mod.Pose(static_image_mode=False, model_complexity=model_complexity,
                                   smooth_landmarks=smooth_landmarks, enable_segmentation=False,
                                   min_detection_confidence=min_detection_confidence,
                                   min_tracking_confidence=min_tracking_confidence)

    def process_frame(self, frame_bgr: np.ndarray) -> Optional[PoseResult]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self._pose.process(rgb)
        if res.pose_landmarks is None:
            return None
        lm = res.pose_landmarks.landmark
        px = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
        vis = np.array([p.visibility for p in lm], dtype=np.float32)
        return PoseResult(px, vis)

    def close(self) -> None:
        self._pose.close()

    def __enter__(self) -> "PoseEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _resize_max_width(frame: np.ndarray, max_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if max_width and w > max_width:
        s = max_width / w
        return cv2.resize(frame, (max_width, int(round(h * s))), interpolation=cv2.INTER_AREA)
    return frame


def _read_fps(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    return float(fps) if fps and np.isfinite(fps) and 1 < fps < 1000 else 30.0


def extract_keypoints(video_path: str, engine: PoseEngine, max_width: int = 640,
                      progress_cb: PROGRESS = None) -> VideoPoseData:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    fps = _read_fps(cap)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    rows, size, i = [], (0, 0), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = _resize_max_width(frame, max_width)
        size = (frame.shape[1], frame.shape[0])
        pose = engine.process_frame(frame)
        if pose is None:
            rows.append(np.full((33, 3), np.nan, dtype=np.float32))
        else:
            rows.append(np.concatenate([pose.px, pose.visibility[:, None]], axis=1))
        i += 1
        if progress_cb and total and i % 5 == 0:
            progress_cb(min(1.0, i / total))
    cap.release()
    if not rows:
        raise IOError("The video contains no readable frames.")
    if progress_cb:
        progress_cb(1.0)
    return VideoPoseData(np.stack(rows).astype(np.float32), fps, size)


def extract_keypoints_cached(video_path: str, cache_dir: str, engine_factory: Callable[[], PoseEngine],
                             max_width: int = 640, progress_cb: PROGRESS = None) -> VideoPoseData:
    """Like extract_keypoints but stores/reads an .npz so pro reference clips are only processed once."""
    os.makedirs(cache_dir, exist_ok=True)
    st = os.stat(video_path)
    key = hashlib.md5(f"{os.path.abspath(video_path)}|{st.st_size}|{st.st_mtime}|{max_width}".encode()).hexdigest()[:16]
    path = os.path.join(cache_dir, f"{key}.npz")
    if os.path.exists(path):
        try:
            return VideoPoseData.load(path)
        except Exception:
            pass
    with engine_factory() as eng:
        data = extract_keypoints(video_path, eng, max_width, progress_cb)
    data.save(path)
    return data


# --------------------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------------------
def put_text(img, text, org, scale=0.6, color=(255, 255, 255), thick=1) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_skeleton(frame: np.ndarray, px: np.ndarray, vis: np.ndarray, min_vis: float = 0.4,
                  thickness: Optional[int] = None) -> np.ndarray:
    """Draw body skeleton (landmarks 11-32) + nose. Left = blue, right = orange, centre = green."""
    if px is None or not np.isfinite(px).any():
        return frame
    th = thickness or max(2, int(round(frame.shape[1] / 300)))
    for a, b in POSE_CONNECTIONS:
        if a < 11 or b < 11:
            continue
        if vis[a] < min_vis or vis[b] < min_vis or not (np.isfinite(px[a]).all() and np.isfinite(px[b]).all()):
            continue
        col = COL_LEFT if (a % 2 and b % 2) else COL_RIGHT if (a % 2 == 0 and b % 2 == 0) else COL_MID
        cv2.line(frame, tuple(int(v) for v in px[a]), tuple(int(v) for v in px[b]), col, th, cv2.LINE_AA)
    for i in range(11, 33):
        if vis[i] >= min_vis and np.isfinite(px[i]).all():
            cv2.circle(frame, tuple(int(v) for v in px[i]), th + 1, (255, 255, 255), -1, cv2.LINE_AA)
    if vis[0] >= min_vis and np.isfinite(px[0]).all():
        cv2.circle(frame, tuple(int(v) for v in px[0]), th + 2, COL_MID, -1, cv2.LINE_AA)
    return frame


def draw_angle_marks(frame: np.ndarray, px: np.ndarray, triplets: Sequence[Tuple[int, int, int]]) -> None:
    scale = max(0.5, frame.shape[1] / 900)
    for a, b, c in triplets:
        pts = [px[a], px[b], px[c]]
        if not all(np.isfinite(p).all() for p in pts):
            continue
        th = max(3, int(round(frame.shape[1] / 220)))
        cv2.line(frame, tuple(int(v) for v in px[a]), tuple(int(v) for v in px[b]), COL_HILITE, th, cv2.LINE_AA)
        cv2.line(frame, tuple(int(v) for v in px[b]), tuple(int(v) for v in px[c]), COL_HILITE, th, cv2.LINE_AA)
        cv2.circle(frame, tuple(int(v) for v in px[b]), th + 3, (0, 0, 255), -1, cv2.LINE_AA)
        ang = calculate_angle(px[a], px[b], px[c])
        put_text(frame, f"{ang:.0f}\u00b0", (int(px[b][0]) + 10, int(px[b][1]) - 10), scale, COL_HILITE, 2)


def _expand_schedule(annotations: Dict[int, List[str]], marks: Dict[int, List[Tuple[int, int, int]]],
                     hold: int, n: int) -> Tuple[Dict[int, List[str]], Dict[int, List[Tuple[int, int, int]]]]:
    texts: Dict[int, List[str]] = {}
    trip: Dict[int, List[Tuple[int, int, int]]] = {}
    for f, lines in (annotations or {}).items():
        for i in range(max(0, f - hold), min(n, f + hold + 1)):
            texts.setdefault(i, []).extend(lines)
    for f, tl in (marks or {}).items():
        for i in range(max(0, f - hold), min(n, f + hold + 1)):
            trip.setdefault(i, []).extend(tl)
    return texts, trip


def _phase_at(key_frames: Dict[str, int], idx: int, hold: int) -> Optional[str]:
    best = None
    for label, f in key_frames.items():
        d = abs(idx - f)
        if d <= hold and (best is None or d < best[0]):
            best = (d, label)
    return best[1] if best else None


def _annotate(frame, data: VideoPoseData, idx: int, texts, trips, phase: Optional[str]) -> np.ndarray:
    kp = data.keypoints[min(idx, data.n - 1)]
    px, vis = kp[:, :2], np.nan_to_num(kp[:, 2], nan=0.0)
    draw_skeleton(frame, px, vis)
    if idx in trips:
        draw_angle_marks(frame, px, trips[idx])
    scale = max(0.5, frame.shape[1] / 900)
    y = int(28 * scale)
    if phase:
        put_text(frame, f"PHASE: {phase.upper()}", (10, y), scale * 0.9, (0, 255, 255), 2)
        y += int(26 * scale)
    for line in dict.fromkeys(texts.get(idx, [])):
        put_text(frame, line, (10, y), scale * 0.75, (255, 255, 255), 1)
        y += int(24 * scale)
    return frame


# --------------------------------------------------------------------------------------
# Encoding helpers
# --------------------------------------------------------------------------------------
def find_ffmpeg() -> Optional[str]:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def transcode_h264(src: str, dst: str) -> bool:
    """Re-encode OpenCV's mp4v output to browser/phone-playable H.264. Returns True on success."""
    exe = find_ffmpeg()
    if not exe:
        return False
    cmd = [exe, "-y", "-i", src, "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
           "-crf", "24", "-movflags", "+faststart", dst]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
        return os.path.exists(dst) and os.path.getsize(dst) > 0
    except Exception:
        return False


def _finalize(raw_path: str, out_path: str) -> Tuple[str, bool]:
    if transcode_h264(raw_path, out_path):
        try:
            os.remove(raw_path)
        except OSError:
            pass
        return out_path, True
    return raw_path, False


# --------------------------------------------------------------------------------------
# Solo annotated video
# --------------------------------------------------------------------------------------
def render_annotated_video(video_path: str, data: VideoPoseData, out_path: str,
                           annotations: Optional[Dict[int, List[str]]] = None,
                           angle_marks: Optional[Dict[int, List[Tuple[int, int, int]]]] = None,
                           key_frames: Optional[Dict[str, int]] = None, speed: float = 1.0,
                           progress_cb: PROGRESS = None) -> Tuple[str, Dict[str, np.ndarray], bool]:
    """Returns (path, stills{label: RGB image}, browser_playable)."""
    cap = cv2.VideoCapture(video_path)
    w, h = data.size
    raw = out_path + ".raw.mp4"
    writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, data.fps * speed), (w, h))
    hold = max(2, int(round(0.35 * data.fps)))
    texts, trips = _expand_schedule(annotations or {}, angle_marks or {}, hold, data.n)
    key_frames = key_frames or {}
    by_frame = {f: k for k, f in key_frames.items()}
    stills: Dict[str, np.ndarray] = {}
    i = 0
    while i < data.n:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        _annotate(frame, data, i, texts, trips, _phase_at(key_frames, i, hold))
        writer.write(frame)
        if i in by_frame:
            stills[by_frame[i]] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        i += 1
        if progress_cb and i % 5 == 0:
            progress_cb(min(1.0, i / data.n))
    cap.release()
    writer.release()
    path, playable = _finalize(raw, out_path)
    return path, stills, playable


# --------------------------------------------------------------------------------------
# Phase alignment + side-by-side comparison
# --------------------------------------------------------------------------------------
def build_phase_alignment(user_kf: Dict[str, int], pro_kf: Dict[str, int], n_user: int, n_pro: int
                          ) -> Tuple[np.ndarray, List[Tuple[str, int, int]]]:
    """
    For every USER frame return the PRO frame that is in the same movement phase.
    Matching key frames (takeoff, apex, landing ...) become anchors; between anchors the pro clip is
    linearly time-warped, so both athletes hit takeoff / apex / landing on the same output frame.
    Returns (pro_index_per_user_frame, anchors[(label, user_frame, pro_frame)]).
    """
    pairs = sorted((user_kf[k], pro_kf[k], k) for k in user_kf if k in pro_kf)
    anchors: List[Tuple[str, int, int]] = []
    lu = lp = -1
    for u, p, k in pairs:
        if u > lu and p > lp:
            anchors.append((k, int(u), int(p)))
            lu, lp = u, p
    if not anchors:
        idx = np.linspace(0, n_pro - 1, n_user)
    else:
        u0, p0 = anchors[0][1:]
        uL, pL = anchors[-1][1:]
        xs = [a[1] for a in anchors]
        ys = [a[2] for a in anchors]
        if u0 > 0:
            xs, ys = [0] + xs, [max(0, p0 - u0)] + ys
        if uL < n_user - 1:
            xs, ys = xs + [n_user - 1], ys + [min(n_pro - 1, pL + (n_user - 1 - uL))]
        idx = np.interp(np.arange(n_user), xs, ys)
    idx = np.maximum.accumulate(np.clip(np.rint(idx), 0, n_pro - 1)).astype(int)
    return idx, anchors


def _even(v: int) -> int:
    return v - (v % 2)


def render_comparison_video(user_video: str, user_data: VideoPoseData, user_ann: Dict,
                            pro_video: str, pro_data: VideoPoseData, pro_ann: Dict,
                            out_path: str, pro_label: str = "PRO", speed: float = 0.5,
                            panel_height: int = 480, progress_cb: PROGRESS = None
                            ) -> Tuple[str, Dict[str, np.ndarray], bool, List[Tuple[str, int, int]]]:
    """
    Side-by-side YOU | PRO video with skeleton overlays, phase-synchronised.
    user_ann / pro_ann: dicts with keys annotations, angle_marks, key_frames (from AnalysisResult).
    Returns (path, phase_stills, browser_playable, anchors).
    """
    pro_idx, anchors = build_phase_alignment(user_ann["key_frames"], pro_ann["key_frames"], user_data.n, pro_data.n)
    uh = user_data.fps
    hold = max(2, int(round(0.35 * uh)))
    u_txt, u_trip = _expand_schedule(user_ann["annotations"], user_ann["angle_marks"], hold, user_data.n)
    p_hold = max(2, int(round(0.35 * pro_data.fps)))
    p_txt, p_trip = _expand_schedule(pro_ann["annotations"], pro_ann["angle_marks"], p_hold, pro_data.n)

    ph = _even(panel_height)
    uw = _even(int(round(user_data.size[0] * ph / user_data.size[1])))
    pw = _even(int(round(pro_data.size[0] * ph / pro_data.size[1])))
    bar = 36
    W, H = uw + pw, ph + bar

    raw = out_path + ".raw.mp4"
    writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, uh * speed), (W, H))
    capu, capp = cv2.VideoCapture(user_video), cv2.VideoCapture(pro_video)
    anchor_by_user = {u: lab for lab, u, _ in anchors}
    stills: Dict[str, np.ndarray] = {}

    pro_cur, pro_frame = -1, None
    for i in range(user_data.n):
        ok, uf = capu.read()
        if not ok:
            break
        uf = cv2.resize(uf, user_data.size, interpolation=cv2.INTER_AREA)
        target = int(pro_idx[i])
        while pro_cur < target:
            ok2, f2 = capp.read()
            if not ok2:
                break
            pro_frame, pro_cur = cv2.resize(f2, pro_data.size, interpolation=cv2.INTER_AREA), pro_cur + 1
        if pro_frame is None:
            pro_frame = np.zeros((pro_data.size[1], pro_data.size[0], 3), np.uint8)

        # shared phase label (taken from the user's phase anchors, so both panels show the same text)
        phase = _phase_at({lab: u for lab, u, _ in anchors}, i, hold)
        uf = _annotate(uf.copy(), user_data, i, u_txt, u_trip, phase)
        pf = _annotate(pro_frame.copy(), pro_data, max(pro_cur, 0), p_txt, p_trip, phase)
        uf = cv2.resize(uf, (uw, ph), interpolation=cv2.INTER_AREA)
        pf = cv2.resize(pf, (pw, ph), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((H, W, 3), np.uint8)
        canvas[:bar, :uw] = (170, 90, 20)
        canvas[:bar, uw:] = (40, 40, 160)
        put_text(canvas, "YOU", (10, 25), 0.8, (255, 255, 255), 2)
        put_text(canvas, pro_label[:48], (uw + 10, 25), 0.8, (255, 255, 255), 2)
        canvas[bar:, :uw] = uf
        canvas[bar:, uw:] = pf
        cv2.line(canvas, (uw, 0), (uw, H), (255, 255, 255), 2)
        writer.write(canvas)
        if i in anchor_by_user:
            stills[anchor_by_user[i]] = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        if progress_cb and i % 5 == 0:
            progress_cb(min(1.0, (i + 1) / user_data.n))
    capu.release()
    capp.release()
    writer.release()
    path, playable = _finalize(raw, out_path)
    return path, stills, playable, anchors
