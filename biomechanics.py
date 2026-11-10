"""
biomechanics.py
===============
Pure NumPy biomechanics for the Track & Field AI Coach (no MediaPipe/OpenCV needed).

Pipeline
--------
raw BlazePose keypoints (N,33,3 = x_px, y_px, visibility)
    -> build_kinematics()   : gap-fill, orient the athlete to run left->right, flip y-axis to "up",
                              smooth, estimate scale (px per metre), ground level and foot contacts
    -> analyze_<event>()    : event-specific metrics + key frames + overlay annotations
    -> coaching_engine      : compares metrics to elite ranges and writes advice

World coordinate convention used everywhere after build_kinematics():
    +x = direction of travel,  +y = UP,  units = pixels (use Kinematics.to_cm() for centimetres).

IMPORTANT ASSUMPTIONS (state them to your athletes):
    * Static camera, side-on view (perpendicular to the runway / bar), athlete fully in frame.
    * Ground contacts are inferred from foot height + foot speed, so >= 60 fps video is strongly recommended.
    * Everything is 2-D, so all angles are *projected* angles. Frontal-plane measures (high-jump lean,
      hurdle trail-leg abduction) are best filmed from the front/rear; see CAMERA_HINTS.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")

NAN = float("nan")

# --------------------------------------------------------------------------------------
# BlazePose landmark indices (subset)
# --------------------------------------------------------------------------------------
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32  # foot index = toe

SIDE: Dict[str, Dict[str, int]] = {
    "L": dict(shoulder=L_SHOULDER, elbow=L_ELBOW, wrist=L_WRIST, hip=L_HIP, knee=L_KNEE,
              ankle=L_ANKLE, heel=L_HEEL, toe=L_FOOT),
    "R": dict(shoulder=R_SHOULDER, elbow=R_ELBOW, wrist=R_WRIST, hip=R_HIP, knee=R_KNEE,
              ankle=R_ANKLE, heel=R_HEEL, toe=R_FOOT),
}


def other(side: str) -> str:
    return "R" if side == "L" else "L"


CAMERA_HINTS: Dict[str, str] = {
    "Long Jump": "Side-on camera, static, 60+ fps, framing the last ~5 strides, board and landing.",
    "High Jump": "Side-on (parallel to the bar) for arch metrics. Takeoff lean is best seen from a "
                 "front/rear camera; side-on measures the lean in the image plane only.",
    "Triple Jump": "Side-on, static, wide enough to include board to pit (or film each phase "
                   "separately). 60+ fps.",
    "Pole Vault": "Side-on, static, framing the last 3 strides through bar clearance. 60+ fps.",
    "Hurdles": "Side-on, static, 60+ fps, 1-3 hurdles in frame. Trail-leg abduction is "
               "best captured from a front/rear camera; side-on gives a projected angle.",
}


# --------------------------------------------------------------------------------------
# Generic math helpers
# --------------------------------------------------------------------------------------
def calculate_angle(a, b, c) -> float:
    """Angle at vertex ``b`` (degrees, 0-180) formed by points a-b-c. Accepts (x, y[, ...]) points."""
    a = np.asarray(a, dtype=float)[:2]
    b = np.asarray(b, dtype=float)[:2]
    c = np.asarray(c, dtype=float)[:2]
    ba, bc = a - b, c - b
    na, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    if not np.isfinite(na * nc) or na < 1e-9 or nc < 1e-9:
        return NAN
    cosang = np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def angle_series(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Vectorised calculate_angle for (N,2) arrays -> (N,) degrees."""
    ba, bc = a - b, c - b
    num = np.einsum("ij,ij->i", ba, bc)
    den = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosang = np.clip(num / den, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def angle_from_up(vx, vy):
    """Degrees between a vector and straight UP; positive when the vector leans toward +x."""
    return np.degrees(np.arctan2(vx, vy))


def angle_from_down(vx, vy):
    """Degrees between a vector and straight DOWN; positive when the vector points toward +x
    (0 = hanging down, 90 = horizontal forward, 180 = straight up)."""
    return np.degrees(np.arctan2(vx, -vy))


def interpolate_nans(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float).copy()
    bad = ~np.isfinite(y)
    if not bad.any() or bad.all():
        return y
    idx = np.arange(len(y))
    y[bad] = np.interp(idx[bad], idx[~bad], y[~bad])
    return y


def moving_average(a: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average along axis 0 with edge padding (works on (N,) or (N,K))."""
    a = np.asarray(a, dtype=float)
    if window <= 1 or len(a) < window:
        return a.copy()
    pad = window // 2
    ap = np.pad(a, [(pad, window - 1 - pad)] + [(0, 0)] * (a.ndim - 1), mode="edge")
    c = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(ap, axis=0)], axis=0)
    return (c[window:] - c[:-window]) / window


def runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Inclusive (start, end) index pairs of consecutive True values."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def clean_mask(mask: np.ndarray, min_len: int = 2, max_gap: int = 1) -> np.ndarray:
    """Fill tiny False gaps, then drop True runs shorter than ``min_len``."""
    m = np.asarray(mask, dtype=bool).copy()
    for s, e in runs(~m):
        if s > 0 and e < len(m) - 1 and (e - s + 1) <= max_gap:
            m[s:e + 1] = True
    for s, e in runs(m):
        if (e - s + 1) < min_len:
            m[s:e + 1] = False
    return m


def find_peaks(y: np.ndarray, min_distance: int, min_prominence: float) -> List[int]:
    """Tiny dependency-free peak finder (local maxima, spacing + prominence filtered)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    cand = [i for i in range(1, n - 1)
            if np.isfinite(y[i]) and y[i] >= y[i - 1] and y[i] > y[i + 1]]
    cand.sort(key=lambda i: -y[i])
    chosen: List[int] = []
    for i in cand:
        if any(abs(i - j) < min_distance for j in chosen):
            continue
        lo, hi = max(0, i - min_distance), min(n, i + min_distance + 1)
        left, right = np.nanmin(y[lo:i + 1]), np.nanmin(y[i:hi])
        if y[i] - max(left, right) >= min_prominence:
            chosen.append(i)
    return sorted(chosen)


def _nanargmax(a: np.ndarray) -> int:
    a = np.asarray(a, dtype=float)
    return int(np.nanargmax(a)) if np.isfinite(a).any() else 0


def _finite(*vals) -> bool:
    return all(v is not None and np.isfinite(v) for v in vals)


# --------------------------------------------------------------------------------------
# Kinematics container
# --------------------------------------------------------------------------------------
@dataclass
class Kinematics:
    fps: float
    t: np.ndarray                 # (N,) seconds
    xy: np.ndarray                # (N,33,2) world px, +x forward, +y up, smoothed + gap-filled
    vis: np.ndarray               # (N,33) raw BlazePose visibility
    height_px: float              # estimated stature in px
    leg_len_px: float
    athlete_height_m: float
    direction: int                # +1 = athlete moved left->right in the image, -1 = right->left
    ground_y: float
    valid_ratio: float            # fraction of frames with a usable pose
    contacts: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.t)

    @property
    def px_per_m(self) -> float:
        return self.height_px / self.athlete_height_m

    def to_cm(self, px):
        return np.asarray(px, dtype=float) / self.px_per_m * 100.0

    def pt(self, idx: int) -> np.ndarray:
        return self.xy[:, idx, :]

    def mid(self, a: int, b: int) -> np.ndarray:
        return 0.5 * (self.xy[:, a, :] + self.xy[:, b, :])

    def com(self) -> np.ndarray:
        """Whole-body centre-of-mass estimate (Dempster-style segment weights), shape (N,2)."""
        head = self.pt(NOSE)
        trunk = 0.5 * (self.mid(L_SHOULDER, R_SHOULDER) + self.mid(L_HIP, R_HIP))
        arms = 0.25 * (self.pt(L_SHOULDER) + self.pt(L_WRIST) + self.pt(R_SHOULDER) + self.pt(R_WRIST))
        thighs = 0.25 * (self.pt(L_HIP) + self.pt(L_KNEE) + self.pt(R_HIP) + self.pt(R_KNEE))
        shanks = 0.25 * (self.pt(L_KNEE) + self.pt(L_ANKLE) + self.pt(R_KNEE) + self.pt(R_ANKLE))
        return 0.081 * head + 0.497 * trunk + 0.100 * arms + 0.200 * thighs + 0.122 * shanks


def build_kinematics(keypoints: np.ndarray, fps: float, frame_size: Tuple[int, int],
                     athlete_height_m: float = 1.75, direction: int = 0,
                     min_vis: float = 0.3) -> Kinematics:
    """
    keypoints  : (N,33,3) array of [x_px, y_px, visibility]; NaN rows = no detection.
    frame_size : (width, height) of the frame the keypoints live in.
    direction  : 0 = auto-detect, +1 = athlete moves left->right, -1 = right->left (in the image).
    """
    kp = np.asarray(keypoints, dtype=float)
    n = kp.shape[0]
    width, height = frame_size
    vis = np.nan_to_num(kp[:, :, 2], nan=0.0)
    xy = kp[:, :, :2].copy()
    xy[(vis < min_vis) | ~np.isfinite(xy).all(axis=2)] = np.nan

    hips_ok = np.isfinite(xy[:, [L_HIP, R_HIP], 0]).all(axis=1)
    valid_ratio = float(hips_ok.mean()) if n else 0.0

    for j in range(33):
        for c in (0, 1):
            xy[:, j, c] = interpolate_nans(xy[:, j, c])

    # direction of travel
    if direction == 0:
        hip_x = np.nanmean(xy[:, [L_HIP, R_HIP], 0], axis=1)
        ok = np.isfinite(hip_x)
        slope = np.polyfit(np.arange(n)[ok], hip_x[ok], 1)[0] if ok.sum() >= 3 else 0.0
        direction = 1 if slope >= 0 else -1
    if direction == -1:
        xy[:, :, 0] = width - xy[:, :, 0]
    xy[:, :, 1] = height - xy[:, :, 1]  # y up

    win = max(3, int(round(fps / 15.0)))
    win += (win % 2 == 0)
    xy = moving_average(xy.reshape(n, -1), win).reshape(n, 33, 2)

    def seg(a, b):
        return np.linalg.norm(xy[:, a] - xy[:, b], axis=1)

    torso = np.nanmedian(np.linalg.norm(0.5 * (xy[:, L_SHOULDER] + xy[:, R_SHOULDER])
                                        - 0.5 * (xy[:, L_HIP] + xy[:, R_HIP]), axis=1))
    thigh = np.nanmedian(np.r_[seg(L_HIP, L_KNEE), seg(R_HIP, R_KNEE)])
    shank = np.nanmedian(np.r_[seg(L_KNEE, L_ANKLE), seg(R_KNEE, R_ANKLE)])
    height_px = (torso + thigh + shank) / 0.78  # shoulder->ankle length ~ 0.78 stature
    if not np.isfinite(height_px) or height_px <= 1:
        height_px = 0.5 * height
    leg_len = thigh + shank if np.isfinite(thigh + shank) else 0.5 * height_px

    low = np.minimum(np.minimum(xy[:, L_HEEL, 1], xy[:, L_FOOT, 1]),
                     np.minimum(xy[:, R_HEEL, 1], xy[:, R_FOOT, 1]))
    ground_y = float(np.nanpercentile(low, 3)) if np.isfinite(low).any() else 0.0

    kin = Kinematics(fps=float(fps), t=np.arange(n) / float(fps), xy=xy, vis=vis,
                     height_px=float(height_px), leg_len_px=float(leg_len),
                     athlete_height_m=float(athlete_height_m), direction=int(direction),
                     ground_y=ground_y, valid_ratio=valid_ratio)
    kin.contacts = compute_contacts(kin)
    return kin


def compute_contacts(kin: Kinematics, tol_frac: float = 0.045, speed_frac: float = 1.2
                     ) -> Dict[str, np.ndarray]:
    """Boolean ground-contact mask per foot: near ground level AND slow (in stature/s)."""
    out = {}
    for side, ids in SIDE.items():
        low = np.minimum(kin.xy[:, ids["heel"], 1], kin.xy[:, ids["toe"], 1])
        near = (low - kin.ground_y) < tol_frac * kin.height_px
        pos = 0.5 * (kin.xy[:, ids["heel"], :] + kin.xy[:, ids["toe"], :])
        if kin.n >= 3:
            speed = np.linalg.norm(np.gradient(pos, kin.t, axis=0), axis=1) / kin.height_px
        else:
            speed = np.zeros(kin.n)
        out[side] = clean_mask(near & (speed < speed_frac), min_len=2, max_gap=1)
    return out


def find_flights(kin: Kinematics, min_duration_s: float) -> List[Tuple[int, int]]:
    """Airborne phases (no foot in contact). Returned as inclusive (first_air, last_air) frames."""
    c = kin.contacts["L"] | kin.contacts["R"]
    min_len = max(2, int(round(min_duration_s * kin.fps)))
    return [(s, e) for s, e in runs(~c) if (e - s + 1) >= min_len and s > 0]


def _side_at(kin: Kinematics, frame: int) -> str:
    """Which foot is on the ground at `frame` (falls back to the lower ankle)."""
    frame = int(np.clip(frame, 0, kin.n - 1))
    c = [s for s in ("L", "R") if kin.contacts[s][frame]]
    if len(c) == 1:
        return c[0]
    cand = c or ["L", "R"]
    return min(cand, key=lambda s: kin.xy[frame, SIDE[s]["ankle"], 1])


def best_visible_side(kin: Kinematics) -> str:
    score = {s: float(np.nanmean(kin.vis[:, [i["shoulder"], i["hip"], i["knee"], i["ankle"]]]))
             for s, i in SIDE.items()}
    return max(score, key=score.get)


def _contact_segments(kin: Kinematics) -> List[Tuple[int, int, str]]:
    segs = [(s, e, side) for side in ("L", "R") for s, e in runs(kin.contacts[side])]
    return sorted(segs)


# --------------------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------------------
@dataclass
class AnalysisResult:
    event: str
    t: np.ndarray
    metrics: Dict[str, float] = field(default_factory=dict)
    key_frames: Dict[str, int] = field(default_factory=dict)
    annotations: Dict[int, List[str]] = field(default_factory=dict)          # frame -> text lines
    angle_marks: Dict[int, List[Tuple[int, int, int]]] = field(default_factory=dict)  # frame -> (a,b,c)
    series: Dict[str, np.ndarray] = field(default_factory=dict)             # name -> (N,) values
    curves: Dict[str, np.ndarray] = field(default_factory=dict)             # name -> short normalised curve
    warnings: List[str] = field(default_factory=list)

    def note(self, frame: int, text: str) -> None:
        self.annotations.setdefault(int(frame), []).append(text)

    def mark_angle(self, frame: int, a: int, b: int, c: int) -> None:
        self.angle_marks.setdefault(int(frame), []).append((a, b, c))


def _init(event: str, kin: Kinematics, keys: Sequence[str]) -> AnalysisResult:
    r = AnalysisResult(event, kin.t)
    for k in keys:
        r.metrics[k] = NAN
    if kin.valid_ratio < 0.6:
        r.warnings.append(f"Pose was detected in only {kin.valid_ratio:.0%} of frames - results may be unreliable "
                          "(check lighting, framing and that the full body is visible).")
    return r


# --------------------------------------------------------------------------------------
# LONG JUMP
# --------------------------------------------------------------------------------------
def analyze_long_jump(kin: Kinematics) -> AnalysisResult:
    """Takeoff angle, penultimate-step COM drop, landing extension."""
    r = _init("Long Jump", kin, ["takeoff_angle_deg", "penultimate_com_drop_pct", "penultimate_com_drop_cm",
                                 "knee_drive_deg", "landing_reach_ratio", "landing_knee_angle_deg"])
    m, N = r.metrics, kin.n
    flights = find_flights(kin, 0.20)
    if not flights:
        r.warnings.append("No airborne phase found. Use a static side-on clip that includes takeoff and landing.")
        return r
    s, e = max(flights, key=lambda f: f[1] - f[0])
    take, land = s - 1, min(e + 1, N - 1)
    com = kin.com()
    hip_y = kin.mid(L_HIP, R_HIP)[:, 1]
    r.series["COM height (cm)"] = kin.to_cm(com[:, 1] - kin.ground_y)
    r.key_frames["Takeoff"] = take

    # --- takeoff angle: direction of COM velocity just after leaving the board ---
    w = max(3, int(round(0.10 * kin.fps)))
    idx = np.arange(take, min(take + w, land) + 1)
    if len(idx) >= 3 and np.isfinite(com[idx]).all():
        vx = np.polyfit(kin.t[idx], com[idx, 0], 1)[0]
        vy = np.polyfit(kin.t[idx], com[idx, 1], 1)[0]
        if vx > 0:
            m["takeoff_angle_deg"] = float(np.degrees(np.arctan2(vy, vx)))
            r.note(take, f"Takeoff angle {m['takeoff_angle_deg']:.0f} deg")
        else:
            r.warnings.append("Could not resolve forward velocity at takeoff (is the camera panning?).")

    apex = s + _nanargmax(com[s:e + 1, 1])
    r.key_frames["Flight apex"] = apex

    # --- free-knee drive: hip flexion (shoulder-hip-knee) of the free leg, best frame around takeoff ---
    fi = SIDE[other(_side_at(kin, take))]
    win = np.arange(max(0, take - 1), min(N - 1, take + 2) + 1)
    kd = angle_series(kin.pt(fi["shoulder"])[win], kin.pt(fi["hip"])[win], kin.pt(fi["knee"])[win])
    if np.isfinite(kd).any():
        j = int(win[_nanargmax(-kd)])
        m["knee_drive_deg"] = float(np.nanmin(kd))
        r.mark_angle(j, fi["shoulder"], fi["hip"], fi["knee"])
        r.note(j, f"Knee drive {m['knee_drive_deg']:.0f} deg")

    # --- penultimate-step COM drop ---
    segs = _contact_segments(kin)
    if segs:
        ti = int(np.argmin([abs(sg[1] - take) for sg in segs]))
        if ti >= 1:
            ps, pe, _ = segs[ti - 1]
            ref_frames = np.concatenate([np.arange(a, b + 1) for a, b, _ in segs[max(0, ti - 4):ti - 1]]) \
                if ti >= 2 else np.array([], dtype=int)
            if len(ref_frames) >= 2:
                ref = float(np.nanmedian(hip_y[ref_frames]))
                pm = ps + _nanargmax(-hip_y[ps:pe + 1])
                drop_px = ref - float(hip_y[pm])
                m["penultimate_com_drop_cm"] = float(kin.to_cm(drop_px))
                m["penultimate_com_drop_pct"] = float(drop_px / kin.height_px * 100.0)
                r.key_frames["Penultimate step"] = pm
                r.note(pm, f"Penultimate COM drop {m['penultimate_com_drop_cm']:.1f} cm")
            else:
                r.warnings.append("Not enough run-up contacts detected before the penultimate step.")
        else:
            r.warnings.append("Penultimate step not found - include more of the run-up in frame.")

    # --- landing extension (use last frames of flight) ---
    cand = range(max(s, e - 2), min(N - 1, e + 1) + 1)
    best = (-np.inf, None, None)
    hip_mid = kin.mid(L_HIP, R_HIP)
    for f in cand:
        for side in ("L", "R"):
            reach = (kin.xy[f, SIDE[side]["heel"], 0] - hip_mid[f, 0]) / kin.leg_len_px
            if np.isfinite(reach) and reach > best[0]:
                best = (reach, f, side)
    if best[1] is not None:
        reach, f, side = best
        ids = SIDE[side]
        m["landing_reach_ratio"] = float(reach)
        m["landing_knee_angle_deg"] = calculate_angle(kin.pt(ids["hip"])[f], kin.pt(ids["knee"])[f],
                                                      kin.pt(ids["ankle"])[f])
        r.key_frames["Landing"] = f
        r.mark_angle(f, ids["hip"], ids["knee"], ids["ankle"])
        r.note(f, f"Landing reach {reach:.2f}x leg, knee {m['landing_knee_angle_deg']:.0f} deg")
    return r


# --------------------------------------------------------------------------------------
# HIGH JUMP (Fosbury Flop)
# --------------------------------------------------------------------------------------
def analyze_high_jump(kin: Kinematics) -> AnalysisResult:
    """Takeoff lean and flight arch (shoulder-hip-knee) with premature-arch timing checks."""
    r = _init("High Jump", kin, ["takeoff_lean_deg", "peak_arch_angle_deg", "arch_onset_frac",
                                 "arch_peak_vs_apex_s", "hip_rise_cm"])
    m, N = r.metrics, kin.n
    flights = find_flights(kin, 0.25)
    if not flights:
        r.warnings.append("No airborne phase found. Use a static clip that includes the takeoff and bar clearance.")
        return r
    s, e = max(flights, key=lambda f: f[1] - f[0])
    take = s - 1
    side = best_visible_side(kin)
    ids = SIDE[side]
    hip = kin.mid(L_HIP, R_HIP)
    hy = hip[:, 1]
    apex = s + _nanargmax(hy[s:e + 1])
    r.key_frames.update({"Takeoff": take, "Hip apex": apex})

    # takeoff lean: whole-body line from the takeoff ankle to mid-shoulders, measured from vertical
    tside = _side_at(kin, take)
    vec = kin.mid(L_SHOULDER, R_SHOULDER)[take] - kin.pt(SIDE[tside]["ankle"])[take]
    if np.isfinite(vec).all():
        m["takeoff_lean_deg"] = float(abs(angle_from_up(vec[0], vec[1])))
        r.note(take, f"Takeoff lean {m['takeoff_lean_deg']:.0f} deg")

    # flight arch
    arch = angle_series(kin.pt(ids["shoulder"]), kin.pt(ids["hip"]), kin.pt(ids["knee"]))
    r.series["Arch angle S-H-K (deg)"] = arch
    r.series["Hip height (cm)"] = kin.to_cm(hy - kin.ground_y)
    seg = arch[s:e + 1]
    if np.isfinite(seg).any():
        pk = s + _nanargmax(-seg)
        m["peak_arch_angle_deg"] = float(arch[pk])
        m["arch_peak_vs_apex_s"] = float(kin.t[pk] - kin.t[apex])
        r.key_frames["Peak arch"] = pk
        r.mark_angle(pk, ids["shoulder"], ids["hip"], ids["knee"])
        r.note(pk, f"Peak arch {arch[pk]:.0f} deg")
        rise_window = np.arange(s, max(apex, s) + 1)
        below = [f for f in rise_window if np.isfinite(arch[f]) and arch[f] < 160.0]
        m["arch_onset_frac"] = float((below[0] - s) / max(1, apex - s)) if below else 1.0
    m["hip_rise_cm"] = float(kin.to_cm(hy[apex] - hy[take]))
    return r


# --------------------------------------------------------------------------------------
# TRIPLE JUMP
# --------------------------------------------------------------------------------------
def _foot_mid_x(kin: Kinematics, side: str, f: int) -> float:
    ids = SIDE[side]
    return float(0.5 * (kin.xy[f, ids["heel"], 0] + kin.xy[f, ids["toe"], 0]))


def analyze_triple_jump(kin: Kinematics) -> AnalysisResult:
    """Hop/step/jump distance ratios and active-landing geometry of the hop and step landings."""
    keys = ["hop_pct", "step_pct", "jump_pct", "total_distance_m"]
    for ph in ("hop", "step"):
        keys += [f"{ph}_landing_foot_angle_deg", f"{ph}_landing_leg_angle_deg"]
    r = _init("Triple Jump", kin, keys)
    m, N = r.metrics, kin.n
    flights = find_flights(kin, 0.15)
    if len(flights) < 3:
        r.warnings.append(f"Found {len(flights)} airborne phase(s); a triple jump needs 3 (hop, step, jump). "
                          "Use a static, wide side-on clip (board to pit) at 60+ fps.")
        return r
    top = sorted(sorted(flights, key=lambda f: -(f[1] - f[0]))[:3])
    phases = [(s - 1, min(e + 1, N - 1)) for s, e in top]
    hip_y = kin.com()[:, 1]
    r.series["COM height (cm)"] = kin.to_cm(hip_y - kin.ground_y)

    t0 = phases[0][0]
    marks = [float(kin.xy[t0, SIDE[_side_at(kin, t0)]["toe"], 0])]  # toe on the board
    for k, (_, lf) in enumerate(phases):
        if k < 2:
            marks.append(_foot_mid_x(kin, _side_at(kin, lf), lf))
        else:
            marks.append(float(max(kin.xy[lf, L_HEEL, 0], kin.xy[lf, R_HEEL, 0])))  # pit mark
    d = np.diff(marks)
    labels = ["Hop", "Step", "Jump"]
    for k, (tf, lf) in enumerate(phases):
        r.key_frames[f"{labels[k]} takeoff"] = tf
    r.key_frames["Final landing"] = phases[2][1]

    if np.all(d > 0) and np.isfinite(d).all():
        total = float(d.sum())
        m["hop_pct"], m["step_pct"], m["jump_pct"] = [float(x / total * 100.0) for x in d]
        m["total_distance_m"] = float(total / kin.px_per_m)
        for k, lab in enumerate(labels):
            r.note(phases[k][0], f"{lab} {m[lab.lower() + '_pct']:.0f}%")
    else:
        r.warnings.append("Phase distances were not monotonic - contact detection probably failed "
                          "(camera panning, occlusion or low fps).")

    # active landing: foot attitude and leg angle at touchdown of hop (k=0) and step (k=1)
    for k, ph in enumerate(("hop", "step")):
        lf = phases[k][1]
        side = _side_at(kin, lf)
        ids = SIDE[side]
        fr = [f for f in (lf, min(lf + 1, N - 1))]
        foot = np.nanmean([np.degrees(np.arctan2(
            kin.xy[f, ids["toe"], 1] - kin.xy[f, ids["heel"], 1],
            kin.xy[f, ids["toe"], 0] - kin.xy[f, ids["heel"], 0])) for f in fr])
        leg = np.nanmean([angle_from_down(kin.xy[f, ids["ankle"], 0] - kin.xy[f, ids["hip"], 0],
                                          kin.xy[f, ids["ankle"], 1] - kin.xy[f, ids["hip"], 1]) for f in fr])
        m[f"{ph}_landing_foot_angle_deg"] = float(foot)
        m[f"{ph}_landing_leg_angle_deg"] = float(leg)
        r.key_frames[f"{ph.capitalize()} landing"] = lf
        r.mark_angle(lf, ids["hip"], ids["knee"], ids["ankle"])
        r.note(lf, f"{ph.capitalize()} landing: foot {foot:+.0f} deg, leg {leg:+.0f} deg")
    return r


# --------------------------------------------------------------------------------------
# POLE VAULT
# --------------------------------------------------------------------------------------
def analyze_pole_vault(kin: Kinematics) -> AnalysisResult:
    """Plant angle (arms vs torso at takeoff), bottom-arm collapse, inversion angle and swing speed."""
    r = _init("Pole Vault", kin, ["plant_angle_deg", "bottom_arm_min_elbow_deg", "peak_inversion_deg",
                                  "peak_swing_velocity_dps", "time_to_peak_inversion_s"])
    m, N = r.metrics, kin.n
    flights = find_flights(kin, 0.50)
    if not flights:
        r.warnings.append("No long airborne phase found. Include the takeoff and swing in a static side-on clip.")
        return r
    s, e = max(flights, key=lambda f: f[1] - f[0])
    take = s - 1
    r.key_frames["Takeoff / plant"] = take

    # top vs bottom arm at takeoff (higher wrist = top hand)
    top = "L" if kin.xy[take, L_WRIST, 1] >= kin.xy[take, R_WRIST, 1] else "R"
    bot = other(top)
    ti, bi = SIDE[top], SIDE[bot]
    m["plant_angle_deg"] = calculate_angle(kin.pt(ti["hip"])[take], kin.pt(ti["shoulder"])[take],
                                           kin.pt(ti["wrist"])[take])
    r.mark_angle(take, ti["hip"], ti["shoulder"], ti["wrist"])
    r.note(take, f"Plant (top arm vs torso) {m['plant_angle_deg']:.0f} deg")

    elbow = angle_series(kin.pt(bi["shoulder"]), kin.pt(bi["elbow"]), kin.pt(bi["wrist"]))
    w_end = min(N - 1, take + max(3, int(round(0.30 * kin.fps))))
    if np.isfinite(elbow[take:w_end + 1]).any():
        j = take + _nanargmax(-elbow[take:w_end + 1])
        m["bottom_arm_min_elbow_deg"] = float(elbow[j])
        r.mark_angle(j, bi["shoulder"], bi["elbow"], bi["wrist"])
        r.note(j, f"Bottom-arm elbow {elbow[j]:.0f} deg")

    # inversion: hips relative to shoulders. 0 = hips hanging below, 90 = horizontal, 180 = hips overhead
    vec = kin.mid(L_HIP, R_HIP) - kin.mid(L_SHOULDER, R_SHOULDER)
    seg_end = min(e, N - 1)
    ang = angle_from_down(vec[take:seg_end + 1, 0], vec[take:seg_end + 1, 1])
    ang = np.degrees(np.unwrap(np.radians(interpolate_nans(ang))))
    full = np.full(N, np.nan)
    full[take:seg_end + 1] = ang
    r.series["Inversion angle (deg)"] = full
    if len(ang) >= 4 and np.isfinite(ang).all():
        pi_rel = int(np.argmax(ang))
        pi = take + pi_rel
        m["peak_inversion_deg"] = float(ang[pi_rel])
        m["time_to_peak_inversion_s"] = float(kin.t[pi] - kin.t[take])
        av = moving_average(np.gradient(ang, kin.t[take:seg_end + 1]), 3)
        m["peak_swing_velocity_dps"] = float(np.max(av[:pi_rel + 1]))
        r.key_frames["Peak inversion"] = pi
        r.note(pi, f"Inversion {m['peak_inversion_deg']:.0f} deg, swing {m['peak_swing_velocity_dps']:.0f} deg/s")
    return r


# --------------------------------------------------------------------------------------
# HURDLES
# --------------------------------------------------------------------------------------
def analyze_hurdles(kin: Kinematics) -> AnalysisResult:
    """Lead-leg extension, trail-leg abduction/flexion, hip rise over the hurdle, trunk lean."""
    r = _init("Hurdles", kin, ["lead_leg_extension_deg", "trail_leg_abduction_deg", "trail_knee_flex_deg",
                               "hip_rise_pct", "hip_rise_cm", "trunk_lean_deg", "n_hurdles"])
    m, N, fps = r.metrics, kin.n, kin.fps
    hip = kin.mid(L_HIP, R_HIP)
    hy = moving_average(hip[:, 1], 3)
    r.series["Hip height (cm)"] = kin.to_cm(hy - kin.ground_y)
    peaks = find_peaks(hy, max(2, int(0.45 * fps)), 0.035 * kin.height_px)
    m["n_hurdles"] = float(len(peaks))
    if not peaks:
        r.warnings.append("No hurdle clearances detected (no clear hip-height peaks). "
                          "Use a static side-on clip with at least one full hurdle.")
        return r

    hw = max(2, int(round(0.30 * fps)))
    mask = np.ones(N, dtype=bool)
    for p in peaks:
        mask[max(0, p - hw):p + hw + 1] = False
    baseline = float(np.nanmedian(hy[mask])) if mask.sum() >= 5 else float(np.nanpercentile(hy, 25))

    knee_ang = {s: angle_series(kin.pt(i["hip"]), kin.pt(i["knee"]), kin.pt(i["ankle"])) for s, i in SIDE.items()}
    hip_ang = {s: angle_series(kin.pt(i["shoulder"]), kin.pt(i["hip"]), kin.pt(i["knee"])) for s, i in SIDE.items()}
    sh_mid = kin.mid(L_SHOULDER, R_SHOULDER)

    lead_ext, trail_abd, trail_flex, rises, leans = [], [], [], [], []
    hw2 = max(1, int(round(0.10 * fps)))
    for k, p in enumerate(peaks, start=1):
        a, b = max(0, p - hw), min(N - 1, p + hw)
        ank_max = {s: np.nanmax(kin.xy[a:b + 1, SIDE[s]["ankle"], 1]) for s in ("L", "R")}
        lead = "L" if ank_max["L"] >= ank_max["R"] else "R"
        trail = other(lead)
        fa = a + _nanargmax(kin.xy[a:b + 1, SIDE[lead]["ankle"], 1])
        la, lb = max(0, fa - hw2), min(N - 1, fa + hw2)
        ext = np.nanmax(knee_ang[lead][la:lb + 1])
        lead_ext.append(ext)

        tb = min(N - 1, fa + max(2, int(round(0.40 * fps))))
        rel = kin.xy[fa:tb + 1, SIDE[trail]["knee"], 1] - hip[fa:tb + 1, 1]
        ft = fa + _nanargmax(rel)
        trail_abd.append(hip_ang[trail][ft])
        trail_flex.append(knee_ang[trail][ft])

        vec = sh_mid[p] - hip[p]
        leans.append(float(angle_from_up(vec[0], vec[1])))
        rises.append(hy[p] - baseline)

        r.key_frames[f"Hurdle {k} hip apex"] = p
        r.mark_angle(fa, SIDE[lead]["hip"], SIDE[lead]["knee"], SIDE[lead]["ankle"])
        r.mark_angle(ft, SIDE[trail]["shoulder"], SIDE[trail]["hip"], SIDE[trail]["knee"])
        r.note(fa, f"Hurdle {k} lead leg {ext:.0f} deg")
        r.note(ft, f"Hurdle {k} trail leg {hip_ang[trail][ft]:.0f} deg")
        r.note(p, f"Hurdle {k} hip rise {float(kin.to_cm(hy[p] - baseline)):.0f} cm")

    # COM (hip) trajectory over each hurdle: +/-0.5 s around the apex, as % stature above running height
    CURVE_PTS, half = 25, int(round(0.5 * fps))
    curves = []
    for p in peaks:
        if p - half >= 0 and p + half <= N - 1 and half >= 2:
            xs = np.linspace(p - half, p + half, CURVE_PTS)
            curves.append((np.interp(xs, np.arange(N), hy) - baseline) / kin.height_px * 100.0)
    if curves:
        r.curves["com_trajectory"] = np.nanmean(np.vstack(curves), axis=0)
    else:
        r.warnings.append("Clip too short around the hurdle to extract the +/-0.5 s COM trajectory.")

    m["lead_leg_extension_deg"] = float(np.nanmean(lead_ext))
    m["trail_leg_abduction_deg"] = float(np.nanmean(trail_abd))
    m["trail_knee_flex_deg"] = float(np.nanmean(trail_flex))
    m["hip_rise_cm"] = float(kin.to_cm(np.nanmean(rises)))
    m["hip_rise_pct"] = float(np.nanmean(rises) / kin.height_px * 100.0)
    m["trunk_lean_deg"] = float(np.nanmean(leans))
    return r


ANALYZERS: Dict[str, Callable[[Kinematics], AnalysisResult]] = {
    "Long Jump": analyze_long_jump,
    "High Jump": analyze_high_jump,
    "Triple Jump": analyze_triple_jump,
    "Pole Vault": analyze_pole_vault,
    "Hurdles": analyze_hurdles,
}


def analyze_event(event: str, kin: Kinematics) -> AnalysisResult:
    if event not in ANALYZERS:
        raise ValueError(f"Unknown event '{event}'. Choose from {list(ANALYZERS)}")
    return ANALYZERS[event](kin)
