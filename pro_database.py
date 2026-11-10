"""
pro_database.py
===============
Reference ("pro") biomechanics used by the coaching engine.

Each entry stores
  * ideal metric values + tolerance windows (value +/- tol = "matches the pro"),
  * the diagnosis text and corrective drill for being BELOW or ABOVE the pro value,
  * keyframe data (reference joint angles at crucial phases, shown for context),
  * an optional reference VIDEO filename (looked up in ./pro_videos/).

HONEST NOTE ON THE DATA
  The numbers below are *approximate reference values* compiled from commonly published
  biomechanics ranges for elite 2-D side-view analysis. They are a sensible starting point,
  NOT measurements of a named athlete. For best results:
    1. drop licensed/own footage of the athletes you want to compare against into ./pro_videos/
       (see PRO_VIDEO_HINT below) - the app then measures that clip with the same pipeline, and
    2. tune any value here, or override it without touching code via ./pro_database.json
       (same structure: {"Entry name": {"metrics": {"takeoff_angle_deg": {"value": 23}}}}).
  No copyrighted footage is bundled with this project.
"""
from __future__ import annotations

import copy
import json
import math
import os
from typing import Dict, List, Optional

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRO_VIDEO_DIR = os.environ.get("PRO_VIDEO_DIR", os.path.join(BASE_DIR, "pro_videos"))
OVERRIDES_PATH = os.path.join(BASE_DIR, "pro_database.json")
VIDEO_EXTS = (".mp4", ".mov", ".m4v", ".avi", ".webm")
PRO_VIDEO_HINT = ("Place a side-on, static-camera clip of the pro in the 'pro_videos' folder and name it exactly as "
                  "the entry's 'reference_video' field (e.g. elite_long_jump_takeoff.mp4).")


def _hurdle_curve(n: int = 25, peak: float = 7.0, dip: float = 1.0) -> List[float]:
    """Reference COM (hip) path over a hurdle: % stature above running height, +/-0.5 s around the apex."""
    xs = np.linspace(0, 1, n)
    y = peak * np.exp(-((xs - 0.5) / 0.17) ** 2) - dip * np.exp(-((xs - 0.30) / 0.08) ** 2)
    return [round(float(v), 3) for v in y]


def _m(label, unit, value, tol, scale, cue_low, cue_high, drill_low, drill_high, decimals=0):
    return dict(label=label, unit=unit, value=value, tol=tol, scale=scale, decimals=decimals,
                cue_low=cue_low, cue_high=cue_high, drill_low=drill_low, drill_high=drill_high)


PRO_LIBRARY: Dict[str, dict] = {
    # ------------------------------------------------------------------ LONG JUMP
    "Elite Long Jump Takeoff": dict(
        event="Long Jump", athlete_height_m=1.85, reference_video="elite_long_jump_takeoff.mp4",
        description="Last 3 steps, takeoff and flight of an elite long jumper.",
        keyframes={
            "Penultimate step": {"COM drop (% stature)": 4.2, "Trunk lean (deg from vertical)": 4},
            "Takeoff": {"Takeoff angle (deg)": 22, "Free-knee drive / hip flexion (deg)": 90,
                        "Takeoff-leg knee (deg)": 170},
            "Landing": {"Leg reach (x leg length)": 0.9, "Knee angle (deg)": 165},
        },
        metrics={
            "takeoff_angle_deg": _m(
                "Takeoff angle", "°", 22.0, 3.0, (8, 36),
                "Your takeoff is too flat: you are not converting enough run-up speed into height, so flight time (and distance) is lost.",
                "Your takeoff is too steep: you are jumping up rather than out and bleeding horizontal speed at the board.",
                "Short-approach (5-step) jumps over a 40 cm elastic band ~1 m past the board - drive the free knee to hip height and punch the arms up.",
                "Run-through takeoffs: 7-step approach, jump to a low target 1 m out and 20 cm up, chest tall and eyes forward (not up)."),
            "penultimate_com_drop_pct": _m(
                "Penultimate-step COM drop", "% stature", 4.2, 1.6, (-2, 10),
                "Your hips are not lowering on the penultimate step, so the last step cannot generate vertical lift.",
                "Your hips collapse on the penultimate step, costing speed before the board.",
                "Penultimate-lowering drill: 7-step approach with the penultimate mark ~10 cm longer, then a quick, tall final step.",
                "'Long-short' rhythm runs with cones on the last two marks - land the penultimate foot under the hips and stay tall."),
            "knee_drive_deg": _m(
                "Free-knee drive (hip flexion)", "°", 90.0, 10.0, (50, 150),
                "Your free knee is pulled up past horizontal, which can tip the trunk into forward rotation.",
                "Your free knee is not driving up to hip height at takeoff, so you lose vertical impulse.",
                "Controlled takeoff 'holds': 3-step takeoff, freeze with the thigh horizontal for 2 s on a mat.",
                "Wall knee-drive drill (hands on wall, drive the knee to 90 deg) plus A-skips emphasising a fast, high knee."),
        }),
    # ------------------------------------------------------------------ HIGH JUMP
    "Elite High Jump (Fosbury Flop) Clearance": dict(
        event="High Jump", athlete_height_m=1.95, reference_video="elite_high_jump_flop.mp4",
        description="Curved approach, takeoff and bar clearance of an elite Fosbury-flop jumper.",
        keyframes={
            "Takeoff": {"Takeoff lean (deg)": 12, "Takeoff-leg knee (deg)": 175},
            "Peak arch": {"Shoulder-hip-knee (deg)": 135},
        },
        metrics={
            "takeoff_lean_deg": _m(
                "Takeoff lean (inward tilt)", "°", 12.0, 4.0, (0, 35),
                "You are too upright at takeoff: curve speed is not being used, so you rise straight up and land on the bar.",
                "You are leaning too far at takeoff: the body will fall toward the bar instead of rotating over it.",
                "J-curve running at 75% speed leaning from the ankles, then 5-step curved takeoffs onto a mat.",
                "Straight 3-step takeoffs focusing on a tall posture, then gradually widen the curve radius."),
            "peak_arch_angle_deg": _m(
                "Flight arch (shoulder-hip-knee)", "°", 135.0, 12.0, (90, 180),
                "Your arch is too shallow: the hips will sag onto the bar and the bar comes down on your glutes.",
                "You are over-arching: it costs recovery time and makes it hard to kick the legs over the bar.",
                "Back-over drill: 3-step takeoff, drive head and shoulders back and hips up onto a high mat; add bridges and thoracic extensions.",
                "Controlled back-overs focusing on head-chest-hips sequence, then 'hips up, then kick the legs' timing."),
        }),
    # ------------------------------------------------------------------ TRIPLE JUMP
    "Elite Triple Jump - Balanced Phases": dict(
        event="Triple Jump", athlete_height_m=1.88, reference_video="elite_triple_jump.mp4",
        description="Balanced hop-step-jump (approx. 35 / 30 / 35).",
        keyframes={"Hop landing": {"Foot angle (deg)": 5, "Leg angle (deg)": 10},
                   "Step landing": {"Foot angle (deg)": 5, "Leg angle (deg)": 10}},
        metrics={
            "hop_pct": _m("Hop phase", "% of total", 35.0, 3.0, (20, 50),
                          "Your hop is short: you are too cautious off the board and hand speed to the later phases without using it.",
                          "Your hop is too long: it leaves a short step and jump because you cannot keep speed through a big hop.",
                          "Bounding series with a mark at the target hop distance; attack the board with a flat, fast takeoff.",
                          "Hop-to-mark drills: place a cone at the ideal hop length, land short of it, and keep the hop low and flat."),
            "step_pct": _m("Step phase", "% of total", 30.0, 3.0, (15, 45),
                           "Your step is collapsing: speed is lost after the hop landing and the athlete 'dies' in the step.",
                           "Your step is too long relative to the others, which usually shortens the jump.",
                           "Hop-step bounds over low hurdles (3 x 30 cm), active ground contact with the free knee driving up.",
                           "Step-to-mark drills with a cone at the target; focus on pushing the hips forward rather than reaching."),
            "jump_pct": _m("Jump phase", "% of total", 35.0, 3.0, (20, 50),
                           "Your jump phase is short: speed has run out by the final phase.",
                           "Your jump is long relative to the others (usually a short hop/step) - check the earlier phases first.",
                           "Short-approach full jumps (hop-step-jump from 9 strides) emphasising a fast active step landing.",
                           "Even-rhythm bounding (3 bounds each side) to build a balanced rhythm."),
        }),
    "Elite Triple Jump - Hop-Dominant": dict(
        event="Triple Jump", athlete_height_m=1.88, reference_video="elite_triple_jump_hop_dominant.mp4",
        description="Hop-dominant strategy (approx. 38 / 28 / 34).",
        keyframes={"Hop landing": {"Foot angle (deg)": 5, "Leg angle (deg)": 10}},
        metrics={
            "hop_pct": _m("Hop phase", "% of total", 38.0, 3.0, (20, 50),
                          "Your hop is shorter than a hop-dominant strategy needs: attack the board harder.",
                          "Your hop is longer than even a hop-dominant jumper's: the step will starve.",
                          "Bounding series with a mark at the target hop distance; attack the board with a flat, fast takeoff.",
                          "Hop-to-mark drills with a cone at the ideal distance; keep the hop low and flat."),
            "step_pct": _m("Step phase", "% of total", 28.0, 3.0, (15, 45),
                           "Your step is collapsing: speed is lost after the hop landing.",
                           "Your step is too long for this strategy, shortening the jump.",
                           "Hop-step bounds over low hurdles, active ground contact with the free knee driving up.",
                           "Step-to-mark drills with a cone at the target."),
            "jump_pct": _m("Jump phase", "% of total", 34.0, 3.0, (20, 50),
                           "Your jump phase is short: speed has run out by the final phase.",
                           "Your jump is long relative to the target - check the first two phases.",
                           "Short-approach full jumps emphasising an active step landing.",
                           "Even-rhythm bounding to build a balanced rhythm."),
        }),
    # ------------------------------------------------------------------ POLE VAULT
    "Elite Pole Vault Plant & Inversion": dict(
        event="Pole Vault", athlete_height_m=1.85, reference_video="elite_pole_vault.mp4",
        description="Plant, takeoff and swing/inversion of an elite vaulter.",
        keyframes={"Takeoff / plant": {"Top arm vs torso (deg)": 168, "Bottom-arm elbow (deg)": 170},
                   "Peak inversion": {"Hips-over-shoulders angle (deg)": 175}},
        metrics={
            "plant_angle_deg": _m(
                "Plant angle (top arm vs torso)", "°", 168.0, 8.0, (110, 190),
                "Your top arm is low at takeoff: you are not reaching tall, so the pole loads late and energy is lost.",
                "Your arm-torso angle is larger than the pro's range - check for a backward lean at takeoff.",
                "Pole carry + 3-step plants: drive the top hand overhead as the takeoff foot lands; use a rope-climb or hang-clean to learn the 'reach'.",
                "Plant drills on the box with a tall posture: keep the chest over the takeoff foot and shoulders stacked."),
            "peak_inversion_deg": _m(
                "Inversion angle (hips over shoulders)", "°", 175.0, 12.0, (90, 200),
                "Your hips are not getting above your shoulders: the swing is too slow or too early, so you stall below the pole.",
                "You are swinging past vertical, which usually means over-rotating and drifting off the pole.",
                "Rope swing-ups, rings inversions and 'swing and hold' on a short pole to learn the hips-up-and-over timing.",
                "Controlled swing-up with a stop at vertical on a short run-up; focus on pushing the pole away at the top."),
        }),
    # ------------------------------------------------------------------ HURDLES
    "Elite 110m Hurdle Clearance": dict(
        event="Hurdles", athlete_height_m=1.88, reference_video="elite_hurdle_clearance.mp4",
        description="One full hurdle clearance, takeoff step to landing step.",
        com_curve=_hurdle_curve(),
        keyframes={"Hurdle 1 hip apex": {"Lead-leg knee (deg)": 170, "Hip rise above running height (% stature)": 7,
                                         "Trunk lean (deg)": 30}},
        metrics={
            "lead_leg_extension_deg": _m(
                "Lead-leg extension (hip-knee-ankle)", "°", 170.0, 6.0, (120, 190),
                "Your lead leg is bent over the hurdle, which delays the snap-down and costs time on the landing step.",
                "Your lead leg is hyper-extended past the pro's range - risk of a stiff, braking landing.",
                "Lead-leg wall swings: hold a wall, swing the lead leg straight up to hurdle height 3 x 10 each leg to improve extension flexibility.",
                "Lead-leg 'snap-down' drills over a low hurdle, focusing on a relaxed, quick strike back to the ground."),
            "hip_rise_pct": _m(
                "COM peak above running height", "% stature", 7.0, 2.0, (0, 20),
                "Your hips barely rise: you may be sinking into the hurdle and clipping it with the lead knee or trail leg.",
                "You are jumping the hurdle: the hips rise much higher than the pro's, wasting time in the air.",
                "Trail-leg and lead-leg wall drills, then 5-step hurdle runs with a lower takeoff distance.",
                "Low-hurdle runs (0.6-0.7 m): 'run over, not jump over' - keep the chest forward and the lead knee driving, not up."),
            "com_traj_rmse_pct": _m(
                "COM trajectory shape error", "% stature", 0.0, 1.5, (0, 6),
                "", "Your hip path through the hurdle differs from the pro's flat, efficient arc.",
                "", "Hurdle walk-overs and 3-step rhythm runs: practise a flat hip line from takeoff to landing step.",
                decimals=1),
        }),
}


# --------------------------------------------------------------------------------------
# Public helpers
# --------------------------------------------------------------------------------------
def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_library() -> Dict[str, dict]:
    """PRO_LIBRARY with optional overrides from pro_database.json merged in."""
    lib = copy.deepcopy(PRO_LIBRARY)
    if os.path.exists(OVERRIDES_PATH):
        try:
            with open(OVERRIDES_PATH, "r", encoding="utf-8") as fh:
                _deep_update(lib, json.load(fh))
        except Exception:
            pass
    return lib


def list_entries(event: Optional[str] = None) -> List[str]:
    return [n for n, e in load_library().items() if event is None or e["event"] == event]


def get_entry(name: str) -> dict:
    return load_library()[name]


def resolve_reference_video(entry: dict) -> Optional[str]:
    """Absolute path of the entry's reference clip if it exists in PRO_VIDEO_DIR, else None."""
    fn = entry.get("reference_video")
    if not fn or not os.path.isdir(PRO_VIDEO_DIR):
        return None
    stem = os.path.splitext(fn)[0].lower()
    for f in os.listdir(PRO_VIDEO_DIR):
        if os.path.splitext(f)[0].lower() == stem and f.lower().endswith(VIDEO_EXTS):
            return os.path.join(PRO_VIDEO_DIR, f)
    return None


def with_measured_values(entry: dict, measured: Dict[str, float], curves: Optional[Dict[str, np.ndarray]] = None) -> dict:
    """Copy of `entry` whose metric values are replaced by what was measured from the pro clip."""
    e = copy.deepcopy(entry)
    for k, spec in e["metrics"].items():
        v = measured.get(k)
        if v is not None and math.isfinite(v) and k != "com_traj_rmse_pct":
            spec["value"] = round(float(v), 2)
    if curves and "com_trajectory" in curves and np.isfinite(curves["com_trajectory"]).all():
        e["com_curve"] = [round(float(x), 3) for x in curves["com_trajectory"]]
    return e


def export_json(path: str = OVERRIDES_PATH) -> None:
    """Write the full library to JSON so it can be edited by hand (and re-loaded as overrides)."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(PRO_LIBRARY, fh, indent=2, ensure_ascii=False)
