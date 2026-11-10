"""
coaching_engine.py
==================
Compares the athlete's measured biomechanics (biomechanics.AnalysisResult) with a pro reference entry
(pro_database) and writes specific, actionable advice that always includes the DELTA and a corrective drill.

Rule of thumb used everywhere:
    delta = user - pro
    |delta| <= tol       -> "match"   (inside the pro's window)
    |delta| <= 2 * tol   -> "close"   (worth refining)
    otherwise            -> "off"     (priority fix)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from biomechanics import AnalysisResult

NAN = float("nan")


@dataclass
class MetricComparison:
    key: str
    label: str
    unit: str
    user: float
    pro: float
    tol: float
    delta: float
    status: str                 # match | close | off | missing
    advice: str
    drill: str
    scale: Tuple[float, float]
    decimals: int = 0
    score: float = 0.0          # 0..1 similarity

    @property
    def severity(self) -> float:
        return abs(self.delta) / self.tol if self.status in ("close", "off") and self.tol else 0.0


@dataclass
class ComparisonReport:
    event: str
    pro_name: str
    items: List[MetricComparison]
    insights: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        got = [i.score for i in self.items if i.status != "missing"]
        return float(np.mean(got) * 100.0) if got else NAN

    @property
    def priorities(self) -> List[MetricComparison]:
        return sorted([i for i in self.items if i.status in ("close", "off")], key=lambda i: -i.severity)

    @property
    def headline(self) -> str:
        sc = self.overall_score
        if not math.isfinite(sc):
            return "Could not measure enough to compare - see the warnings."
        p = self.priorities
        base = f"{sc:.0f}% match to the pro."
        if not p:
            return base + " Every measured metric is inside the pro's window - great work."
        return base + f" Biggest gap: {p[0].label} ({_fmt(p[0].user, p[0])} vs pro {_fmt(p[0].pro, p[0])})."


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _ok(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating)) and math.isfinite(v)


def _fmt(v: float, it_or_spec) -> str:
    dec = it_or_spec.decimals if hasattr(it_or_spec, "decimals") else it_or_spec.get("decimals", 0)
    unit = it_or_spec.unit if hasattr(it_or_spec, "unit") else it_or_spec["unit"]
    return f"{v:.{dec}f}{'' if unit == '°' else ' '}{unit}"


def _status(delta: float, tol: float) -> str:
    a = abs(delta)
    if a <= tol:
        return "match"
    return "close" if a <= 2 * tol else "off"


def _score(delta: float, tol: float) -> float:
    a = abs(delta)
    return 1.0 if a <= tol else max(0.0, 1.0 - (a - tol) / (2.0 * tol))


def _compare(entry: dict, key: str, user: float, extra_advice: str = "") -> MetricComparison:
    """Core comparison for a single metric (uses cue/drill text stored in the pro database)."""
    spec = entry["metrics"][key]
    pro, tol = float(spec["value"]), float(spec["tol"])
    base = dict(key=key, label=spec["label"], unit=spec["unit"], pro=pro, tol=tol,
                scale=tuple(spec["scale"]), decimals=spec.get("decimals", 0))
    if not _ok(user):
        return MetricComparison(user=NAN, delta=NAN, status="missing", score=0.0,
                                advice=f"{spec['label']} could not be measured from this clip (see warnings).",
                                drill="Re-film side-on with a static camera at 60+ fps, full body in frame.", **base)
    delta = float(user) - pro
    if abs(delta) <= tol:  # inside the pro window
        advice = (f"On target: your {spec['label'].lower()} is {_fmt(user, spec)} vs the pro's {_fmt(pro, spec)} "
                  f"(within +/-{tol:g}).")
        drill = ""
    else:
        high = delta > 0
        cue = spec["cue_high"] if high else spec["cue_low"]
        drill = spec["drill_high"] if high else spec["drill_low"]
        advice = (f"The pro's {spec['label'].lower()} is {_fmt(pro, spec)}, but yours is {_fmt(user, spec)} "
                  f"(delta {delta:+.{base['decimals']}f}{'' if spec['unit'] == '°' else ' ' + spec['unit']}). {cue}")
    if extra_advice:
        advice += " " + extra_advice
    return MetricComparison(user=float(user), delta=delta, status=_status(delta, tol), score=_score(delta, tol),
                            advice=advice, drill=drill, **base)


def _get(items: List[MetricComparison], key: str) -> Optional[MetricComparison]:
    return next((i for i in items if i.key == key), None)


# --------------------------------------------------------------------------------------
# Event comparators: each returns (items, insights)
# --------------------------------------------------------------------------------------
def compare_long_jump(entry: dict, res: AnalysisResult) -> Tuple[List[MetricComparison], List[str]]:
    """Takeoff angle + penultimate-step COM drop (+ free-knee drive)."""
    m = res.metrics
    items = [_compare(entry, "takeoff_angle_deg", m.get("takeoff_angle_deg", NAN)),
             _compare(entry, "penultimate_com_drop_pct", m.get("penultimate_com_drop_pct", NAN)),
             _compare(entry, "knee_drive_deg", m.get("knee_drive_deg", NAN))]
    ang, drop, knee = (_get(items, k) for k in ("takeoff_angle_deg", "penultimate_com_drop_pct", "knee_drive_deg"))
    ins = []
    if ang.status in ("close", "off") and drop.status in ("close", "off") and ang.delta < 0 and drop.delta < 0:
        ins.append("Your flat takeoff and shallow penultimate lowering are linked: without a lower COM entering the "
                   "last step there is no room to rise. Fix the penultimate step first, then re-test the takeoff angle.")
    if ang.status in ("close", "off") and ang.delta < 0 and knee.status in ("close", "off") and knee.delta > 0:
        ins.append("A weak free-knee drive is a likely cause of the flat takeoff: the knee drive is what lifts the COM "
                   "through the board.")
    if ang.status in ("close", "off") and ang.delta > 0 and drop.status in ("close", "off") and drop.delta > 0:
        ins.append("Over-lowering on the penultimate step often produces a steep, braking final step - stay tall "
                   "and keep the foot landing under the hips.")
    return items, ins


def compare_high_jump(entry: dict, res: AnalysisResult) -> Tuple[List[MetricComparison], List[str]]:
    """Takeoff lean + flight arch, with premature-arch timing checks."""
    m = res.metrics
    onset = m.get("arch_onset_frac", NAN)
    offs = m.get("arch_peak_vs_apex_s", NAN)
    arch_extra = ""
    if _ok(onset) and onset < 0.40:
        arch_extra = (f"Your arch also starts after only {onset * 100:.0f}% of the rise - premature arching; "
                      "stay tall and keep the chest up until the head is over the bar.")
    elif _ok(offs) and offs > 0.20:
        arch_extra = (f"Your arch peaks {offs:.2f} s after the hips top out - too late; the hips drop onto the bar "
                      "before you kick the legs up.")
    items = [_compare(entry, "takeoff_lean_deg", m.get("takeoff_lean_deg", NAN)),
             _compare(entry, "peak_arch_angle_deg", m.get("peak_arch_angle_deg", NAN), arch_extra)]
    lean, arch = items
    ins = []
    if lean.status in ("close", "off") and lean.delta < 0 and arch.status in ("close", "off") and arch.delta > 0:
        ins.append("Being too upright at takeoff usually produces a flat arch: the curve gives you the rotation "
                   "that lets the shoulders drop back. Work on the curve lean first.")
    if _ok(offs) and offs < -0.12:
        ins.append(f"Your peak arch happens {abs(offs):.2f} s BEFORE the hips reach the top - you are curling over "
                   "too early, which kills your vertical lift.")
    if lean.status in ("close", "off") and lean.delta > 0:
        ins.append("Camera note: takeoff lean is measured in the image plane. If you filmed side-on, the inward "
                   "lean is under-/over-estimated - film from the front or rear to confirm.")
    return items, ins


def compare_triple_jump(entry: dict, res: AnalysisResult) -> Tuple[List[MetricComparison], List[str]]:
    """Hop / step / jump phase ratios."""
    m = res.metrics
    items = [_compare(entry, k, m.get(k, NAN)) for k in ("hop_pct", "step_pct", "jump_pct")]
    hop, step, jump = items
    ins = []
    if all(i.status != "missing" for i in items):
        if hop.delta > hop.tol and step.delta < -step.tol:
            ins.append("Pattern detected: HOP-DOMINANT WITH A STEP COLLAPSE. The hop takes too much speed, leaving a "
                       "short step. Shorten the hop slightly and stay flat so the step can carry speed.")
        elif jump.delta < -jump.tol and hop.delta > hop.tol:
            ins.append("Pattern detected: the hop is eating the jump. Cap the hop distance (cone at the target) and "
                       "keep the first landing active.")
        elif step.delta > step.tol and jump.delta < -jump.tol:
            ins.append("Pattern detected: OVER-REACHING STEP. A long step usually means a braking landing and a "
                       "short final jump - land under the hips and push the hips through.")
        elif all(i.status == "match" for i in items):
            ins.append("Your phase distribution is within the pro window - now check the active landing angles.")
        for lab in ("hop", "step"):
            foot = m.get(f"{lab}_landing_foot_angle_deg", NAN)
            leg = m.get(f"{lab}_landing_leg_angle_deg", NAN)
            if _ok(foot) and _ok(leg) and foot > 20 and leg > 22:
                ins.append(f"Your {lab} landing is heel-first with the leg {leg:.0f} deg ahead of the hips "
                           f"(foot {foot:.0f} deg toe-up) - that is a braking landing. Land flat and paw the foot "
                           "back under the hips.")
    return items, ins


def compare_pole_vault(entry: dict, res: AnalysisResult) -> Tuple[List[MetricComparison], List[str]]:
    """Plant angle + inversion angle (with bottom-arm pressure and swing speed context)."""
    m = res.metrics
    items = [_compare(entry, "plant_angle_deg", m.get("plant_angle_deg", NAN)),
             _compare(entry, "peak_inversion_deg", m.get("peak_inversion_deg", NAN))]
    plant, inv = items
    ins = []
    elbow = m.get("bottom_arm_min_elbow_deg", NAN)
    if _ok(elbow) and elbow < 150:
        ins.append(f"Bottom-arm pressure: your bottom elbow collapses to {elbow:.0f} deg after takeoff (a firm "
                   "bottom arm stays above ~165 deg). Keep that arm straight and press the pole forward/up.")
    sw = m.get("peak_swing_velocity_dps", NAN)
    if inv.status in ("close", "off") and inv.delta < 0 and _ok(sw) and sw < 250:
        ins.append(f"Swing speed is low ({sw:.0f} deg/s peak): the hips are rising slowly toward the shoulders. "
                   "Drive the trail leg and hips up and forward immediately after takeoff.")
    if plant.status in ("close", "off") and plant.delta < 0 and inv.status in ("close", "off") and inv.delta < 0:
        ins.append("A late/low plant limits the swing: fix the reach at takeoff and the inversion will usually improve.")
    return items, ins


def _curve_stats(user: np.ndarray, pro: np.ndarray) -> Tuple[float, float, int]:
    n = min(len(user), len(pro))
    u, p = np.asarray(user[:n], float), np.asarray(pro[:n], float)
    rmse = float(np.sqrt(np.nanmean((u - p) ** 2)))
    return rmse, float(np.nanmax(u) - np.nanmax(p)), int(np.nanargmax(u) - np.nanargmax(p))


def compare_hurdles(entry: dict, res: AnalysisResult) -> Tuple[List[MetricComparison], List[str]]:
    """Lead-leg extension + centre-of-mass trajectory (peak rise and path shape)."""
    m = res.metrics
    rmse, peak_diff, shift = NAN, NAN, 0
    user_curve = res.curves.get("com_trajectory")
    pro_curve = entry.get("com_curve")
    if user_curve is not None and pro_curve is not None and np.isfinite(user_curve).any():
        rmse, peak_diff, shift = _curve_stats(user_curve, np.array(pro_curve, float))
    items = [_compare(entry, "lead_leg_extension_deg", m.get("lead_leg_extension_deg", NAN)),
             _compare(entry, "hip_rise_pct", m.get("hip_rise_pct", NAN)),
             _compare(entry, "com_traj_rmse_pct", rmse)]
    lead, rise, traj = items
    ins = []
    if rise.status in ("close", "off") and rise.delta > 0:
        ins.append(f"You are JUMPING the hurdle: your COM peaks {m.get('hip_rise_cm', NAN):.0f} cm above running "
                   "height. Efficient hurdlers keep the hip path flat and snap the lead leg down early.")
    if traj.status in ("close", "off") and shift != 0:
        t_shift = shift * 1.0 / 24.0  # curve spans 1.0 s over 25 samples
        when = "later" if shift > 0 else "earlier"
        ins.append(f"Your COM peak is {abs(t_shift):.2f} s {when} than the pro's: "
                   + ("you are late over the hurdle - take off a touch farther out and attack the hurdle."
                      if shift > 0 else "you are chopping into the hurdle - check takeoff distance (too close)."))
    if lead.status in ("close", "off") and lead.delta < 0 and rise.status in ("close", "off") and rise.delta > 0:
        ins.append("A bent lead leg combined with a high hip peak means you are lifting the knee instead of "
                   "shooting the leg out - the lead-leg wall swings are your first priority.")
    return items, ins


COMPARATORS: Dict[str, Callable[[dict, AnalysisResult], Tuple[List[MetricComparison], List[str]]]] = {
    "Long Jump": compare_long_jump,
    "High Jump": compare_high_jump,
    "Triple Jump": compare_triple_jump,
    "Pole Vault": compare_pole_vault,
    "Hurdles": compare_hurdles,
}


def compare_to_pro(pro_name: str, entry: dict, result: AnalysisResult) -> ComparisonReport:
    """Main entry point used by the app."""
    ev = entry["event"]
    if ev not in COMPARATORS:
        raise ValueError(f"No comparator for event '{ev}'")
    items, insights = COMPARATORS[ev](entry, result)
    return ComparisonReport(ev, pro_name, items, insights, list(result.warnings))
