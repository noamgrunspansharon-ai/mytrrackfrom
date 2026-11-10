"""
app.py - Track & Field AI Coach (Streamlit)

Modes
  1. Pro Comparison         : upload your video, pick a pro reference, get side-by-side phase-synced skeleton
                              playback + metric-by-metric deltas with corrective drills.
  2. Reaction Time Practice : "On your marks / Set / Beep" + reaction time from horizontal movement.

Run:  streamlit run app.py
"""
from __future__ import annotations

import math
import os
import tempfile
import uuid

import numpy as np
import pandas as pd
import streamlit as st

from biomechanics import ANALYZERS, CAMERA_HINTS, analyze_event, build_kinematics
from coaching_engine import ComparisonReport, MetricComparison, compare_to_pro
from pose_engine import (PoseEngine, extract_keypoints, extract_keypoints_cached, render_annotated_video,
                         render_comparison_video)
from pro_database import (PRO_VIDEO_HINT, get_entry, list_entries, resolve_reference_video, with_measured_values)
from starter_gun import (StarterGun, analyze_reaction_video, detect_beep_in_video, run_live_session)

st.set_page_config(page_title="Track & Field AI Coach", page_icon="🏃", layout="wide")

WORKDIR = os.path.join(tempfile.gettempdir(), "tf_coach")
CACHE_DIR = os.path.join(WORKDIR, "pro_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

CSS = """
<style>
.gauge{position:relative;height:16px;border-radius:8px;background:rgba(128,128,128,.25);margin:14px 4px 6px 4px}
.gauge .band{position:absolute;top:0;height:100%;background:rgba(34,197,94,.45);border-radius:8px}
.gauge .mk{position:absolute;top:-6px;width:6px;height:28px;border-radius:3px;transform:translateX(-3px)}
.gauge .mk.pro{background:#2563eb}
.legend{font-size:.8rem;opacity:.85;display:flex;gap:14px;flex-wrap:wrap}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px}
.chip{display:inline-block;padding:2px 10px;border-radius:12px;font-weight:600;font-size:.85rem;color:#fff}
.bigscore{font-size:2.6rem;font-weight:800;line-height:1}
@media (max-width:640px){.bigscore{font-size:2rem}}
</style>
"""
STATUS_COL = {"match": "#22c55e", "close": "#f59e0b", "off": "#ef4444", "missing": "#6b7280"}
STATUS_TXT = {"match": "MATCH", "close": "CLOSE", "off": "FIX THIS", "missing": "NOT MEASURED"}


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def save_upload(upload) -> str:
    path = os.path.join(WORKDIR, f"{uuid.uuid4().hex[:10]}{os.path.splitext(upload.name)[1] or '.mp4'}")
    with open(path, "wb") as fh:
        fh.write(upload.getbuffer())
    return path


def fmt(v: float, it: MetricComparison) -> str:
    return f"{v:.{it.decimals}f}{'' if it.unit == '°' else ' '}{it.unit}"


def gauge_html(it: MetricComparison) -> str:
    lo, hi = it.scale

    def pos(v: float) -> float:
        return 100.0 * min(1.0, max(0.0, (v - lo) / (hi - lo)))

    col = STATUS_COL[it.status]
    bl, br = pos(it.pro - it.tol), pos(it.pro + it.tol)
    you = (f'<div class="mk" style="left:{pos(it.user):.1f}%;background:{col}"></div>'
           if it.status != "missing" else "")
    return (f'<div class="gauge"><div class="band" style="left:{bl:.1f}%;width:{br - bl:.1f}%"></div>'
            f'<div class="mk pro" style="left:{pos(it.pro):.1f}%"></div>{you}</div>'
            f'<div class="legend"><span><span class="dot" style="background:#2563eb"></span>Pro {fmt(it.pro, it)}</span>'
            f'<span><span class="dot" style="background:{col}"></span>You '
            f'{fmt(it.user, it) if it.status != "missing" else "n/a"}</span>'
            f'<span><span class="dot" style="background:rgba(34,197,94,.7)"></span>Pro window ±{it.tol:g}</span></div>')


def metric_card(it: MetricComparison) -> None:
    with st.container(border=True):
        left, right = st.columns([1, 1.5])
        with left:
            st.markdown(f"**{it.label}**  "
                        f"<span class='chip' style='background:{STATUS_COL[it.status]}'>{STATUS_TXT[it.status]}</span>",
                        unsafe_allow_html=True)
            if it.status != "missing":
                st.metric("You vs pro", fmt(it.user, it), f"{it.delta:+.{it.decimals}f} vs pro {fmt(it.pro, it)}",
                          delta_color="off" if it.status == "match" else "inverse")
            st.markdown(gauge_html(it), unsafe_allow_html=True)
        with right:
            st.write(it.advice)
            if it.drill:
                st.success(f"🛠️ **Drill:** {it.drill}")


# --------------------------------------------------------------------------------------
# PRO COMPARISON
# --------------------------------------------------------------------------------------
def run_comparison(upload, event, entry_name, height_cm, direction, complexity, speed, max_width,
                   custom_pro, use_measured) -> None:
    entry = get_entry(entry_name)
    bar = st.progress(0.0, text="Reading your video...")
    user_path = save_upload(upload)
    with PoseEngine(complexity) as eng:
        udata = extract_keypoints(user_path, eng, max_width, lambda p: bar.progress(0.3 * p, text="Tracking your skeleton..."))
    if udata.n_valid < 8:
        bar.empty()
        st.error("Could not find a person in enough frames. Film side-on, full body, good light.")
        return
    ukin = build_kinematics(udata.keypoints, udata.fps, udata.size, height_cm / 100.0, direction)
    ures = analyze_event(event, ukin)

    # pro reference clip: uploaded > bundled > none
    pro_path, pro_label = None, entry_name
    if custom_pro is not None:
        pro_path, pro_label = save_upload(custom_pro), f"Custom: {custom_pro.name}"
    else:
        pro_path = resolve_reference_video(entry)
    pdata = pres = None
    if pro_path:
        bar.progress(0.32, text="Tracking the pro reference...")
        pdata = extract_keypoints_cached(pro_path, CACHE_DIR, lambda: PoseEngine(complexity), max_width,
                                         lambda p: bar.progress(0.32 + 0.18 * p, text="Tracking the pro reference..."))
        pkin = build_kinematics(pdata.keypoints, pdata.fps, pdata.size, entry.get("athlete_height_m", 1.8), 0)
        pres = analyze_event(event, pkin)
        if custom_pro is not None or use_measured:
            entry = with_measured_values(entry, pres.metrics, pres.curves)

    report = compare_to_pro(entry_name, entry, ures)
    out = os.path.join(WORKDIR, f"cmp_{uuid.uuid4().hex[:8]}.mp4")
    bar.progress(0.5, text="Rendering synced side-by-side video...")
    prog = lambda p: bar.progress(0.5 + 0.5 * p, text="Rendering video...")
    ud = dict(annotations=ures.annotations, angle_marks=ures.angle_marks, key_frames=ures.key_frames)
    if pdata is not None:
        pd_ = dict(annotations=pres.annotations, angle_marks=pres.angle_marks, key_frames=pres.key_frames)
        path, stills, playable, anchors = render_comparison_video(user_path, udata, ud, pro_path, pdata, pd_, out,
                                                                  pro_label, speed, progress_cb=prog)
    else:
        path, stills, playable = render_annotated_video(user_path, udata, out, ures.annotations, ures.angle_marks,
                                                        ures.key_frames, speed, prog)
        anchors = []
    bar.empty()
    st.session_state["cmp"] = dict(report=report, ures=ures, pres=pres, entry=entry, path=path, playable=playable,
                                   stills=stills, anchors=anchors, side_by_side=pdata is not None,
                                   event=event, pro_name=pro_label, user_path=user_path, kin_dir=ukin.direction)


def show_comparison(s: dict) -> None:
    report: ComparisonReport = s["report"]
    entry = s["entry"]
    score = report.overall_score
    st.divider()
    top = st.columns([1, 3])
    with top[0]:
        col = "#22c55e" if score >= 80 else "#f59e0b" if score >= 55 else "#ef4444"
        st.markdown(f"<div class='bigscore' style='color:{col}'>{score:.0f}%</div>match to pro"
                    if math.isfinite(score) else "<div class='bigscore'>–</div>", unsafe_allow_html=True)
    top[1].markdown(f"**{report.headline}**")
    for w in report.warnings:
        st.warning(w)

    vid, panel = st.columns([3, 2])
    with vid:
        st.markdown("##### YOU vs PRO (phase-synced)" if s["side_by_side"] else "##### Your annotated video")
        if s["playable"]:
            st.video(s["path"])
        else:
            st.warning("Could not make a browser-playable file (ffmpeg missing). Use the download button.")
        with open(s["path"], "rb") as fh:
            st.download_button("⬇️ Download video", fh.read(), file_name="comparison.mp4", mime="video/mp4")
        if s["side_by_side"]:
            if s["anchors"]:
                st.caption("Synced at: " + " → ".join(a[0] for a in s["anchors"]) +
                           ". The pro clip is time-warped between these phases so every phase lines up. "
                           "Video plays in slow motion.")
            else:
                st.caption("No matching phases found in both clips - the pro was simply stretched to your clip length.")
        else:
            st.info("No pro video is available for this entry, so you see your own annotated video compared against "
                    "the reference values. " + PRO_VIDEO_HINT + " Or upload a clip of your own reference above.")
    with panel:
        st.markdown("##### Top priorities")
        if not report.priorities:
            st.success("All measured metrics are inside the pro's window.")
        for i, it in enumerate(report.priorities[:3], 1):
            with st.container(border=True):
                st.markdown(f"**{i}. {it.label}** — you {fmt(it.user, it)} vs pro {fmt(it.pro, it)}")
                if it.drill:
                    st.markdown(f"🛠️ {it.drill}")
        for ins in report.insights:
            st.info("💡 " + ins)

    st.markdown("### Metric-by-metric: you vs the pro")
    for it in report.items:
        metric_card(it)

    tabs = st.tabs(["Phase snapshots", "Pro keyframe data", "Charts", "Raw metrics"])
    with tabs[0]:
        if s["stills"]:
            cols = st.columns(2)
            for n, (lab, img) in enumerate(s["stills"].items()):
                cols[n % 2].image(img, caption=lab)
        else:
            st.caption("No key-frame snapshots were produced for this clip.")
    with tabs[1]:
        st.caption("Reference joint angles / values at crucial phases (approximate; editable in pro_database.py).")
        for phase, vals in entry.get("keyframes", {}).items():
            st.markdown(f"**{phase}**")
            st.table(pd.DataFrame({"Value": vals}))
    with tabs[2]:
        ures = s["ures"]
        if ures.curves.get("com_trajectory") is not None and entry.get("com_curve"):
            n = len(ures.curves["com_trajectory"])
            df = pd.DataFrame({"You": ures.curves["com_trajectory"], "Pro": entry["com_curve"][:n]},
                              index=np.round(np.linspace(-0.5, 0.5, n), 2))
            st.markdown("**COM trajectory over the hurdle (% stature above running height; x = s from apex)**")
            st.line_chart(df)
        for name, ser in ures.series.items():
            st.markdown(f"**{name}**")
            st.line_chart(pd.DataFrame({name: ser}, index=np.round(ures.t, 3)))
    with tabs[3]:
        st.dataframe(pd.DataFrame([(k, v) for k, v in s["ures"].metrics.items()], columns=["Metric", "Value"]),
                     hide_index=True)


def pro_comparison_page() -> None:
    st.header("🏆 Pro Comparison")
    c1, c2 = st.columns(2)
    event = c1.selectbox("Event", list(ANALYZERS))
    names = list_entries(event)
    entry_name = c2.selectbox("Pro reference", names)
    entry = get_entry(entry_name)
    clip = resolve_reference_video(entry)
    st.caption(f"{entry['description']}  •  " + ("🎞️ pro video available - side-by-side playback"
                                                 if clip else "📊 reference values only (no pro clip in pro_videos/)"))
    st.caption("📷 " + CAMERA_HINTS[event])

    with st.expander("Settings", expanded=False):
        a, b, c = st.columns(3)
        height = a.number_input("Your height (cm)", 120, 230, 175)
        dir_choice = b.selectbox("Direction of travel", ["Auto", "Left → Right", "Right → Left"])
        speed = c.selectbox("Playback speed", [0.5, 0.25, 1.0], format_func=lambda v: f"{v}x")
        d, e, f = st.columns(3)
        complexity = d.selectbox("Pose model", [1, 0, 2], format_func=lambda v: {0: "Lite (fast)", 1: "Full", 2: "Heavy (accurate)"}[v])
        max_w = e.select_slider("Processing width (px)", [480, 640, 800, 960], 640)
        use_measured = f.checkbox("Use values measured from the pro clip instead of database values", False,
                                  disabled=clip is None)
        custom = st.file_uploader("Optional: your own pro reference clip (overrides the built-in)",
                                  type=["mp4", "mov", "m4v", "avi", "webm"], key="custom_pro")

    upload = st.file_uploader("Upload YOUR video (or record one on your phone)", type=["mp4", "mov", "m4v", "avi", "webm"])
    if upload and st.button("Compare with the pro", type="primary", use_container_width=True):
        direction = {"Auto": 0, "Left → Right": 1, "Right → Left": -1}[dir_choice]
        try:
            run_comparison(upload, event, entry_name, float(height), direction, complexity, speed, max_w,
                           custom, use_measured)
        except Exception as exc:  # surface friendly error, keep app alive
            st.error(f"Analysis failed: {exc}")
    if "cmp" in st.session_state:
        show_comparison(st.session_state["cmp"])


# --------------------------------------------------------------------------------------
# REACTION TIME
# --------------------------------------------------------------------------------------
def show_reaction(res, still=None) -> None:
    st.divider()
    if not res.valid:
        st.error(res.message or "No reaction detected.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Reaction time", f"{res.reaction_time_s * 1000:.0f} ms")
        c2.metric("Rating", res.category.split(" (")[0].split(" -")[0])
        c3.metric("Time resolution", f"±{res.resolution_s * 1000:.0f} ms")
        (st.error if res.false_start else st.success)(res.message + (" " + res.category if res.false_start else ""))
    for w in res.warnings:
        st.warning(w)
    if len(res.t_rel):
        m = (res.t_rel > -1.2) & (res.t_rel < 1.8)
        df = pd.DataFrame({"Forward COM speed (torso-lengths/s)": res.speed[m],
                           "Detection threshold": np.full(m.sum(), res.threshold)}, index=np.round(res.t_rel[m], 3))
        st.markdown("**Forward speed vs time (0 = beep)**")
        st.line_chart(df)
    if still is not None:
        st.image(still, caption="Frame at movement onset")
    st.caption("Measures first visible horizontal movement, so it reads longer than block-pressure timing. "
               "Accuracy ≈ ±1 video frame.")


def reaction_page() -> None:
    st.header("⏱️ Reaction Time Practice")
    tab_live, tab_video = st.tabs(["Live camera (runs on this computer)", "Recorded video (works on phone)"])
    dir_map = {"Auto": 0, "Left → Right": 1, "Right → Left": -1}

    with tab_live:
        st.caption("Needs a webcam + speakers on the computer that RUNS the app. Place the camera side-on, hips and "
                   "shoulders in frame. On a phone / hosted app use the 'Recorded video' tab.")
        a, b, c = st.columns(3)
        cam = a.number_input("Camera index", 0, 10, 0)
        ready = b.slider("Get-ready time (s)", 3, 15, 6)
        direction = c.selectbox("Run direction", list(dir_map), key="live_dir")
        d, e, f = st.columns(3)
        gap = d.slider("'On your marks' → 'Set' (s)", 1.0, 5.0, 3.0, 0.5)
        lo, hi = e.slider("'Set' → beep (random, s)", 1.0, 4.0, (1.5, 2.5), 0.1)
        lat = f.number_input("Camera latency compensation (ms)", 0, 250, 0)
        tts = st.checkbox("Spoken voice cues (falls back to tones if no TTS)", True)
        if st.button("▶ Start", type="primary", use_container_width=True):
            status, view = st.empty(), st.empty()

            def on_frame(img, label):
                status.markdown(f"## {label}")
                view.image(img, channels="BGR")
            try:
                with PoseEngine(0) as eng:
                    res = run_live_session(eng, StarterGun(gap, (lo, hi), tts), int(cam), float(ready),
                                           direction=dir_map[direction], camera_latency_s=lat / 1000.0, frame_cb=on_frame)
                status.empty()
                st.session_state["live_res"] = res
            except Exception as exc:
                status.empty()
                st.error(str(exc))
        if "live_res" in st.session_state:
            show_reaction(st.session_state["live_res"])
            r = st.session_state["live_res"]
            st.caption(f"Audio backend: {r.backend} • sampled at {r.sampling_fps:.0f} fps")

    with tab_video:
        st.caption("Film yourself in the 'Set' position with the beep audible (the phone mic picks it up), then upload. "
                   "Include ~1 s of stillness before the beep.")
        with st.expander("🔊 Generate starter audio to play while filming"):
            g1, g2 = st.columns(2)
            gap2 = g1.slider("'Marks' → 'Set' (s)", 1.0, 5.0, 3.0, 0.5, key="gap2")
            tts2 = g2.checkbox("Spoken cues", True, key="tts2")
            if st.button("Generate", key="gen_audio"):
                sess = StarterGun(gap2, (1.5, 2.5), tts2).build()
                wav = sess.to_wav_bytes()
                st.audio(wav, format="audio/wav")
                st.download_button("Download WAV", wav, "starter_sequence.wav", "audio/wav")
                st.caption(f"Beep at {sess.beep_start:.2f} s into the audio. "
                           + ("" if sess.used_tts else "(No TTS engine found - using tone cues.)"))
        up = st.file_uploader("Upload your start video", type=["mp4", "mov", "m4v", "avi", "webm"], key="rt_video")
        mode = st.radio("Beep time", ["Auto-detect from the video's audio", "Enter manually"], horizontal=True)
        manual = st.number_input("Beep time in the video (s)", 0.0, 600.0, 2.0, 0.01, disabled=mode.startswith("Auto"))
        dirv = st.selectbox("Run direction", list(dir_map), key="vid_dir")
        if up and st.button("Analyze reaction", type="primary", use_container_width=True):
            path = save_upload(up)
            beep = detect_beep_in_video(path) if mode.startswith("Auto") else manual
            if beep is None:
                st.error("No 1 kHz beep found in the audio. Use 'Enter manually' (scrub the video to the beep).")
            else:
                st.info(f"Beep at {beep:.2f} s in the video.")
                with st.spinner("Tracking movement..."), PoseEngine(1) as eng:
                    res, still = analyze_reaction_video(path, eng, beep, dir_map[dirv])
                st.session_state["vid_res"] = (res, still)
        if "vid_res" in st.session_state:
            show_reaction(*st.session_state["vid_res"])


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("🏃 Track & Field AI Coach")
    with st.sidebar:
        mode = st.radio("Mode", ["Pro Comparison", "Reaction Time Practice"])
        with st.expander("📱 Install on phone / PC"):
            st.markdown("**Phone:** open this app's web address in Safari/Chrome → Share/⋮ menu → "
                        "**Add to Home Screen**.\n\n**PC:** Chrome/Edge → ⋮ → *Install app* / *Create shortcut*.\n\n"
                        "See README.md for hosting (Streamlit Cloud/Docker) or running on your PC.")
    (pro_comparison_page if mode == "Pro Comparison" else reaction_page)()


main()
