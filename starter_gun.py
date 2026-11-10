"""
starter_gun.py
==============
"On your marks" ... "Set" ... random delay ... BEEP, plus reaction-time measurement from the athlete's
horizontal movement (centre-of-mass proxy = hips + shoulders, in torso-lengths per second so the
result does not depend on camera distance).

Two ways to measure
  1. run_live_session()          - LOCAL ONLY: webcam on the machine running the app + that machine's speakers.
                                   The whole cue sequence is rendered into ONE audio buffer, so the beep
                                   time is known exactly (no jitter between separate sounds).
  2. analyze_reaction_video()    - works anywhere (including on a phone via the hosted app): upload a clip that
                                   contains the beep. The beep can be auto-detected from the video's audio track
                                   (a pure 1 kHz tone) or entered manually.

Accuracy: limited by camera frame rate (+/- 1 frame, i.e. 33 ms at 30 fps) and, live, by camera latency
(use the latency compensation setting). Official reaction times use force sensors in the blocks; this
measures first visible horizontal movement, so it will read somewhat longer than a block-pressure value.
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from biomechanics import L_HIP, L_SHOULDER, R_HIP, R_SHOULDER

FS = 22050          # session sample rate
BEEP_HZ = 1000.0


# --------------------------------------------------------------------------------------
# Audio synthesis / TTS
# --------------------------------------------------------------------------------------
def synth_tone(freq: float, dur: float, fs: int = FS, vol: float = 0.9) -> np.ndarray:
    t = np.arange(int(dur * fs)) / fs
    x = np.sin(2 * np.pi * freq * t)
    ramp = max(1, int(0.005 * fs))
    env = np.ones_like(x)
    env[:ramp], env[-ramp:] = np.linspace(0, 1, ramp), np.linspace(1, 0, ramp)
    return (vol * x * env).astype(np.float32)


def read_wav(path: str) -> Tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        n, ch, sw, fs = w.getnframes(), w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(n)
    if sw == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError("unsupported WAV sample width")
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, fs


def _resample(x: np.ndarray, fs_from: int, fs_to: int) -> np.ndarray:
    if fs_from == fs_to or len(x) == 0:
        return x.astype(np.float32)
    n = int(round(len(x) * fs_to / fs_from))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def _tts_clip(text: str, rate: int = 150) -> Optional[np.ndarray]:
    """Offline TTS to a numpy array (pyttsx3, or macOS 'say'). Returns None when unavailable."""
    tmp = os.path.join(tempfile.gettempdir(), f"tf_tts_{abs(hash(text)) % 10**8}.wav")
    try:
        if sys.platform == "darwin" and shutil.which("say"):
            subprocess.run(["say", "-r", str(rate + 25), "-o", tmp, "--data-format=LEI16@22050", text],
                           check=True, timeout=20)
        else:
            import pyttsx3
            eng = pyttsx3.init()
            eng.setProperty("rate", rate)
            eng.save_to_file(text, tmp)
            eng.runAndWait()
            try:
                eng.stop()
            except Exception:
                pass
        x, fs = read_wav(tmp)
        if len(x) / fs < 0.25 or float(np.max(np.abs(x))) < 1e-3:
            return None
        x = _resample(x, fs, FS)
        return (0.9 * x / float(np.max(np.abs(x)))).astype(np.float32)
    except Exception:
        return None


_CLIP_CACHE: Dict[Tuple[str, bool], Tuple[np.ndarray, bool]] = {}


def _voice_clip(text: str, fallback_freq: float, fallback_dur: float, use_tts: bool) -> Tuple[np.ndarray, bool]:
    key = (text, use_tts)
    if key not in _CLIP_CACHE:
        clip = _tts_clip(text) if use_tts else None
        _CLIP_CACHE[key] = (clip, True) if clip is not None else (synth_tone(fallback_freq, fallback_dur), False)
    return _CLIP_CACHE[key]


@dataclass
class SessionAudio:
    samples: np.ndarray       # float32 mono @ FS
    fs: int
    marks_start: float
    set_start: float
    beep_start: float         # seconds from the start of the buffer
    random_delay_s: float
    used_tts: bool

    @property
    def duration(self) -> float:
        return len(self.samples) / self.fs

    def to_wav_bytes(self) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.fs)
            w.writeframes((np.clip(self.samples, -1, 1) * 32767).astype("<i2").tobytes())
        return buf.getvalue()


class StarterGun:
    """Builds the full cue sequence: 'On your marks' -> 3 s -> 'Set' -> random 1.5-2.5 s -> BEEP."""

    def __init__(self, marks_to_set_s: float = 3.0, set_delay_range: Tuple[float, float] = (1.5, 2.5),
                 use_tts: bool = True, lead_in_s: float = 0.5, tail_s: float = 0.3):
        self.marks_to_set_s, self.set_delay_range = marks_to_set_s, set_delay_range
        self.use_tts, self.lead_in_s, self.tail_s = use_tts, lead_in_s, tail_s

    def build(self, rng: Optional[np.random.Generator] = None) -> SessionAudio:
        rng = rng or np.random.default_rng()
        marks, ok1 = _voice_clip("On your marks", 440.0, 0.7, self.use_tts)
        setc, ok2 = _voice_clip("Set", 660.0, 0.35, self.use_tts)
        beep = synth_tone(BEEP_HZ, 0.40)
        delay = float(rng.uniform(*self.set_delay_range))

        def sil(sec: float) -> np.ndarray:
            return np.zeros(int(sec * FS), np.float32)

        parts = [sil(self.lead_in_s), marks, sil(self.marks_to_set_s), setc, sil(delay), beep, sil(self.tail_s)]
        marks_start = self.lead_in_s
        set_start = marks_start + len(marks) / FS + self.marks_to_set_s
        beep_start = set_start + len(setc) / FS + delay
        return SessionAudio(np.concatenate(parts), FS, marks_start, set_start, beep_start, delay, ok1 and ok2)


# --------------------------------------------------------------------------------------
# Playback (sounddevice with exact timing, else OS player)
# --------------------------------------------------------------------------------------
class Playback:
    def __init__(self, session: SessionAudio, backend: str, audio_start_perf: float, accurate: bool,
                 stop: Callable[[], None]):
        self.session, self.backend, self.audio_start_perf = session, backend, audio_start_perf
        self.accurate, self._stop = accurate, stop

    @property
    def beep_perf(self) -> float:
        return self.audio_start_perf + self.session.beep_start

    def stop(self) -> None:
        try:
            self._stop()
        except Exception:
            pass


def _play_sounddevice(session: SessionAudio) -> Playback:
    import sounddevice as sd
    buf, pos, first = session.samples.astype(np.float32), [0], {"t": None, "dac": None}

    def cb(outdata, frames, time_info, status):
        if first["t"] is None:
            first["t"] = time.perf_counter()
            try:
                first["dac"] = time_info.outputBufferDacTime - time_info.currentTime
            except Exception:
                first["dac"] = None
        n = max(0, min(frames, len(buf) - pos[0]))
        if n:
            outdata[:n, 0] = buf[pos[0]:pos[0] + n]
        pos[0] += n
        if n < frames:
            outdata[n:, 0] = 0
            raise sd.CallbackStop

    stream = sd.OutputStream(samplerate=session.fs, channels=1, dtype="float32", callback=cb)
    stream.start()
    deadline = time.perf_counter() + 1.0
    while first["t"] is None and time.perf_counter() < deadline:
        time.sleep(0.001)
    if first["t"] is None:
        stream.close()
        raise RuntimeError("audio callback never started")
    dac = first["dac"]
    lat = dac if (dac is not None and 0.0 <= dac <= 0.5) else float(stream.latency)
    return Playback(session, "sounddevice", first["t"] + lat, True, lambda: (stream.stop(), stream.close()))


def _play_system(session: SessionAudio) -> Playback:
    path = os.path.join(tempfile.gettempdir(), "tf_starter_session.wav")
    with open(path, "wb") as fh:
        fh.write(session.to_wav_bytes())
    t0 = time.perf_counter()
    if sys.platform.startswith("win"):
        import winsound
        winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        stop = lambda: winsound.PlaySound(None, winsound.SND_PURGE)
    else:
        cmd = (["afplay", path] if sys.platform == "darwin" else
               next(([c, *a, path] for c, a in (("paplay", []), ("aplay", ["-q"]), ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet"]))
                     if shutil.which(c)), None))
        if cmd is None:
            raise RuntimeError("No audio player found. Install 'sounddevice' (pip install sounddevice).")
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        stop = proc.terminate
    return Playback(session, "system-player", t0 + 0.10, False, stop)   # ~100 ms start-up guess


def play_session(session: SessionAudio) -> Playback:
    try:
        return _play_sounddevice(session)
    except Exception:
        return _play_system(session)


# --------------------------------------------------------------------------------------
# Reaction time maths
# --------------------------------------------------------------------------------------
@dataclass
class ReactionResult:
    valid: bool = False
    reaction_time_s: float = float("nan")
    false_start: bool = False
    category: str = ""
    message: str = ""
    direction: int = 0
    threshold: float = float("nan")
    peak_speed: float = float("nan")
    resolution_s: float = float("nan")
    sampling_fps: float = float("nan")
    onset_abs_s: float = float("nan")
    t_rel: np.ndarray = field(default_factory=lambda: np.array([]))
    speed: np.ndarray = field(default_factory=lambda: np.array([]))
    warnings: List[str] = field(default_factory=list)
    backend: str = ""
    audio_accurate: bool = True


def _classify(rt: float) -> str:
    if rt < 0.100:
        return "False start (< 0.100 s: anticipation, or timing/sync error)"
    if rt < 0.150:
        return "Elite"
    if rt < 0.200:
        return "Very good"
    if rt < 0.250:
        return "Good"
    if rt < 0.350:
        return "Average"
    return "Slow - work on set-position tension and focus on the sound"


def compute_reaction_time(t, hip_x, sh_x, torso, beep_t: float, direction: int = 0, camera_latency_s: float = 0.0,
                          k_sigma: float = 4.0, min_speed: float = 0.35, consecutive: int = 2,
                          baseline: Tuple[float, float] = (-1.0, -0.3)) -> ReactionResult:
    """
    t, hip_x, sh_x, torso : per-frame arrays (seconds, pixels, pixels, pixels) on the same clock as beep_t.
    direction             : +1 athlete runs left->right in the image, -1 right->left, 0 = auto.
    Returns the time from the beep to the onset of forward horizontal COM-proxy movement.
    """
    res = ReactionResult()
    t, hip_x, sh_x, torso = (np.asarray(a, float) for a in (t, hip_x, sh_x, torso))
    order = np.argsort(t)
    t, hip_x, sh_x, torso = t[order], hip_x[order], sh_x[order], torso[order]
    if len(t) < 12:
        res.message = "Not enough tracked frames - keep the whole body in view."
        return res
    scale = float(np.nanmedian(torso))
    if not np.isfinite(scale) or scale < 5:
        res.message = "Torso length could not be measured."
        return res
    res.sampling_fps = float(1.0 / np.median(np.diff(t)))
    res.resolution_s = float(np.median(np.diff(t)))
    x = (0.6 * hip_x + 0.4 * sh_x) / scale                        # COM proxy, in torso-lengths
    # Backward difference, placed at the interval midpoint: no centred smoothing, so no early-onset bias.
    dts = np.diff(t)
    keep = dts > 1e-6
    v = (np.diff(x) / np.where(keep, dts, 1.0))[keep]             # torso-lengths per second
    t = (0.5 * (t[1:] + t[:-1]))[keep]                            # time of each velocity sample
    rel = t - beep_t

    if direction == 0:
        post = (rel >= 0.05) & (rel <= 1.0)
        direction = 1 if (post.any() and np.mean(v[post]) >= 0) else -1
    res.direction = direction
    vd = v * direction
    res.t_rel, res.speed = rel, vd

    base = (rel >= baseline[0]) & (rel < baseline[1])
    if base.sum() < 5:
        res.message = ("Need about 1 s of tracked stillness before the beep (in the 'Set' position). "
                       "Check the beep time or start the clip earlier.")
        return res
    mu, sd = float(np.mean(vd[base])), float(np.std(vd[base]))
    thr = max(min_speed, mu + k_sigma * sd)
    onset_thr = mu + 0.3 * (thr - mu)
    res.threshold = thr
    if abs(mu) > 0.5 * min_speed or sd > 0.5 * min_speed:
        res.warnings.append("Athlete was not steady in the 'Set' position (movement/noise before the beep); "
                            "the result may be unreliable.")

    cand = np.where((rel >= baseline[1]) & (rel <= 1.5))[0]
    hit = next((i for i in cand if i + consecutive <= len(vd) and np.all(vd[i:i + consecutive] > thr)), None)
    if hit is None:
        res.message = "No start detected within 1.5 s of the beep (movement stayed below the threshold)."
        return res
    j = hit
    while j > 0 and rel[j - 1] >= baseline[1] - 0.2 and vd[j - 1] > onset_thr:
        j -= 1
    if j > 0 and vd[j - 1] <= onset_thr < vd[j]:
        frac = (onset_thr - vd[j - 1]) / (vd[j] - vd[j - 1])
        onset = t[j - 1] + frac * (t[j] - t[j - 1])
    else:
        onset = t[j]
    rt = float(onset - camera_latency_s - beep_t)
    res.valid, res.reaction_time_s, res.onset_abs_s = True, rt, float(onset)
    res.peak_speed = float(np.max(vd[(rel >= 0) & (rel <= 1.5)]))
    res.false_start = rt < 0.100
    res.category = _classify(rt)
    res.message = (f"Movement began {rt * 1000:.0f} ms after the beep." if rt >= 0 else
                   f"Movement began {abs(rt) * 1000:.0f} ms BEFORE the beep.")
    return res


class ReactionTracker:
    """Collects per-frame hip/shoulder x positions (pixels) for compute_reaction_time()."""

    def __init__(self) -> None:
        self.t: List[float] = []
        self.hip_x: List[float] = []
        self.sh_x: List[float] = []
        self.torso: List[float] = []

    def add(self, t: float, pose) -> bool:
        if pose is None:
            return False
        ids = [L_HIP, R_HIP, L_SHOULDER, R_SHOULDER]
        if np.min(pose.visibility[ids]) < 0.3 or not np.isfinite(pose.px[ids]).all():
            return False
        hip, sh = pose.px[[L_HIP, R_HIP]].mean(axis=0), pose.px[[L_SHOULDER, R_SHOULDER]].mean(axis=0)
        self.t.append(t)
        self.hip_x.append(float(hip[0]))
        self.sh_x.append(float(sh[0]))
        self.torso.append(float(np.linalg.norm(hip - sh)))
        return True

    def analyze(self, beep_t: float, **kw) -> ReactionResult:
        return compute_reaction_time(self.t, self.hip_x, self.sh_x, self.torso, beep_t, **kw)


# --------------------------------------------------------------------------------------
# Live session (local machine: webcam + speakers)
# --------------------------------------------------------------------------------------
def run_live_session(engine, gun: StarterGun, camera_index: int = 0, get_ready_s: float = 6.0,
                     tail_s: float = 2.0, direction: int = 0, camera_latency_s: float = 0.0,
                     audio_latency_adjust_s: float = 0.0, resolution: Tuple[int, int] = (640, 480),
                     frame_cb: Optional[Callable[[np.ndarray, str], None]] = None) -> ReactionResult:
    """Blocking. frame_cb(bgr_frame, status_text) is called ~every other frame for the live preview."""
    import cv2
    from pose_engine import draw_skeleton

    api = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(camera_index, api)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}. Live mode needs a webcam on the computer running "
                           "the app - on a phone/hosted app use the 'Recorded video' tab instead.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    session, tracker, playback = gun.build(), ReactionTracker(), None
    t_begin, n = time.perf_counter(), 0
    try:
        while True:
            ok, frame = cap.read()
            t = time.perf_counter()                      # timestamp immediately after the frame arrives
            if not ok:
                raise RuntimeError("Camera stopped delivering frames.")
            if playback is None and (t - t_begin) >= get_ready_s:
                playback = play_session(session)
            pose = engine.process_frame(frame)
            tracker.add(t, pose)
            n += 1
            if playback is None:
                status = f"Get in position... {max(0.0, get_ready_s - (t - t_begin)):.0f}"
            else:
                rel = t - playback.audio_start_perf
                status = ("Listen..." if rel < session.marks_start else "On your marks" if rel < session.set_start
                          else "SET - hold still" if rel < session.beep_start else "GO!")
            if frame_cb and n % 2 == 0:
                if pose is not None:
                    draw_skeleton(frame, pose.px, pose.visibility)
                frame_cb(frame, status)
            if playback is not None and t > playback.beep_perf + tail_s:
                break
            if t - t_begin > get_ready_s + session.duration + tail_s + 10:
                break
    finally:
        cap.release()
        if playback is not None:
            playback.stop()
    if playback is None:
        raise RuntimeError("Session ended before audio started.")
    res = tracker.analyze(beep_t=playback.beep_perf + audio_latency_adjust_s, direction=direction,
                          camera_latency_s=camera_latency_s)
    res.backend, res.audio_accurate = playback.backend, playback.accurate
    if not playback.accurate:
        res.warnings.append("Audio played through the OS player (start time approximate, +/-100 ms). "
                            "Install 'sounddevice' for millisecond-accurate beep timing.")
    return res


# --------------------------------------------------------------------------------------
# Recorded-video mode (works hosted / on phone)
# --------------------------------------------------------------------------------------
def find_tone_onset(x: np.ndarray, sr: int, hz: float = BEEP_HZ) -> Optional[float]:
    """Time (s) of the first sustained, pure tone at `hz` in mono samples x (-1..1), else None."""
    frame, hop = int(0.04 * sr), int(0.01 * sr)
    if len(x) < frame + 10 * hop:
        return None
    nfr = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(nfr)[:, None]
    spec = np.abs(np.fft.rfft(x[idx] * np.hanning(frame), axis=1)) ** 2
    f = np.fft.rfftfreq(frame, 1.0 / sr)
    band = spec[:, (f >= hz - 60) & (f <= hz + 60)].sum(axis=1)
    broad = spec[:, (f >= 300) & (f <= 4000)].sum(axis=1) + 1e-12
    cand = (band / broad > 0.5) & (band > 0.15 * band.max())
    run = 0
    for i, c in enumerate(cand):
        run = run + 1 if c else 0
        if run >= 6:
            return float(((i - run + 1) * hop + frame / 2) / sr)
    return None


def detect_beep_in_video(video_path: str, hz: float = BEEP_HZ, sr: int = 16000) -> Optional[float]:
    """Extract the audio track with ffmpeg and find the starter beep. None if no audio/ffmpeg/beep."""
    from pose_engine import find_ffmpeg
    exe = find_ffmpeg()
    if not exe:
        return None
    wav = os.path.join(tempfile.gettempdir(), f"tf_audio_{abs(hash(video_path)) % 10**8}.wav")
    try:
        subprocess.run([exe, "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "wav", wav],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
        x, fs = read_wav(wav)
        return find_tone_onset(x, fs, hz)
    except Exception:
        return None
    finally:
        try:
            os.remove(wav)
        except OSError:
            pass


def analyze_reaction_video(video_path: str, engine, beep_time_s: float, direction: int = 0,
                           max_width: int = 640) -> Tuple[ReactionResult, Optional[np.ndarray]]:
    """Track hips/shoulders around the beep in a recorded clip. Returns (result, RGB still at movement onset)."""
    import cv2
    from pose_engine import _read_fps, _resize_max_width, draw_skeleton

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Could not open the video.")
    fps = _read_fps(cap)
    lo, hi = beep_time_s - 1.6, beep_time_s + 2.0
    tracker, i = ReactionTracker(), 0
    while True:
        t = i / fps
        if t > hi:
            break
        if t < lo:
            if not cap.grab():
                break
        else:
            ok, frame = cap.read()
            if not ok:
                break
            tracker.add(t, engine.process_frame(_resize_max_width(frame, max_width)))
        i += 1
    cap.release()
    res = tracker.analyze(beep_t=beep_time_s, direction=direction)

    still = None
    if res.valid:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(res.onset_abs_s * fps))))
        ok, frame = cap.read()
        cap.release()
        if ok:
            frame = _resize_max_width(frame, max_width)
            pose = engine.process_frame(frame)
            if pose is not None:
                draw_skeleton(frame, pose.px, pose.visibility)
            still = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return res, still
