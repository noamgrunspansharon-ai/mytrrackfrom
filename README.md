# 🏃 Track & Field AI Coach

Upload a video of yourself, pick a pro reference, and get a **phase-synced side-by-side skeleton video**, metric-by-metric
**deltas vs the pro**, and a **corrective drill** for each gap. Also includes an on-your-marks/set/beep **reaction-time** trainer.

Events: Long Jump · High Jump (Fosbury Flop) · Triple Jump · Pole Vault · Hurdles

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI (Pro Comparison + Reaction Time) |
| `pose_engine.py` | MediaPipe BlazePose wrapper, skeleton drawing, synced side-by-side video |
| `biomechanics.py` | Angle/velocity math, ground-contact + flight detection, per-event metrics |
| `pro_database.py` | Pro reference values, tolerances, keyframes, drills (override via `pro_database.json`) |
| `coaching_engine.py` | Compares you vs pro, writes advice with the delta + drill |
| `starter_gun.py` | Audio cues, random delay, beep, reaction time from horizontal movement |

## A) Run on your PC (Windows / Mac / Linux)
Needs **Python 3.9–3.12** (MediaPipe 0.10.14 has no wheels for 3.13).
- **Windows:** double-click `run_windows.bat`
- **Mac/Linux:** `./run_mac_linux.sh`

Then open http://localhost:8501. Live-camera reaction mode works here (webcam + speakers of this PC).

## B) Use it on your phone

**Option 1 – same Wi-Fi as your PC (quickest, free):**
run `run_windows.bat phone` (or `./run_mac_linux.sh phone`), then on the phone open `http://<PC-IP>:8501`
(Windows: `ipconfig`, Mac: `ipconfig getifaddr en0`). Allow the app through the firewall if asked.

**Option 2 – host it online (works anywhere, free tier):**
1. Put this folder in a GitHub repo (keep `requirements.txt` and `packages.txt` at the top level).
2. Go to https://share.streamlit.io → *New app* → pick the repo → main file `app.py` → Python 3.11 → Deploy.
3. Open the URL on your phone. Docker hosts (Render, Fly.io, Cloud Run, Railway) can use the included `Dockerfile`.

**Make it look like an app:** open the URL in Safari/Chrome on the phone → Share / ⋮ menu → **Add to Home Screen**.
On PC Chrome/Edge: ⋮ → *Install app*.

> Streamlit apps are web apps, not App Store/Play Store apps. "Add to Home Screen" gives an icon that opens full-screen
> but it still needs the PC/server running or the hosted URL online.

### What works where
| Feature | PC (local) | Phone via hosted/Wi-Fi URL |
|---|---|---|
| Pro Comparison (upload video) | ✅ | ✅ (use *Browse → Take video*) |
| Reaction time – recorded video | ✅ | ✅ |
| Reaction time – live camera + speakers | ✅ | ❌ (camera/speakers belong to the server) |
| Spoken "On your marks / Set" | ✅ (`requirements-local.txt`) | tone cues fallback |

## Pro videos
No pro footage is bundled. Put licensed/own side-on clips in `pro_videos/` named as in `pro_database.py`
(`reference_video`, e.g. `elite_long_jump_takeoff.mp4`). Without a clip you still get comparison against the reference
values; with a clip you get the synced side-by-side playback. You can also upload any reference clip in *Settings*.
The built-in numbers are approximate reference values, not measurements of a named athlete - tune them in
`pro_database.py` or `pro_database.json`.

## Filming tips (matters a lot)
Static camera, side-on, 60+ fps, full body in frame, good light, no panning. High-jump lean and hurdle trail-leg
abduction are best seen from the front/rear; side-on gives a projected (2-D) angle.

## Reaction-time notes
Measures first visible horizontal movement of hips/shoulders, accuracy ≈ ±1 video frame; it reads longer than block-pressure
timing. For the recorded-video mode the phone mic must hear the 1 kHz beep (use *Generate starter audio*), or enter the beep time manually.
