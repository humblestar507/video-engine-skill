"""Layer 1: onsets, tempo, loudness envelope and 3 spectral bands, on the video timeline.

Audio is decoded to mono 22.05 kHz, hop 128 samples (5.8 ms). Times are shifted by the
audio stream's start offset relative to the first video frame, so they line up
with video PTS. Onset times come from the peak of the band-summed energy rise,
not from librosa's frame grid, and are reported with their measured strength."""
import time
import warnings

import numpy as np

from . import media
from .store import D, M

VERSION = "2"
SR = 22050
HOP = 128
BANDS = {"low": (20, 250), "mid": (250, 2500), "high": (2500, 10000)}


def _band_env(S, freqs):
    out = {}
    for k, (lo, hi) in BANDS.items():
        m = (freqs >= lo) & (freqs < hi)
        out[k] = np.sqrt((S[m] ** 2).sum(axis=0)) if m.any() else np.zeros(S.shape[1])
    return out


def run(store, force=False):
    params = {"sr": SR, "hop": HOP, "bands": BANDS}
    if not force and store.fresh("audio", VERSION, params):
        return store.get("audio"), True
    t_start = time.time()
    pr = store.get("probe")
    if not pr["audio"]:
        out = {"present": False, "summary_line": "no audio stream", "onsets": [], "warnings": []}
        return store.put("audio", out, VERSION, params, deps=[("probe",)], t_start=t_start), False
    import librosa
    wav = media.extract_audio(store.video, store.file("audio.wav"), SR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(str(wav), sr=SR, mono=True)
    offset = float(pr["audio"]["offset_s"]["v"])
    dur = len(y) / SR
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=HOP, window="hann"))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=1024)
    t = librosa.frames_to_time(np.arange(S.shape[1]), sr=SR, hop_length=HOP, n_fft=1024) + offset
    rms = librosa.feature.rms(S=S, frame_length=1024, hop_length=HOP)[0]
    env = _band_env(S, freqs)

    # onset strength: positive rise of log band energy, summed over bands
    flux = np.zeros(S.shape[1])
    for k, e in env.items():
        le = np.log1p(40 * e)
        flux += np.maximum(0, np.diff(le, prepend=le[0]))
    flux = np.convolve(flux, np.ones(3) / 3, mode="same")
    peaks = librosa.util.peak_pick(flux, pre_max=int(0.03 * SR / HOP), post_max=int(0.03 * SR / HOP),
                                   pre_avg=int(0.1 * SR / HOP), post_avg=int(0.1 * SR / HOP),
                                   delta=max(0.15, 0.25 * float(np.percentile(flux, 99))),
                                   wait=int(0.06 * SR / HOP))
    onsets = []
    for i in peaks:
        # walk back to where the rise starts (steepest-rise start), so the time is the attack, not the max
        j = i
        floor = 0.2 * flux[i]
        while j > 0 and flux[j - 1] > floor and flux[j - 1] <= flux[j] + 1e-9:
            j -= 1
        seg = slice(max(0, i - 2), min(len(t), i + int(0.05 * SR / HOP)))
        e = {k: float(v[seg].max()) for k, v in env.items()}
        tot = sum(e.values()) or 1.0
        dom = max(e, key=e.get)
        onsets.append({"t": float(t[j]), "strength": float(flux[i]), "level": float(rms[seg].max()),
                       "band": dom, "band_share": {k: round(v / tot, 2) for k, v in e.items()}})
    # secondary onsets: weak rises within 0.25 s after a much stronger one (decay ripple, not a new hit)
    for k, o in enumerate(onsets):
        for q in onsets[:k]:
            if 0 < o["t"] - q["t"] < 0.25 and o["strength"] < 0.4 * q["strength"]:
                o["secondary_of"] = q["t"]
                break
    primary = [o for o in onsets if "secondary_of" not in o]
    times = np.array([o["t"] for o in primary])

    tempo = None
    if len(times) >= 4:
        ioi = np.diff(times)
        # robust period: median of intervals within 25% of the median (ignores gaps and extra hits)
        med = np.median(ioi)
        core = ioi[(ioi > 0.75 * med) & (ioi < 1.25 * med)]
        if len(core) >= 3:
            per = float(np.median(core))
            tempo = {"bpm": D(60.0 / per, round(min(0.9, len(core) / max(1, len(ioi))), 2), "bpm",
                              f"median of {len(core)} of {len(ioi)} inter-onset intervals"),
                     "period_s": per, "jitter_ms": float(np.std(core) * 1000)}
    silence = float(np.mean(rms < 0.002))
    # envelope for plotting/queries: 50 Hz, rms + bands
    step = max(1, int(round(SR / HOP / 50)))
    np.savez_compressed(store.file("signals_audio.npz"), t=t[::step], rms=rms[::step],
                        low=env["low"][::step], mid=env["mid"][::step], high=env["high"][::step])
    peak_t = float(t[int(np.argmax(rms))])
    sync = _sync(store, times)
    bands = {b: sum(1 for o in primary if o["band"] == b) for b in BANDS}
    n_sec = len(onsets) - len(primary)
    out = {
        "present": True, "duration": M(dur, 1.0, "s"), "offset_applied_s": offset,
        "onsets": onsets, "tempo": tempo,
        "loudness": {"peak_rms": float(rms.max()), "peak_t": peak_t, "silent_frac": silence},
        "sync": sync, "envelope_file": "signals_audio.npz",
        "summary_line": f"{len(primary)} onsets ({', '.join(f'{n} {b}' for b, n in bands.items() if n)})"
                        + (f" +{n_sec} secondary" if n_sec else "")
                        + (f", ~{tempo['bpm']['v']:.0f} bpm (conf {tempo['bpm']['conf']})" if tempo else ", no steady tempo")
                        + (f", {len(sync)} linked to video events" if sync else ""),
        "warnings": ([f"audio {dur:.2f}s vs video {pr['video']['duration']['v']:.2f}s"]
                     if abs(dur + offset - pr["video"]["duration"]["v"]) > 0.25 else []),
    }
    return store.put("audio", out, VERSION, params, deps=[("probe",), ("scenes",)], t_start=t_start), False


def _sync(store, onset_t, tol=0.05):
    """Video events (cuts, flashes, object-motion starts) with an audio onset within +-tol seconds."""
    sc = store.get("scenes")
    events = [(e["type"], e["t"]) for e in sc["events"] if e["type"] in ("cut", "flash")]
    mo = store.get("motion")
    if mo:
        events += [("motion_start", s["t0"]) for s in mo["object_segments"]]
    out = []
    for kind, te in events:
        if len(onset_t):
            j = int(np.argmin(np.abs(onset_t - te)))
            if abs(onset_t[j] - te) <= tol:
                out.append({"video_event": kind, "t_video": te, "t_audio": float(onset_t[j]),
                            "lag_ms": float((onset_t[j] - te) * 1000)})
    return out
