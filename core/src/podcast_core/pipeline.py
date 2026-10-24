#!/usr/bin/env python3
"""podcast_core.pipeline — two tracks (each host recorded locally) → finished episode.

Stages (each writes its result into the episode folder; a rerun reuses what is already there):
  1. analyze      sync: offset + clock drift of the second device                             → sync.json
                  * shared sound in the tracks (voice leaking from a speaker) — GCC-PHAT, sample-accurate;
                  * otherwise — by turn-taking: "rarely talk at once" + "reply right after the other",
                    reply gaps are balanced (half the call latency). Manual correction: sync["b_delay"].
  2. transcribe   stereo (A left, B right) to AssemblyAI multichannel + own word replacements → transcript.json
  3. make_plan    LLM (OpenRouter): start/end by words, title, description, chapters         → plan.json
  4. measure_levels  loudness and gate threshold of each track over the episode, EQ included → levels.json
  5. render(preview=True)   intro + first minute — check sync, loudness, effects            → preview/
     render()               full episode: MP3 with chapters, caption/description, transcript txt/srt → out/

Processing (render): B aligned to A → trim → high-pass < 70 Hz → denoise → loudness → gate →
voice EQ (track_eq) → mix → compressor → −16 LUFS → intro/outro → limiter (4× rate) → MP3 192k.

Time is always seconds on track A ("raw" timeline). B ↔ A: t_B = offset + (1+drift)·t_A.
Settings: DEFAULTS below ← settings.json (shared) ← <episode folder>/settings.json (per-episode edits).
Shared settings.json and intro/outro files next to the app — configure(app_dir).

CLI:
  python -m podcast_core.pipeline run A.m4a B.m4a --names Ann Bob [--work DIR] [--title ...]   full build, no preview
  python -m podcast_core.pipeline render DIR [--start 1:23] [--end 58:10] [--chapters chapters.txt]
  python -m podcast_core.pipeline analyze A.m4a B.m4a                                           sync only
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from . import locale_ru

SR = 48000          # render sample rate
ASR = 8000          # analysis sample rate
HOP = 0.01          # VAD frame, s
FR = int(ASR * HOP)

DEFAULTS = {
    "target_lufs": -16.0,      # final loudness (stereo podcast: Apple/Spotify)
    "track_lufs": -20.0,       # each track before mixing
    "music_lufs": -17.0,       # intro/outro
    "true_peak_db": -1.5,
    "highpass_hz": 70,
    "denoise": True,
    "denoise_nr": 12,
    "gate": True,
    "gate_range_db": -14,
    "eq": True,                # equalizer on/off
    # microphone profiles: name → ffmpeg filter chain; "auto" — match EQ to the common tonal balance
    "mic_profiles": {"neutral": ""},
    "track_profile": [],       # profile of each track (order as in meta.tracks); empty — track_eq below
    "deesser": False,          # de-esser after voice EQ (sibilant s, sh, ch)
    "dynaudnorm": False,       # level riding over the episode (~4 s window) — when a host drifts from the mic
    "tmp_codec": "wavpack",    # intermediate tracks: wavpack (lossless float, ~2× smaller) or pcm_f32le
    # per-track EQ (before mixing), order as in meta.tracks; empty — common EQ after mixing:
    # −2.5 dB 250 Hz, +2 dB 3.5 kHz, +1.5 dB shelf 10 kHz
    "track_eq": [],
    "comp_threshold_db": -10,  # gentle: only acts on loud passages
    "comp_ratio": 3,
    "intro_overlap": 0.5,      # s, how far speech (with pre_roll silence) starts under the intro tail
    "intro_fade": 1.0,         # fade-out of the intro end (jingles often stop abruptly)
    "outro_overlap": 1.0,
    "pre_roll": 0.4,           # silence before the first phrase
    "post_roll": 0.8,
    "bitrate": "192k",         # MP3 CBR 44.1 kHz stereo; 128k — Apple minimum, 192k — headroom for jingle music
    "language": "en",
    "llm_model": "claude-sonnet-5",                     # via AssemblyAI LLM Gateway when there is no OpenRouter key
    "openrouter_model": "deepseek/deepseek-v4.1-flash", # when OPENROUTER_API_KEY / secrets/openrouter_key is set
    "openrouter_fallbacks": ["deepseek/deepseek-v4-flash"],   # if the main model did not answer
    "assets": "",              # intro/outro/cover folder (intro.*, outro.*, cover.*); configure() sets <app_dir>/assets
    "podcast_name": "",
}


def _log(msg):
    print(time.strftime("%H:%M:%S"), msg, file=sys.stderr, flush=True)   # stdout stays clean for JSON


log = _log


# ───────────────────────── utilities ─────────────────────────

def ff(args, quiet=True, on_frac=None, total=None):
    """ffmpeg with error check; returns stderr (loudnorm/ebur128 output is there).
    on_frac(0..1) — progress by output position relative to total seconds."""
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-y"] + (["-v", "info"] if not quiet else []) + [str(a) for a in args]
    if on_frac and total:
        cmd[4:4] = ["-progress", "pipe:1", "-nostats"]
        with tempfile.TemporaryFile("w+") as err:     # stderr to a file: a pipe would fill up and stall
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, text=True)
            for line in p.stdout:
                if line.startswith("out_time_us="):
                    try:
                        on_frac(min(1.0, max(0.0, int(line.split("=")[1]) / 1e6 / total)))
                    except ValueError:
                        pass
            p.wait()
            err.seek(0)
            e = err.read()
        if p.returncode:
            raise RuntimeError("ffmpeg: " + e[-1500:])
        return e
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError("ffmpeg: " + p.stderr[-1500:])
    return p.stderr


def duration(path):
    """File duration in seconds (ffprobe)."""
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, check=True)
    return float(p.stdout.strip())


def db2lin(db):
    return 10 ** (db / 20)


def fmt_chap(t):
    """Timestamp for show notes: 08:37 under an hour, 1:02:13 above."""
    t = max(0, int(round(t)))
    return f"{t // 3600}:{t // 60 % 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60:02d}:{t % 60:02d}"


def fmt_ts(t, force_h=False):
    """Seconds → "mm:ss" or "hh:mm:ss" (reports and transcripts; for show notes — fmt_chap)."""
    t = max(0, int(round(t)))
    h, m, s = t // 3600, t // 60 % 60, t % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if (h or force_h) else f"{m:02d}:{s:02d}"


def parse_ts(s):
    """"1:02:13", "62:13", "95.5", "-0:30", "105s", "1:45,5" → seconds (sign supported).
    LLMs sometimes return time as a string with units — the extra is dropped."""
    orig = s
    s = re.sub(r"[^\d:.,+-]", "", str(s).strip()).replace(",", ".")
    if not re.search(r"\d", s):
        raise ValueError(f"bad time {orig!r}: use seconds or mm:ss / hh:mm:ss")
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        return float(s)
    sign = -1 if s.startswith("-") else 1
    v = 0.0
    for p in s.lstrip("+-").split(":"):
        v = v * 60 + float(p)
    return sign * v


def jload(p, default=None):
    """JSON file → object; default if the file is missing."""
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else default


def jsave(p, obj):
    """Atomic JSON write (.tmp + rename): state is not corrupted if the process dies mid-write."""
    tmp = Path(str(p) + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    tmp.replace(p)


APP_DIR = None     # app folder: <APP_DIR>/settings.json and <APP_DIR>/assets (configure)
APP_TITLE = "podcast-core"   # app name in OpenRouter requests (x-title)
LANG = "en"        # language of user-facing strings (progress, chapter names, file names): en | ru
LOCALES = {"ru": locale_ru.TEXT}
AUTO = ("auto", "авто")   # profile name for auto EQ (the Russian name is kept for existing settings)


def _t(s, **kw):
    """User-facing string in LANG (keys are English); format(**kw)."""
    return LOCALES.get(LANG, {}).get(s, s).format(**kw)


def configure(app_dir, title=None, lang=None):
    """Bind to the app folder (shared settings.json and assets next to it), x-title name and UI language."""
    global APP_DIR, APP_TITLE, LANG
    APP_DIR = Path(app_dir)
    APP_TITLE = title or APP_TITLE
    if lang:
        LANG = DEFAULTS["language"] = lang
        DEFAULTS["mic_profiles"] = {_t("neutral"): ""}
    DEFAULTS["assets"] = str(APP_DIR / "assets")


def load_cfg(work=None):
    """Settings: DEFAULTS ← <APP_DIR>/settings.json ← $PODCAST_SETTINGS (shared)
    ← <work>/settings.json (per-episode edits: EQ, compressor).
    """
    cfg = dict(DEFAULTS)
    for p in [*([APP_DIR / "settings.json"] if APP_DIR else []), Path(os.environ.get("PODCAST_SETTINGS", "/nonexistent"))]:
        cfg.update(jload(p, {}))
    if work:
        cfg.update(jload(Path(work) / "settings.json", {}))
    return cfg


# ───────────────────────── analysis / sync ─────────────────────────

def load_analysis(path):
    """File → mono float32 8 kHz for analysis (sync, VAD, gate thresholds). Whole file in memory: 1 h ≈ 115 MB."""
    p = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-ac", "1", "-ar", str(ASR),
                        "-f", "s16le", "-"], capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.int16).astype(np.float32) / 32768


def frame_db(x):
    """Per-frame loudness (HOP = 10 ms), dBFS. Basis of VAD and gate thresholds."""
    n = len(x) // FR
    f = x[: n * FR].reshape(n, FR)
    return 20 * np.log10(np.sqrt(np.mean(f * f, axis=1)) + 1e-7)


def runs(v):
    """Bool array → (starts, ends) of contiguous True runs; end index is exclusive."""
    d = np.diff(np.r_[0, v.astype(np.int8), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def vad(db):
    """Simple loudness-based speech detector: threshold between noise (10th percentile) and speech (97th),
    pauses < 0.25 s merged, clicks < 0.12 s dropped. → (frame mask, stats).
    """
    nf, sp = np.percentile(db, 10), np.percentile(db, 97)
    thr = max(nf + 0.4 * (sp - nf), nf + 6)
    v = db > thr
    s, e = runs(v)
    for i in range(len(s) - 1):           # merge pauses < 0.25 s
        if s[i + 1] - e[i] < 25:
            v[e[i]: s[i + 1]] = True
    s, e = runs(v)
    for a, b in zip(s, e):                # drop clicks < 0.12 s
        if b - a < 12:
            v[a:b] = False
    return v, {"noise_db": float(nf), "speech_db": float(sp), "thr_db": float(thr), "talk": float(v.mean())}


def xcorr(a, b, lags):
    """c[L] = Σ a[t]·b[t+L] for the given lags."""
    nfft = 1 << int(math.ceil(math.log2(len(a) + len(b))))
    c = np.fft.irfft(np.conj(np.fft.rfft(a, nfft)) * np.fft.rfft(b, nfft), nfft)
    return c[np.asarray(lags) % nfft]


def overlap_ratio(va, vb, lags):
    """How much more/less often A and B talk at once than by chance. The minimum is the right offset."""
    a, b = va.astype(np.float64), vb.astype(np.float64)
    lags = np.asarray(lags)
    O = xcorr(a, b, lags)
    ca, cb = np.r_[0, np.cumsum(a)], np.r_[0, np.cumsum(b)]
    nA, nB = len(a), len(b)
    t0, t1 = np.maximum(0, -lags), np.minimum(nA, nB - lags)
    n = np.maximum(t1 - t0, 0)
    SA = ca[np.clip(t1, 0, nA)] - ca[np.clip(t0, 0, nA)]
    SB = cb[np.clip(t1 + lags, 0, nB)] - cb[np.clip(t0 + lags, 0, nB)]
    E = SA * SB / np.maximum(n, 1)
    ok = (n >= 0.3 * min(nA, nB)) & (E > 1)
    return np.where(ok, O / np.maximum(E, 1e-9), np.nan)


def gap_balance(va, vb, lag):
    """Offset refinement: the "A answers B" and "B answers A" gaps should be equal on average
    (call latency is split in half). lag in frames, B[j] ↔ A[j-lag]."""
    sa, ea = runs(va)
    sb, eb = runs(vb)
    sb, eb = sb - lag, eb - lag

    def gaps(on, off_other):
        off_other = np.sort(off_other)
        idx = np.searchsorted(off_other, on + 50, side="right") - 1
        ok = idx >= 0
        g = on[ok] - off_other[idx[ok]]
        return g[(g > -50) & (g < 250)]

    ga, gb = gaps(sa, eb), gaps(sb, ea)
    if len(ga) < 15 or len(gb) < 15:
        return float(lag), 0
    iqm = lambda g: float(np.mean(np.sort(g)[len(g) // 4: len(g) - len(g) // 4]))   # interquartile mean — robust, not quantized
    d = (iqm(ga) - iqm(gb)) / 2
    return float(lag - d), min(len(ga), len(gb))


def turn_score(va, vb, lags):
    """Dialogue coherence: one host's turn starts within 1 s after the other's ends.
    More such pairs at offset L — better. Sharper than overlap; no false minima ~1 s away."""
    def imp(v, on):
        x = np.zeros(len(v))
        idx = runs(v)[0 if on else 1]
        x[idx[idx < len(v)]] = 1
        return x
    k = np.ones(100)
    offB = np.convolve(imp(vb, False), k)[: len(vb)]
    offA = np.convolve(imp(va, False), k)[: len(va)]
    return xcorr(imp(va, True), offB, lags) + xcorr(offA, imp(vb, True), lags)


def zs(x):
    """Median-based z-score (robust to outliers); NaNs ignored."""
    x = np.asarray(x, float)
    return (x - np.nanmedian(x)) / (np.nanstd(x) + 1e-12)


def plateau_center(r, lags):
    """For overlap_ratio the right offset is not a point but a "plateau" as wide as a typical pause.
    Take the plateau center around the minimum. → (lag, confidence in MAD units, minimum value).
    """
    i = int(np.nanargmin(r))
    rmin, med = r[i], np.nanmedian(r)
    lim = rmin + 0.15 * (med - rmin)
    lo = hi = i
    while lo > 0 and r[lo - 1] <= lim:
        lo -= 1
    while hi < len(r) - 1 and r[hi + 1] <= lim:
        hi += 1
    mad = np.nanmedian(np.abs(r - med)) * 1.4826 + 1e-9
    return (lags[lo] + lags[hi]) / 2, float((med - rmin) / mad), float(rmin)


def gcc_phat(a, b, t0, t1, lo, hi):
    """Find window A[t0,t1) (s) in B at offset L∈[lo,hi]: B(t+L)=A(t). → (L, peak z-score)."""
    i0, i1 = int(t0 * ASR), int(t1 * ASR)
    j0, j1 = int((t0 + lo) * ASR), int((t1 + hi) * ASR)
    j0c, j1c = max(0, j0), min(len(b), j1)
    if i1 > len(a) or j1c - j0c < (i1 - i0):
        return None, 0.0
    wa, wb = a[i0:i1], b[j0c:j1c]
    nfft = 1 << int(math.ceil(math.log2(len(wa) + len(wb))))
    X = np.conj(np.fft.rfft(wa, nfft)) * np.fft.rfft(wb, nfft)
    c = np.fft.irfft(X / (np.abs(X) + 1e-9), nfft)
    m = len(wb) - len(wa) + 1                 # valid offsets inside wb
    if m <= 1:
        return None, 0.0
    c = c[:m]
    k = int(np.argmax(c))
    z = float((c[k] - c.mean()) / (c.std() + 1e-12))
    return (j0c + k - i0) / ASR, z


def theil_sen(x, y):
    """Robust linear regression (median of all pairwise slopes). For clock drift: one or two windows
    with a wrong offset do not spoil the estimate. → (slope, intercept).
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return 0.0, float(np.median(y)) if len(y) else 0.0
    i, j = np.triu_indices(len(x), 1)
    dx = x[j] - x[i]
    ok = dx > 0
    s = float(np.median((y[j] - y[i])[ok] / dx[ok]))
    return s, float(np.median(y - s * x))


# ───────────────────────── sync signal ─────────────────────────
# A short chirp played by the sync-signal app at a precisely known moment (server clock).
# Found in both tracks, it gives sync as accurate as the phones' clocks (~10–30 ms).
CHIRP = {"f0": 800.0, "f1": 3200.0, "dur": 0.6}
CSR = 16000                                   # sample rate for signal search (chirp up to 3.2 kHz)


def chirp_wave(sr, c=CHIRP):
    """The signal itself: linear sweep f0→f1 with a Hann window. The sync web page generates the same signal."""
    t = np.arange(int(c["dur"] * sr)) / sr
    ph = 2 * np.pi * (c["f0"] * t + (c["f1"] - c["f0"]) * t * t / (2 * c["dur"]))
    return (np.sin(ph) * np.hanning(len(t))).astype(np.float32)


def find_chirps(path, top=8, min_ncc=0.25):
    """Where the sync signal sounds in a file: [(start second, NCC 0..1)], best first.
    Normalized cross-correlation in 60 s chunks; audio is streamed from ffmpeg, so memory stays
    constant (a few MB) for any recording length. Speech gives NCC ≲ 0.15, the signal 0.9+."""
    ref = chirp_wave(CSR)
    m = len(ref)
    ref_e = float(np.sqrt(np.sum(ref * ref)))
    step = 60 * CSR
    nfft = 1 << int(math.ceil(math.log2(step + m)))
    R = np.conj(np.fft.rfft(ref, nfft))
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-ac", "1", "-ar", str(CSR),
                             "-f", "f32le", "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    hits, buf, pos0, carry, eof = [], np.zeros(0, np.float32), 0, b"", False
    while not eof or len(buf) >= m:
        if not eof and len(buf) < step + m - 1:              # read up to a full chunk
            raw = proc.stdout.read(4 * (step + m - 1 - len(buf)))
            if not raw:
                eof = True
            else:
                raw = carry + raw
                cut = len(raw) // 4 * 4
                carry = raw[cut:]
                buf = np.concatenate([buf, np.frombuffer(raw[:cut], np.float32)])
                continue
        seg = buf[: step + m - 1]
        if len(seg) < m:
            break
        c = np.fft.irfft(np.fft.rfft(seg, nfft) * R, nfft)[: len(seg) - m + 1]
        cs = np.concatenate([[0.0], np.cumsum(seg.astype(np.float64) ** 2)])
        e = np.sqrt(np.maximum(cs[m:] - cs[:-m], 1e-12))[: len(c)]
        ncc = c / (e * ref_e)
        k = int(np.argmax(ncc))
        while ncc[k] >= min_ncc:                    # all distinct peaks of the chunk (1 s dead zone)
            hits.append(((pos0 + k) / CSR, float(ncc[k])))
            ncc[max(0, k - CSR): k + CSR] = 0
            k = int(np.argmax(ncc))
        buf, pos0 = buf[step:], pos0 + step
    proc.wait()
    hits.sort(key=lambda h: -h[1])
    out = []                                        # duplicates at chunk borders — keep the strongest
    for h in hits:
        if all(abs(h[0] - o[0]) > 1.0 for o in out):
            out.append(h)
    return out[:top]


def match_marks(hits, marks, tol=0.5):
    """Match found signals to button-press moments (server time). → [(T, position in file)].
    One signal — the strongest; several — the set whose gaps match the gaps between presses."""
    marks = sorted(marks)
    if not hits or not marks:
        return []
    if len(marks) == 1:
        return [(marks[0], hits[0][0])]
    best, best_score = [], -1.0
    pos = sorted(h[0] for h in hits)
    for anchor in pos:                          # assume anchor is the first signal
        got, score = [(marks[0], anchor)], 0.0
        for T in marks[1:]:
            want = anchor + (T - marks[0])
            near = [p for p in pos if abs(p - want) <= tol]
            if near:
                got.append((T, min(near, key=lambda p: abs(p - want))))
                score += 1
        if score > best_score:
            best, best_score = got, score
    return best


def signal_sync(a_path, b_path, marks):
    """Sync by sync signals. marks = [[T…] of host A, [T…] of host B] (seconds, server clock).
    Model: track time = rate · (T − zero). → dict(offset, drift, …) or None if no signal was found."""
    out = []
    for path, mk in zip((a_path, b_path), marks):
        if not mk:
            return None
        hits = find_chirps(path)
        pairs = match_marks(hits, mk)
        log(f"sync signal: {Path(path).name}: presses {len(mk)}, found {len(hits)}, matched {len(pairs)}"
            + (f", best NCC {hits[0][1]:.2f}" if hits else ""))
        if not pairs:
            return None
        if len(pairs) >= 2:
            (T1, p1), (T2, p2) = pairs[0], pairs[-1]
            rate = (p2 - p1) / (T2 - T1) if T2 - T1 > 60 else 1.0
        else:
            rate = 1.0
        zero = float(np.median([T - p / rate for T, p in pairs]))
        out.append({"rate": rate, "zero": zero, "pairs": pairs, "ncc": hits[0][1]})
    a, b = out
    drift = b["rate"] / a["rate"] - 1
    offset = b["rate"] * (a["zero"] - b["zero"])
    return {"offset": offset, "drift": drift if abs(drift) < 3e-4 else 0.0,
            "has_drift": min(len(a["pairs"]), len(b["pairs"])) >= 2,
            "pairs": [a["pairs"], b["pairs"]], "ncc": [a["ncc"], b["ncc"]]}


def blind_signal_sync(a_path, b_path, min_ncc=0.3, tol=0.25):
    """Sync signal without timestamps: chirps are searched over both whole tracks, paired by equal gaps.
    Needs ≥ 2 matching signals (a single one may be a random peak). Model t_B = offset + (1+drift)·t_A.
    Threshold 0.3: speech gives NCC ≲ 0.15, while a neighbour phone's speaker caught by a recorder gives only 0.4–0.5.
    → dict like signal_sync (pairs: [[(t_A, t_A)…], [(t_A, t_B)…]]) or None."""
    ha = sorted(t for t, c in find_chirps(a_path, top=20, min_ncc=min_ncc))
    hb = sorted(t for t, c in find_chirps(b_path, top=20, min_ncc=min_ncc))
    log(f"blind sync signal: found {len(ha)} in A, {len(hb)} in B (NCC ≥ {min_ncc})")
    best = []
    for a0 in ha:                                   # hypothesis: a0 and b0 are the same signal
        for b0 in hb:
            got = []
            for ta in ha:
                want = b0 + (ta - a0)
                near = [tb for tb in hb if abs(tb - want) <= tol + 3e-4 * abs(ta - a0)]
                if near:
                    got.append((ta, min(near, key=lambda tb: abs(tb - want))))
            if len(got) > len(best):
                best = got
    if len(best) < 2:
        return None
    x, y = np.array([p[0] for p in best]), np.array([p[1] for p in best])
    span = x.max() - x.min()
    slope = float(np.polyfit(x, y, 1)[0]) if span > 60 else 1.0
    drift = slope - 1 if abs(slope - 1) < 3e-4 else 0.0
    offset = float(np.median(y - (1 + drift) * x))
    return {"offset": offset, "drift": drift, "has_drift": span > 60, "blind": True,
            "pairs": [[(a, a) for a, _ in best], [(a, b) for a, b in best]], "ncc": [min_ncc, min_ncc]}


def signal_gaps_agree(sig, tol=0.05, max_ppm=100):
    """Do the gaps between signals agree in A and B (|gapA − gapB| < tol + drift ≤ max_ppm)? → (yes/no, reason)."""
    pa, pb = [p for _, p in sig["pairs"][0]], [p for _, p in sig["pairs"][1]]
    if len(pa) < 2 or len(pa) != len(pb):
        return False, "fewer than two signals in the pair"
    raw, span = 0.0, 0.0
    for i in range(1, len(pa)):
        ga, gb = pa[i] - pa[0], pb[i] - pb[0]
        if abs(gb - ga) - max_ppm * 1e-6 * abs(ga) >= tol:
            return False, f"gaps differ by {abs(gb - ga) * 1000:.0f} ms over {ga:.0f} s"
        raw, span = max(raw, abs(gb - ga)), max(span, abs(ga))
    return True, (f"gaps between {len(pa)} signals agree: difference {raw * 1000:.0f} ms over {span:.0f} s "
                  f"(tolerance {tol * 1000:.0f} ms + drift up to {max_ppm} ppm = {(tol + max_ppm * 1e-6 * span) * 1000:.0f} ms)")


def analyze(a_path, b_path, max_offset=600, marks=None, marks_meta=None, blind=False, trust_signal=False):
    """Sync two tracks → sync.json.

    1) Shared sound (the other host's voice leaks into the mic)? GCC-PHAT over 3 windows: if ≥2 windows
       agree (spread < 50 ms) and the peak is distinct (z > 12) — "bleed" mode, sample-accurate.
    2) Otherwise "turns" mode (remote recording on headphones): by VAD find the offset at which
       a) they talk at once least often (overlap_ratio) and b) one host's turn most often starts
       right after the other's ends (turn_score). Both cues are summed as z-scores.
       Then gap_balance equalizes the reply gaps both ways (half the call latency).
    3) Clock drift: the same search in 15-minute windows across the recording, Theil–Sen over the points.
       |drift| > 300 ppm is treated as an estimation error and not applied.
    Returns offset/drift (t_B = offset + (1+drift)·t_A), speech segments of both on the A scale, speech edges.
    """
    log("analyze: decoding tracks")
    a, b = load_analysis(a_path), load_analysis(b_path)
    da, db_ = len(a) / ASR, len(b) / ASR
    dbA, dbB = frame_db(a), frame_db(b)
    vA, sA = vad(dbA)
    vB, sB = vad(dbB)
    maxlag = int(min(max_offset, max(da, db_) * 0.6) / HOP)

    # 1) is there shared sound (the other host's voice leaking into the mic)
    bleed = []
    talk = np.flatnonzero(vA)
    for frac in (0.2, 0.5, 0.8):
        t0 = frac * da
        if t0 + 60 > da:
            continue
        L, z = gcc_phat(a, b, t0, t0 + 60, -maxlag * HOP, maxlag * HOP)
        if L is not None:
            bleed.append((L, z))
    good = [L for L, z in bleed if z > 12]
    mode = None
    if len(good) >= 2 and np.ptp(good) < 0.05:
        mode, L0, conf = "bleed", float(np.median(good)), float(min(z for _, z in bleed if z > 12))

    if mode is None:
        lags = np.arange(-maxlag, maxlag + 1)
        r = overlap_ratio(vA, vB, lags)
        score = np.nan_to_num(zs(turn_score(vA, vB, lags)) - zs(r), nan=-1e9)   # both cues must agree
        i = int(np.argmax(score))
        near = np.abs(lags - lags[i]) <= 150                                      # ±1.5 s around the best
        lag, _, rmin = plateau_center(np.where(near, r, np.nan), lags)
        rest = score[np.abs(lags - lags[i]) > 300]
        conf = float((score[i] - np.median(rest)) / (np.max(rest) - np.median(rest) + 1e-9))  # >1.5 — confident
        lag, n = gap_balance(vA, vB, int(round(lag)))
        mode, L0 = "turns", lag * HOP
        log(f"analyze: by turn-taking, offset {L0:+.2f} s, confidence {conf:.1f}, overlap {rmin:.2f}, gaps {n}")

    # 2) clock drift: local offset per window
    pts = []
    if mode == "bleed":
        for t0 in np.arange(30, da - 90, 240):
            L, z = gcc_phat(a, b, t0, t0 + 60, L0 - 3, L0 + 3)
            if L is not None and z > 10:
                pts.append((t0 + 30, L))
    else:
        W, step = int(900 / HOP), int(300 / HOP)
        for s0 in range(0, max(1, len(vA) - W // 2), step):
            wa = vA[s0: s0 + W]
            j0 = s0 + int(round(L0 / HOP)) - 300
            if j0 < 0 or len(wa) < W // 2:
                continue
            wb = vB[j0: j0 + len(wa) + 600]
            if len(wb) < len(wa) or wa.mean() < 0.05 or wb.mean() < 0.05:
                continue
            ll = np.arange(0, 601)
            r = overlap_ratio(wa, wb, ll)
            if np.all(np.isnan(r)):
                continue
            lag, _, _ = plateau_center(r, ll)
            lag, n = gap_balance(wa, vB[j0:], int(round(lag)))
            if n >= 30:
                pts.append(((s0 + len(wa) / 2) * HOP, (j0 - s0 + lag) * HOP))
    s, L = 0.0, L0
    if len(pts) >= 3:
        s, L = theil_sen([p[0] for p in pts], [p[1] for p in pts])
        if abs(s) > 3e-4:                     # >1 s/hour — more likely an estimation error than the clocks
            s, L = 0.0, L0
    by_talk = {"mode": mode, "offset": round(L, 4), "drift_ms_per_hour": round(s * 3600 * 1000, 1)}

    # 3) sync signal if both pressed: more accurate than talk; drift — from two signals or from talk
    sig = signal_sync(a_path, b_path, marks) if marks and all(marks) else None
    if sig is None and blind:                 # no presses — blind signal search (opt-in)
        sig = blind_signal_sync(a_path, b_path)
    if sig:
        # safety net: phone clocks may have been synced over a slow network — compare with talk at the signal
        tA0 = sig["pairs"][0][0][1]
        talk_at = L + s * tA0
        sig_at = sig["offset"] + sig["drift"] * tA0 if sig["has_drift"] else sig["offset"]
        ok, why = signal_gaps_agree(sig) if trust_signal else (False, "")
        if ok:                                # trust_signal: consistent gaps outweigh the talk check
            sig["trust"] = why + f"; {sig_at - talk_at:+.2f} s from talk — check not needed"
            log(f"analyze: trusting the sync signal — {sig['trust']}")
        elif abs(sig_at - talk_at) > 0.3:
            log(f"analyze: sync signal differs from talk by {sig_at - talk_at:+.2f} s — using talk")
            sig = {"rejected": round(sig_at - talk_at, 3), "ncc": sig["ncc"], **({"trust_denied": why} if why else {})}
    if sig and "rejected" not in sig:
        if not sig["has_drift"]:
            t0 = sig["pairs"][0][0][1]                 # A signal position: the offset is right at this point
            sig["drift"] = s
            sig["offset"] = sig["offset"] - s * t0
        mode, L, s, conf = "signal", sig["offset"], sig["drift"], 99.0
        log(f"analyze: by sync signal offset {L:+.3f} s (talk gave {by_talk['offset']:+.3f} s)")
    res = {
        "mode": mode, "offset": round(L, 4), "drift": s, "drift_ms_per_hour": round(s * 3600 * 1000, 1),
        "confidence": round(conf, 1), "points": [[round(t, 1), round(l, 4)] for t, l in pts],
        "dur_a": da, "dur_b": db_, "vad_a": sA, "vad_b": sB, "by_talk": by_talk,
        "signal": (({"pairs": sig["pairs"], "ncc": [round(x, 2) for x in sig["ncc"]], **({"blind": True} if sig.get("blind") else {}),
                     **({"trust": sig["trust"]} if sig.get("trust") else {}),
                     # clock sync quality on each device (from the sync app): worst of the presses, ms
                     "clock_err": [max((m.get("err") or 0) for m in mm) if mm else None for mm in (marks_meta or [[], []])],
                     "net_rtt": [min((m.get("rtt") or 0) for m in mm) if mm else None for mm in (marks_meta or [[], []])]}
                    if "rejected" not in sig else {k: sig[k] for k in ("rejected", "trust_denied") if k in sig}) if sig else None),
    }
    # common timeline and speech edges (seconds of A)
    sa, ea = runs(vA)
    sb, eb = runs(vB)
    to_a = lambda j: (j * HOP - L) / (1 + s)
    starts = [x for x in [sa[0] * HOP if len(sa) else None, to_a(sb[0]) if len(sb) else None] if x is not None]
    ends = [x for x in [ea[-1] * HOP if len(ea) else None, to_a(eb[-1]) if len(eb) else None] if x is not None]
    res["speech_start"] = max(0.0, min(starts)) if starts else 0.0
    res["speech_end"] = min(da, max(ends)) if ends else da
    res["timeline_end"] = min(da, (db_ - L) / (1 + s))
    # speech segments of both tracks (to snap trims to pauses)
    segs = [[round(x * HOP, 2), round(y * HOP, 2)] for x, y in zip(sa, ea)]
    segs += [[round(to_a(x), 2), round(to_a(y), 2)] for x, y in zip(sb, eb)]
    res["segments"] = sorted(segs)
    log(f"analyze: {mode}, B offset {res['offset']:+.3f} s, drift {res['drift_ms_per_hour']:+.0f} ms/h, points {len(pts)}")
    return res


# ───────────────────────── transcription (AssemblyAI) ─────────────────────────

AAI = "https://api.assemblyai.com"


def _http(method, url, key, body=None, ctype="application/json", timeout=600, headers=None):
    """Minimal JSON client on urllib (no dependencies). key → authorization header."""
    data = body if isinstance(body, (bytes, type(None))) else json.dumps(body).encode()
    h = {"content-type": ctype, **({"authorization": key} if key else {}), **(headers or {})}
    req = urllib.request.Request(url, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{url}: HTTP {e.code} {e.read()[:500]!r}")


def make_stereo_preview(a_path, b_path, sync, out):
    """A left, B right, aligned to A — for per-channel transcription."""
    fb = b_align_filter(sync, 22050)
    ff(["-i", a_path, "-i", b_path, "-filter_complex",
        f"[0:a]aresample=22050,aformat=channel_layouts=mono[a];[1:a]{fb}[b];[a][b]join=inputs=2:channel_layout=stereo[o]",
        "-map", "[o]", "-c:a", "libmp3lame", "-b:a", "96k", "-joint_stereo", "0", out])


def transcribe(audio, key, language="ru", extra=None):
    """AssemblyAI: upload, multichannel job (channel 1 = A, channel 2 = B — the speaker is known
    without diarization), polling every 10 s. extra — extra request params from settings (aai_extra).
    NOTE: their custom_spelling only accepts a ONE-word replacement — own replacements go through apply_spelling.
    """
    log("transcribe: uploading to AssemblyAI")
    with open(audio, "rb") as f:
        up = _http("POST", AAI + "/v2/upload", key, f.read(), "application/octet-stream")
    job = _http("POST", AAI + "/v2/transcript", key, {
        "audio_url": up["upload_url"], "multichannel": True, "language_code": language,
        "punctuate": True, "format_text": True, **(extra or {})})
    log(f"transcribe: job {job['id']}")
    while True:
        time.sleep(10)
        t = _http("GET", f"{AAI}/v2/transcript/{job['id']}", key)
        if t["status"] == "completed":
            return t
        if t["status"] == "error":
            raise RuntimeError("AssemblyAI: " + str(t.get("error")))


def apply_spelling(tr, mapping):
    """Own word replacements in the transcript (e.g. a misheard show name → the right one): AssemblyAI replaces single words only."""
    if not mapping:
        return tr
    rx = [(re.compile(rf"(?<!\w){re.escape(a)}(?!\w)", re.I), b) for a, b in mapping.items()]

    def fix(t):
        for r, b in rx:
            t = r.sub(b, t)
        return t
    for w in tr.get("words") or []:
        w["text"] = fix(w["text"])
    for u in tr.get("utterances") or []:
        u["text"] = fix(u["text"])
        for w in u.get("words") or []:
            w["text"] = fix(w["text"])
    if tr.get("text"):
        tr["text"] = fix(tr["text"])
    return tr


def utterances(transcript, names):
    """[(start, end, name, text, words)] in seconds of A; adjacent turns of the same person merged."""
    out = []
    for u in transcript.get("utterances") or []:
        ch = int(str(u.get("channel") or u.get("speaker") or "1")[0]) - 1
        name = names[ch] if 0 <= ch < len(names) else _t("Voice {n}", n=ch + 1)
        out.append([u["start"] / 1000, u["end"] / 1000, name, u["text"].strip(), u.get("words") or []])
    out.sort(key=lambda x: x[0])
    merged = []
    for u in out:
        if merged and merged[-1][2] == u[2] and u[0] - merged[-1][1] < 1.5:
            m = merged[-1]
            m[1], m[3], m[4] = u[1], m[3] + " " + u[3], m[4] + u[4]
        else:
            merged.append(u)
    return merged


# ───────────────────────── plan: start/end/chapters (LLM) ─────────────────────────

PROMPT = """Below is the transcript of a conversational podcast recording (two hosts{name_hint}). Times are seconds from the start of the recording.
The first and last 10 minutes are given by sentences (exact time of each), the middle by turns.

Return ONLY JSON:
{{
 "start": <second where the episode starts>,
 "start_phrase": "<first 3–6 words of the episode exactly as in the transcript>",
 "start_reason": "<one sentence: why here, what is cut before it>",
 "end": <second where the episode ends>,
 "end_phrase": "<last 3–6 words of the episode exactly as in the transcript>",
 "end_reason": "<one sentence: why here>",
 "title": "<episode title — see TITLE rules below>",
 "description": "<episode description for podcast platforms, 2–4 sentences, no filler, no emoji>",
 "chapters": [{{"start": <second>, "title": "<topic, 2–6 words>"}}, ...]
}}

HOW TO FIND THE REAL START. The listener presses play, hears the intro and must hear the start right away — no setup.
- Recordings almost always begin with setup: "is it recording?", "can you hear me?", claps and counting for sync, discussing the format ("where do we start?", "let's think of an opening"), private jokes, "wait", "let's do it again".
- There may be SEVERAL greetings. A trial or failed greeting is one followed by more setup: discussing how to start, laughing at the take, "wait", "once more", counting "one, two, three". The real start is the LAST greeting after which the hosts run the episode and never return to setup.
- Countdowns and cues before the greeting ("ok, let's go", "one, two, three", "rolling") are NOT part of the episode: start is exactly on the first word of the greeting.
- If there is no greeting at all — start at the first phrase addressed to listeners or introducing the topic.

HOW TO FIND THE REAL END.
- The end is the last goodbye to listeners ("bye everyone", "see you", "until next episode"); end is on the last word of the goodbye.
- Everything after the goodbye not addressed to listeners ("you can stop it", "was that ok?", "stop recording") is cut.
- If after a goodbye the talk for listeners resumes and there is a second goodbye — the end is at the last goodbye.
- If there is no goodbye — end at the last meaningful phrase.

TITLE (no number — it is added automatically: "#12: …"):
- ONE hook, not a list of topics: the main topic, the funniest or most unexpected idea of the episode, or its mood;
- 2–4 words, lowercase (except names and abbreviations), no trailing period; a question is fine;
- do NOT list topics with commas or "and" ("coffee, sleep and moving" is bad: that is a table of contents, chapters do that);
- good: "a talk about sleep", "summer episode", "who are adults?", "a year without social media".{title_examples}

Chapters (timestamps for listeners):
- at most 20, spread evenly; for episodes over 40 minutes — sparser, about every 3–6 minutes; the first one at "start";
- short (2–7 words), catchy and clear, lowercase (except abbreviations and names), capturing the topic at that moment;
- good: "intro", "meeting the guest", "talking about AI", "a funny childhood story", "AI and the future", "goodbyes";
- bad: "A very interesting moment of the podcast" (too generic), "The guest tells a story" (uninformative), "PODCAST START" (caps).

Transcript:
{text}"""
PROMPTS = {"en": PROMPT, "ru": locale_ru.PROMPT}
TITLE_EXAMPLES = {"en": '\n- titles of previous episodes of this podcast — keep their style and length: "{examples}"',
                  "ru": locale_ru.TITLE_EXAMPLES}
TITLE_EXAMPLES_JOIN = {"en": '", "', "ru": locale_ru.TITLE_EXAMPLES_JOIN}


def openrouter_key():
    """OpenRouter key: $OPENROUTER_API_KEY or file $OPENROUTER_KEY_FILE (default /secrets/openrouter_key)."""
    f = Path(os.environ.get("OPENROUTER_KEY_FILE", "/secrets/openrouter_key"))
    return os.environ.get("OPENROUTER_API_KEY") or (f.read_text().strip() if f.exists() else "")


def _openrouter(prompt, model, okey, attempts=2):
    """One OpenRouter request with retries; "thinking" models sometimes return empty content."""
    err = None
    for attempt in range(attempts):
        try:
            r = _http("POST", "https://openrouter.ai/api/v1/chat/completions", None, {
                "model": model, "max_tokens": 32000, "response_format": {"type": "json_object"},
                "reasoning": {"effort": "low"},
                "messages": [{"role": "user", "content": prompt}]}, timeout=600,
                headers={"authorization": f"Bearer {okey}", "x-title": APP_TITLE})
            if "error" in r:
                raise RuntimeError(f"OpenRouter: {r['error']}")
            c = r["choices"][0]
            text = (c["message"].get("content") or "").strip()
            if "{" in text:
                return text
            raise RuntimeError(_t("empty response (finish_reason={reason})", reason=c.get("finish_reason")))
        except Exception as e:  # noqa: BLE001
            err = e
            log(f"LLM {model}, attempt {attempt + 1}: {e}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{model}: {err}")


def llm(prompt, key, cfg):
    """LLM answer down the chain: main OpenRouter model → OpenRouter fallbacks → AssemblyAI LLM Gateway.
    → (text, model). Raises only if nobody answered."""
    errors = []
    okey = openrouter_key()
    if okey:
        for model in [cfg["openrouter_model"]] + list(cfg.get("openrouter_fallbacks") or []):
            try:
                return _openrouter(prompt, model, okey), model
            except Exception as e:  # noqa: BLE001
                errors.append(str(e)[:200])
    if key:
        try:
            r = _http("POST", "https://llm-gateway.assemblyai.com/v1/chat/completions", key, {
                "model": cfg["llm_model"], "max_tokens": 8000,
                "messages": [{"role": "user", "content": prompt}]}, timeout=300)
            text = r["choices"][0]["message"]["content"]
            if "{" in text:
                return text, cfg["llm_model"]
            errors.append(_t("{model}: empty response", model=cfg["llm_model"]))
        except Exception as e:  # noqa: BLE001
            errors.append(f"{cfg['llm_model']}: {str(e)[:200]}")
    raise RuntimeError(_t("LLM did not answer: {errors}", errors=" | ".join(errors)))


def sentences(u):
    """Turn → [(time, text)] by sentences, time of the first word."""
    out, cur = [], []
    for w in u[4]:
        cur.append(w)
        if w["text"].rstrip().endswith((".", "?", "!", "…")):
            out.append((cur[0]["start"] / 1000, " ".join(x["text"] for x in cur)))
            cur = []
    if cur:
        out.append((cur[0]["start"] / 1000, " ".join(x["text"] for x in cur)))
    return out or [(u[0], u[3])]


def transcript_for_llm(us, edge=600):
    """Recording edges (where start and end are) — by sentences with exact time, the middle — by turns."""
    if not us:
        return ""
    end = us[-1][1]
    lines = []
    for u in us:
        if u[0] < edge or u[1] > end - edge:
            lines += [f"[{t:.1f}] {u[2]}: {txt}" for t, txt in sentences(u)]
        else:
            lines.append(f"[{int(u[0])}] {u[2]}: {u[3]}")
    return "\n".join(lines)


def make_plan(work, cfg, key):
    """LLM over the transcript → plan.json: episode start/end, title, description, chapters.

    LLM times are coarse (by sentences), so the LLM also returns start_phrase/end_phrase,
    and the exact time comes from the word-level transcript (find_phrase). Start is pre_roll before the first
    word, but not before the end of the previous word (so a "one, two, three" before "Hi all" is not caught).
    If the LLM did not answer — exception: the build fails and can be retried; no episode without chapters.
    """
    work = Path(work)
    sync, meta = jload(work / "sync.json"), jload(work / "meta.json")
    plan = {"start": sync["speech_start"], "end": sync["speech_end"], "title": meta.get("title") or "",
            "description": "", "chapters": [], "source": "auto"}
    tr = jload(work / "transcript.json")
    if tr and key:
        us = utterances(tr, meta["names"])
        text = transcript_for_llm(us)
        hint = f": {', '.join(meta['names'])}" if meta.get("names") else ""
        lang = cfg["language"] if cfg["language"] in PROMPTS else "en"
        try:
            ex = [t for t in cfg.get("title_examples") or [] if t][-12:]
            ex_txt = TITLE_EXAMPLES[lang].format(examples=TITLE_EXAMPLES_JOIN[lang].join(ex)) if ex else ""
            raw, model = llm(PROMPTS[lang].format(text=text, name_hint=hint, title_examples=ex_txt), key, cfg)
            log(f"plan: model {model} answered")
            plan["llm_model_used"] = model
            j = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
            plan.update({k: j[k] for k in ("start", "end", "start_phrase", "end_phrase", "start_reason", "end_reason",
                                           "title", "description", "chapters") if k in j})
            log(f"plan: start {j.get('start')} \"{j.get('start_phrase')}\" — {j.get('start_reason')}")
            log(f"plan: end {j.get('end')} \"{j.get('end_phrase')}\" — {j.get('end_reason')}")
            plan["source"] = "llm"
        except Exception as e:  # noqa: BLE001
            # an episode without start/end and chapters is no good — let the build fail so it can be retried
            raise RuntimeError(_t("could not find start, end and chapters: {error}", error=e)) from e
    words = sorted((w["start"] / 1000, w["end"] / 1000, w["text"]) for w in (tr or {}).get("words") or [])
    plan["start"], plan["end"] = parse_ts(plan["start"]), parse_ts(plan["end"])   # the LLM may return "01:23"
    ws = find_phrase(words, plan.get("start_phrase"), plan["start"])
    we = find_phrase(words, plan.get("end_phrase"), plan["end"], last=True)
    s0, e0 = snap_trim(sync, plan["start"], plan["end"], cfg)
    if ws is not None:                       # exactly at the word: "Hi all", not "Let's go. Hi all"
        before = [w[1] for w in words if w[1] <= ws + 0.01 and w[0] < ws - 0.01]
        gap = ws - max(before) if before else cfg["pre_roll"] + 1
        s0 = round(max(0.0, ws - min(cfg["pre_roll"], max(0.05, gap - 0.05))), 2)
    if we is not None:
        after = [w[0] for w in words if w[0] >= we - 0.01]
        gap = min(after) - we if after else cfg["post_roll"] + 1
        e0 = round(we + min(cfg["post_roll"], max(0.05, gap - 0.05)), 2)
    plan["start"], plan["end"] = s0, e0
    plan["trim_by"] = _t("word" if ws is not None else "pause") + "/" + _t("word" if we is not None else "pause")
    jsave(work / "plan.json", plan)
    return plan


def _norm(t):
    """Word for comparison: lowercase, ё→е, no punctuation."""
    return re.sub(r"[^\w]+", "", t.lower().replace("ё", "е"))


def find_phrase(words, phrase, near, last=False):
    """Start time (or end, last=True) of a phrase in the word-level transcript; of the hits — the closest to near."""
    if not phrase or not words:
        return None
    toks = [_norm(x) for x in phrase.split() if _norm(x)]
    if not toks:
        return None
    words = [(w[0], w[1], part) for w in words for part in w[2].split()]   # after replacement one word may become two
    norm = [_norm(w[2]) for w in words]
    hits = []
    for i in range(len(words) - len(toks) + 1):
        # words of both channels are interleaved in time — look for consecutive matches allowing one foreign word
        j, k, miss = i, 0, 0
        while j < len(words) and k < len(toks) and miss <= 1:
            if norm[j] == toks[k]:
                k += 1
            elif k:
                miss += 1
            else:
                break
            j += 1
        if k == len(toks):
            hits.append((words[i][0], words[j - 1][1]))
    if not hits:
        return None
    t = min(hits, key=lambda h: abs((h[1] if last else h[0]) - near))
    return t[1] if last else t[0]


def snap_trim(sync, start, end, cfg):
    """Trim at pauses: start before the phrase, end after it."""
    segs = sync.get("segments") or []
    st = [s for s, e in segs if s <= start + 0.5]
    start = (max(st) if st else start) - cfg["pre_roll"]
    en = [e for s, e in segs if e >= end - 0.5]
    end = (min(en) if en else end) + cfg["post_roll"]
    lim = sync.get("timeline_end") or sync["dur_a"]
    start, end = max(0.0, start), min(lim, end)
    if end - start < 10:
        start, end = 0.0, lim
    return round(start, 2), round(end, 2)


# ───────────────────────── render ─────────────────────────

def b_align_filter(sync, sr):
    """Filter that maps B onto the A scale: B'(t) = B(L + (1+s)t)."""
    L, s = sync["offset"] - sync.get("b_delay", 0.0), sync["drift"]   # b_delay > 0 — the second track is later
    f = [f"aresample={SR}", "aformat=sample_fmts=fltp:channel_layouts=mono"]
    f.append(f"atrim=start={L:.4f},asetpts=N/SR/TB" if L >= 0 else f"adelay=delays={-L * 1000:.0f}:all=1")
    if abs(s) * sync["dur_a"] > 0.03:
        hi = SR * 4
        f += [f"aresample={hi}", f"asetrate={round(hi * (1 + s))}"]
    f.append(f"aresample={sr}")
    return ",".join(f)


def loudness(path, extra="", on_frac=None, seek=None):
    """(LUFS, true peak dBTP). ebur128 — same numbers as loudnorm, many times faster. seek=(from, to) — input slice."""
    inp = (["-ss", f"{max(0.0, seek[0]):.3f}", "-to", f"{seek[1]:.3f}"] if seek else []) + ["-i", path]
    total = (seek[1] - max(0.0, seek[0])) if seek else (duration(path) if on_frac else None)
    err = ff(inp + ["-af", (extra + "," if extra else "") + "ebur128=peak=true:framelog=quiet", "-f", "null", "-"],
             quiet=False, on_frac=on_frac, total=total)
    i = re.findall(r"I:\s+(-?[\d.]+|-inf) LUFS", err)
    tp = re.findall(r"Peak:\s+(-?[\d.]+|-inf) dBFS", err)
    to_f = lambda v: -70.0 if "inf" in v else float(v)
    return to_f(i[-1]), to_f(tp[-1]) if tp else 0.0


def find_asset(cfg, name):
    """Intro/outro/cover file: an explicit one from cfg["asset_files"], else intro.*, outro.*, cover.* in assets."""
    f = (cfg.get("asset_files") or {}).get(name)
    if f and Path(f).exists():
        return Path(f)
    if not cfg.get("assets"):
        return None
    d = Path(cfg["assets"])
    for p in sorted(d.glob(name + ".*")) if d.exists() else []:
        if p.suffix.lower() in (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".aiff", ".jpg", ".jpeg", ".png"):
            return p
    return None


def measure_levels(work, cfg, S, E, progress=lambda f: None):
    """Loudness and gate thresholds of each track over the episode [S, E] — from the sources, once.
    Preview and full episode take gain from here, so they sound the same."""
    work = Path(work)
    teq = track_eq(cfg, work, (S, E))
    key = f"{S:.2f}-{E:.2f}|" + "|".join(teq)
    store = jload(work / "levels.json") or {}
    if store.get("key") == key:                 # old format — a single entry
        return store
    if key in store.get("by_key", {}):
        return store["by_key"][key]
    meta, sync = jload(work / "meta.json"), jload(work / "sync.json")
    L, s = sync["offset"], sync["drift"]
    spans = [(S, E), (L + (1 + s) * S, L + (1 + s) * E)]          # the same slice on each source's scale
    out = {"key": key, "tracks": []}
    for i, (path, (a, b)) in enumerate(zip(meta["tracks"], spans)):
        li, _ = loudness(path, f"highpass=f={cfg['highpass_hz']}" + (f",{teq[i]}" if teq[i] else ""), seek=(a, b),
                         on_frac=lambda f, i=i: progress(i * 0.5 + f * 0.4))
        g = max(-20.0, min(30.0, cfg["track_lufs"] - li))
        db = frame_db(load_analysis(path))
        db = db[int(max(0, a) / HOP): int(max(0, b) / HOP)] + g
        nf, sp = (np.percentile(db, 10), np.percentile(db, 97)) if len(db) else (-60.0, -20.0)
        thr = min(max(nf + 0.5 * (sp - nf), nf + 6), sp - 15)
        out["tracks"].append({"lufs": round(li, 2), "gain": round(g, 2), "gate_db": round(float(thr), 2)})
        progress(i * 0.5 + 0.5)
        log(f"levels: track {i + 1}: {li:.1f} LUFS → {g:+.1f} dB, gate {thr:.0f} dB")
    store = {"by_key": {**store.get("by_key", {}), key: out}}
    jsave(work / "levels.json", store)
    return out


def track_eq(cfg, work=None, span=None):
    """EQ of each track (ffmpeg filter strings), honoring the switch; ['', ''] — no per-track EQ.
    Source: mic profiles (track_profile → mic_profiles, "auto" — auto_eq), otherwise the legacy track_eq list."""
    if not cfg["eq"]:
        return ["", ""]
    prof = cfg.get("track_profile") or []
    if prof:
        out = []
        for i in range(2):
            name = prof[i] if i < len(prof) else ""
            if name in AUTO:
                out.append(auto_eq(work, span)[i] if work and span else "")
            else:
                out.append((cfg.get("mic_profiles") or {}).get(name, ""))
        return out
    t = cfg.get("track_eq") or []
    return [t[i] if i < len(t) else "" for i in range(2)]


EQ_BANDS = [250, 500, 1000, 2000, 4000, 8000]    # auto-EQ octaves (below 250 Hz untouched: that is the
                                                 # male/female voice difference, not the mic)


def voice_spectrum(path, a, b, sr=32000):
    """Mean speech spectrum of a track over [a, b] (file seconds): dB per EQ_BANDS octave, relative to 1 kHz.
    Only loud frames (speech) count; silence and noise do not."""
    p = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{max(0.0, a):.3f}", "-to", f"{b:.3f}", "-i", str(path),
                        "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"], capture_output=True, check=True)
    x = np.frombuffer(p.stdout, np.float32)
    n = 2048
    frames = x[: len(x) // n * n].reshape(-1, n)
    if not len(frames):
        return None
    e = (frames ** 2).mean(1)
    speech = frames[e >= np.percentile(e, 60)]               # top 40% by loudness — speech
    spec = (np.abs(np.fft.rfft(speech * np.hanning(n), axis=1)) ** 2).mean(0)
    f = np.fft.rfftfreq(n, 1 / sr)
    band = [spec[(f >= c / 2 ** 0.5) & (f < c * 2 ** 0.5)].mean() for c in EQ_BANDS]
    db = 10 * np.log10(np.array(band) + 1e-20)
    return db - db[EQ_BANDS.index(1000)]


def auto_eq(work, span, limit=4.0):
    """Auto EQ: both voices' spectra are pulled toward their mean (each halfway, at most ±limit dB
    per octave). → [filters A, filters B]; cached in auto_eq.json per slice [S, E]."""
    work = Path(work)
    key = f"{span[0]:.2f}-{span[1]:.2f}"
    cache = jload(work / "auto_eq.json") or {}
    if cache.get("key") == key:
        return cache["filters"]
    meta, sync = jload(work / "meta.json"), jload(work / "sync.json")
    L, s = sync["offset"], sync["drift"]
    S, E = span
    mid = (S + E) / 2                                         # 10 minutes from the middle are enough
    a0, a1 = max(S, mid - 300), min(E, mid + 300)
    spans = [(a0, a1), (L + (1 + s) * a0, L + (1 + s) * a1)]
    curves = [voice_spectrum(p, a, b) for p, (a, b) in zip(meta["tracks"], spans)]
    if any(c is None for c in curves):
        return ["", ""]
    target = (curves[0] + curves[1]) / 2
    filters = []
    for c in curves:
        g = np.clip(target - c, -limit, limit)
        filters.append(",".join(f"equalizer=f={fc}:t=o:w=1:g={x:.1f}" for fc, x in zip(EQ_BANDS, g) if abs(x) >= 0.5))
    jsave(work / "auto_eq.json", {"key": key, "filters": filters,
                                  "curves": [[round(float(v), 1) for v in c] for c in curves]})
    log("auto EQ: " + " | ".join(f or "no change" for f in filters))
    return filters


def preview_segments(work, S, E, length=60):
    """Preview — the first minute after the intro: sync, loudness and effects are checked on it."""
    return [(S, min(E, S + length), _t("start"))]


def quotes(work, S, E, names):
    """First and last turns of the episode — whole turns, clipped exactly to [S, E]."""
    tr = jload(Path(work) / "transcript.json")
    if not tr:
        return "", ""
    clipped = []
    for u in utterances(tr, names):
        ws = [w for w in u[4] if w["start"] / 1000 >= S - 0.05 and w["end"] / 1000 <= E + 0.05]
        if ws:
            clipped.append((ws[0]["start"], u[2], [w["text"] for w in ws]))
    clipped.sort()

    def take(items, limit):
        res, n = [], 0
        for _, who, words in items:
            if n >= limit:
                break
            res.append((who, words[: limit - n] if n + len(words) > limit else words))
            n += len(words)
        return res
    head = take(clipped, 45)
    tail = [(w, ws[::-1]) for w, ws in take([(t, w, ws[::-1]) for t, w, ws in reversed(clipped)], 35)][::-1]
    fmt = lambda xs: "\n".join(f"{w}: {' '.join(ws)}" for w, ws in xs)
    return fmt(head), fmt(tail)


XF = 0.4      # crossfade between pieces (preview, cuts), s


def keep_segments(S, E, cuts=None):
    """[S, E] minus plan["cuts"] ([{from, to}], raw scale) → pieces [(a, b, "")]. No cuts — one piece."""
    segs, a = [], S
    for c in sorted(cuts or [], key=lambda c: c["from"]):
        f, t = max(S, c["from"]), min(E, c["to"])
        if t <= f or t <= a:
            continue
        if f - a > 2 * XF:
            segs.append((a, f, ""))
        a = max(a, t)
    if E - a > 2 * XF or not segs:
        segs.append((a, E, ""))
    return segs


def to_edit(segs, t):
    """Raw time → time from the start of the edited audio (pieces joined with XF crossfade). Inside a cut — start of the next piece."""
    pos = 0.0
    for k, (a, b, _) in enumerate(segs):
        if t < b or k == len(segs) - 1:      # without cuts — exactly t − S, as before
            return pos + (t - a if k == 0 else max(0.0, t - a))
        pos += b - a - XF


def render(work, cfg, progress=lambda *a, **k: None, preview=False):
    """Build: full episode (out/) or preview (preview/, intro + first minute only).

    Chain per track: align B by sync → trim [start, end] from plan → high-pass <70 Hz →
    denoise → loudness (gain from levels.json) → gate → voice EQ (track_eq). Then mix →
    (common EQ if no track_eq) → compressor → loudness to target_lufs → intro/outro → limiter at 4× rate → MP3.
    Full episode also: chapters (ID3 CHAP/CTOC + chapters.txt/json), transcript txt/srt,
    caption.txt (channel post), description.txt. progress(key, fraction, info) — for a progress indicator.
    """
    work = Path(work)
    meta, sync, plan = jload(work / "meta.json"), jload(work / "sync.json"), jload(work / "plan.json")
    S, E = parse_ts(plan["start"]), parse_ts(plan["end"])
    a_path, b_path = meta["tracks"]
    if preview:                               # all preview progress is one stage, "preview"
        base = progress
        span = {"levels": (0.0, 0.35), "clean": (0.35, 0.65), "mix": (0.65, 0.8), "final": (0.8, 1.0)}

        def progress(key, f=None, info=None):
            a, b = span[key]
            base("preview", min(0.99, a + (f or 0) * (b - a)))
    segs = preview_segments(work, S, E) if preview else keep_segments(S, E, plan.get("cuts"))
    D = sum(b - a for a, b, _ in segs) - XF * (len(segs) - 1)
    progress("levels" if preview else "clean", 0.0)
    lv = measure_levels(work, cfg, S, E, (lambda f: progress("levels", f)) if preview else (lambda f: progress("clean", f * 0.3)))
    pre = [f"highpass=f={cfg['highpass_hz']}"]
    if cfg["denoise"] and cfg["denoise_nr"] > 0:
        pre.append(f"afftdn=nr={cfg['denoise_nr']}:tn=1")
    pre = ",".join(pre)
    progress("clean", 0.3 if not preview else 0.0)
    log("render: tracks" + (" (preview)" if preview else ""))
    ext = ".wv" if cfg.get("tmp_codec") == "wavpack" else ".wav"
    tmpc = ["-c:a", cfg.get("tmp_codec") or "pcm_f32le"]
    ta, tb = work / f"A{ext}", work / f"B{ext}"
    heads = [f"aresample={SR},aformat=sample_fmts=fltp:channel_layouts=mono", b_align_filter(sync, SR)]
    for i, (path, head, out_wav) in enumerate(zip(meta["tracks"], heads, (ta, tb))):
        on = (lambda f, i=i: progress("clean", (0.3 if not preview else 0.0) + (i + f) * (0.35 if not preview else 0.5)))
        if len(segs) == 1:
            ff(["-i", path, "-af", f"{head},atrim=start={S:.3f}:end={E:.3f},asetpts=N/SR/TB,"
                                   f"apad=whole_dur={D:.3f},atrim=end={D:.3f},{pre}",
                *tmpc, out_wav], on_frac=on, total=D)
        else:
            # each piece is cleaned with a 5 s margin (the denoiser needs to "listen in"), margin cut off later; pieces crossfaded
            g = [f"[0:a]{head},apad,asplit={len(segs)}" + "".join(f"[x{k}]" for k in range(len(segs)))]
            for k, (a, b, _) in enumerate(segs):
                pad = min(5.0, a)
                g.append(f"[x{k}]atrim=start={a - pad:.3f}:end={b:.3f},asetpts=N/SR/TB,{pre},"
                         f"atrim=start={pad:.3f},asetpts=N/SR/TB[s{k}]")
            cur = "[s0]"
            for k in range(1, len(segs)):
                g.append(f"{cur}[s{k}]acrossfade=d={XF}:c1=tri:c2=tri[c{k}]")
                cur = f"[c{k}]"
            ff(["-i", path, "-filter_complex", ";".join(g), "-map", cur, *tmpc, out_wav],
               on_frac=on, total=D)

    chain = []
    teq = track_eq(cfg, work, (S, E))
    for i, t in enumerate(lv["tracks"]):
        f = f"[{i}:a]volume={t['gain']:.2f}dB"
        if cfg["gate"]:
            f += (f",agate=threshold={db2lin(t['gate_db']):.5f}:range={db2lin(cfg['gate_range_db']):.4f}"
                  f":ratio=3:attack=5:release=300:knee=2")
        if teq[i]:                            # per-voice EQ — before mixing
            f += "," + teq[i]
        if cfg["deesser"]:
            f += ",deesser=i=0.4:m=0.5:f=0.5:s=o"
        if cfg["dynaudnorm"]:                 # slow level riding: 250 ms window × 15 ≈ 4 s, gain at most ×3
            f += ",dynaudnorm=f=250:g=15:p=0.9:m=3"
        chain.append(f + f"[t{i}]")
    post = ["amix=inputs=2:normalize=0:duration=longest"]
    if cfg["eq"] and not any(teq):           # common "studio" EQ when there are no per-track presets
        post += ["equalizer=f=250:t=q:w=1:g=-2.5", "equalizer=f=3500:t=q:w=1:g=2", "highshelf=f=10000:g=1.5"]
    post.append(f"acompressor=threshold={db2lin(cfg['comp_threshold_db']):.5f}:ratio={cfg['comp_ratio']}"
                ":attack=15:release=250:knee=2.8")
    progress("mix", 0.0)
    log("render: mixing")
    main = work / f"main{ext}"
    ff(["-i", ta, "-i", tb, "-filter_complex", ";".join(chain) + ";[t0][t1]" + ",".join(post) + "[m]",
        "-map", "[m]", *tmpc, main], on_frac=lambda f: progress("mix", f * 0.85), total=D)
    li, _ = loudness(main, on_frac=lambda f: progress("mix", 0.85 + 0.15 * f))
    gm = cfg["target_lufs"] - li + 0.5      # the limiter usually eats ~0.5 dB — compensate upfront

    # intro/outro
    intro, outro = find_asset(cfg, "intro"), find_asset(cfg, "outro")
    if preview:
        outro = None                          # preview — start only
    inputs, parts, t_main = ["-i", main], [], 0.0
    if intro:
        Li = duration(intro)
        xo = min(cfg["intro_overlap"], Li / 2)
        t_main = Li - xo
        gi = cfg["music_lufs"] - loudness(intro)[0]
        inputs += ["-i", intro]
        fd = min(cfg["intro_fade"], Li / 2)
        parts.append(f"[1:a]aresample={SR},aformat=sample_fmts=fltp:channel_layouts=stereo,volume={gi:.2f}dB,"
                     f"afade=t=out:st={Li - fd:.3f}:d={fd:.3f}[in]")
    parts.append(f"[0:a]volume={gm:.2f}dB,aformat=channel_layouts=stereo,adelay=delays={t_main * 1000:.0f}:all=1[mn]")
    mix = (["[in]"] if intro else []) + ["[mn]"]
    if outro:
        Lo = duration(outro)
        yo = min(cfg["outro_overlap"], Lo / 2)
        go = cfg["music_lufs"] - loudness(outro)[0]
        k = len(inputs) // 2
        inputs += ["-i", outro]
        parts.append(f"[{k}:a]aresample={SR},aformat=sample_fmts=fltp:channel_layouts=stereo,volume={go:.2f}dB,"
                     f"afade=t=in:d={yo:.3f},adelay=delays={(t_main + D - yo) * 1000:.0f}:all=1[out]")
        mix.append("[out]")
    tp = db2lin(cfg["true_peak_db"] - 0.7)    # limiter at 4× rate catches inter-sample peaks; headroom for MP3
    graph = ";".join(parts) + ";" + "".join(mix) + (
        f"amix=inputs={len(mix)}:normalize=0:duration=longest," if len(mix) > 1 else "anull,") + \
        f"aresample={SR * 4},alimiter=limit={tp:.4f}:level=0:attack=5:release=80,aresample=44100[o]"

    title = plan.get("title") or meta.get("title") or _t("episode")
    n = meta.get("n")
    out = work / ("preview" if preview else "out")
    out.mkdir(exist_ok=True)
    for old in out.glob("*"):
        old.unlink()
    if preview:
        mp3 = out / (_t("preview #{n}.mp3", n=n) if n else _t("preview.mp3"))
    elif n and cfg.get("podcast_name"):
        safe_name = re.sub(r'[\\/:*?"<>|]+', " ", cfg["podcast_name"]).strip(" .") or "podcast"
        mp3 = out / f"{safe_name} №{n}.mp3"
    else:
        mp3 = out / (time.strftime("%y%m%d", time.localtime(meta.get("created", time.time()))) + "_" + slug(title) + ".mp3")
    progress("final", 0.0)
    log("render: final")
    T = t_main + D + 2
    enc = lambda g, a, b: ff(inputs + ["-filter_complex", g, "-map", "[o]", "-ar", "44100", "-c:a", "libmp3lame",
                                       "-b:a", cfg["bitrate"], "-compression_level", "0", mp3], on_frac=lambda f: progress("final", a + f * (b - a)), total=T)
    enc(graph, 0.0, 0.8)
    final_i, final_tp = loudness(mp3, on_frac=lambda f: progress("final", 0.8 + 0.15 * f))
    miss = cfg["target_lufs"] - final_i
    if abs(miss) > 0.6:                        # the limiter "ate" some loudness — second pass with a correction
        log(f"render: loudness correction {miss:+.1f} dB")
        graph = graph.replace(f"[0:a]volume={gm:.2f}dB", f"[0:a]volume={gm + miss:.2f}dB")
        enc(graph, 0.8, 0.95)
        final_i, final_tp = loudness(mp3)
    total = duration(mp3)
    first, last = quotes(work, S, E, meta["names"])

    if preview:
        marks, t = [], t_main
        for a, b, label in segs:
            marks.append({"at": round(t, 1), "src": round(a - S + t_main, 1), "len": round(b - a, 1), "label": label})
            t += b - a - XF
        from mutagen.id3 import ID3, TIT2
        tags = ID3()
        tags.add(TIT2(encoding=3, text=[_t("preview #{n}: {title}", n=n, title=title) if n else _t("preview: {title}", title=title)]))
        tags.save(mp3, v2_version=3)
        rep = {"file": mp3.name, "path": str(mp3), "duration": round(total, 1), "lufs": round(final_i, 1),
               "true_peak": round(final_tp, 1), "segments": marks, "full_duration": round(t_main + (E - S) + (duration(find_asset(cfg, "outro")) if find_asset(cfg, "outro") else 0), 1),
               "first_words": first, "last_words": last, "title": title, "n": n,
               "start_reason": plan.get("start_reason", ""), "end_reason": plan.get("end_reason", ""),
               "sync": {k: sync[k] for k in ("mode", "offset", "drift_ms_per_hour", "confidence")},
               "b_delay": sync.get("b_delay", 0.0), "comp_ratio": cfg["comp_ratio"], "eq": cfg["eq"],
               "profiles": list(cfg.get("track_profile") or []), "deesser": cfg["deesser"],
               "dynaudnorm": cfg["dynaudnorm"], "denoise_nr": cfg["denoise_nr"] if cfg["denoise"] else 0}
        jsave(work / "preview.json", rep)
        for x in (ta, tb, main):
            x.unlink(missing_ok=True)
        log(f"preview ready: {fmt_ts(total)}, {final_i:.1f} LUFS, peak {final_tp:.1f} dBTP")
        return rep

    # chapters and transcript on the episode scale
    shift = lambda t: t_main + to_edit(segs, t)
    chapters = [[max(0.0, shift(parse_ts(c["start"]))), c["title"].strip()] for c in plan.get("chapters") or []]
    chapters = [c for c in chapters if c[0] < total - 5]
    if intro and (not chapters or chapters[0][0] > 15):
        chapters.insert(0, [0.0, _t("start")])
    if chapters:
        chapters[0][0] = 0.0
    # in show notes the first chapter starts at the first word after the intro ("00:09 …")
    shown = [[t, name] for t, name in chapters]
    if shown and intro:
        shown[0][0] = t_main + cfg["pre_roll"]
    chap_txt = "\n".join(f"{fmt_chap(t)} {name}" for t, name in shown)
    (out / "chapters.txt").write_text(chap_txt + "\n")
    jsave(out / "chapters.json", {"version": "1.2.0", "chapters": [{"startTime": round(t, 1), "title": n} for t, n in chapters]})

    tr = jload(work / "transcript.json")
    if tr:
        us = [u for u in utterances(tr, meta["names"]) if any(u[1] > a and u[0] < b for a, b, _ in segs)]
        (out / "transcript.txt").write_text("\n\n".join(f"[{fmt_ts(shift(u[0]), True)}] {u[2]}: {u[3]}" for u in us) + "\n")
        (out / "transcript.srt").write_text(srt(us, shift))

    head = f"#{n}: {title}" if n else title
    caption = head + ("\n\n" + chap_txt if chapters else "")
    (out / "caption.txt").write_text(caption + "\n")                 # channel post format
    desc = [head, ""] + ([plan["description"], ""] if plan.get("description") else []) + ([chap_txt] if chapters else [])
    (out / "description.txt").write_text("\n".join(desc).strip() + "\n")

    write_tags(mp3, head, cfg, meta, chapters, total, plan.get("description", ""))
    rep = {"file": mp3.name, "duration": round(total, 1), "lufs": round(final_i, 1), "true_peak": round(final_tp, 1),
           "trim": [S, E], "cuts": [[a, b] for a, b in zip([x[1] for x in segs], [x[0] for x in segs[1:]])], "intro": bool(intro), "outro": bool(outro), "chapters": len(chapters),
           "sync": {k: sync[k] for k in ("mode", "offset", "drift_ms_per_hour", "confidence")},
           "plan_source": plan.get("source"), "first_words": first, "last_words": last,
           "start_reason": plan.get("start_reason", ""), "end_reason": plan.get("end_reason", ""),
           "n": n, "title": title, "caption": caption, "summary": plan.get("description", "")}
    jsave(work / "report.json", rep)
    for x in (ta, tb, main):
        x.unlink(missing_ok=True)
    log(f"done: {mp3.name}, {fmt_ts(total)}, {final_i:.1f} LUFS, peak {final_tp:.1f} dBTP")
    return rep


def srt(us, shift):
    """Turns → SRT subtitles: cues ≤ 80 chars / ≤ 6 s, ending at a sentence end where possible."""
    cues = []
    for u in us:
        words = [(w["start"] / 1000, w["end"] / 1000, w["text"]) for w in u[4]] or [(u[0], u[1], u[3])]
        cur = []
        for w in words:
            cur.append(w)
            txt = " ".join(x[2] for x in cur)
            if len(txt) > 80 or cur[-1][1] - cur[0][0] > 6 or (len(txt) > 40 and txt[-1] in ".?!…"):
                cues.append((cur[0][0], cur[-1][1], txt))
                cur = []
        if cur:
            cues.append((cur[0][0], cur[-1][1], " ".join(x[2] for x in cur)))

    def t(x):
        x = max(0.0, shift(x))
        return f"{int(x // 3600):02d}:{int(x // 60 % 60):02d}:{int(x % 60):02d},{int(x * 1000 % 1000):03d}"
    return "\n".join(f"{i}\n{t(a)} --> {t(b)}\n{txt}\n" for i, (a, b, txt) in enumerate(cues, 1))


TRANSLIT = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
                    "a b v g d e e zh z i y k l m n o p r s t u f h ts ch sh sch _ y _ e yu ya".split()))


def slug(s):
    """Title → latin snake_case (for file names when the episode number is unknown)."""
    s = "".join(TRANSLIT.get(c, c) for c in s.lower()).replace("_", "")
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")[:60] or "episode"


def write_tags(mp3, title, cfg, meta, chapters, total, description):
    """ID3v2.3: title, artists, album, date, description, cover and chapters (CHAP + CTOC) —
    chapters show up in Apple Podcasts, Overcast, Pocket Casts.
    """
    from mutagen.id3 import APIC, CHAP, COMM, CTOC, ID3, TALB, TDRC, TIT2, TPE1, CTOCFlags
    tags = ID3()
    tags.add(TIT2(encoding=3, text=[title]))
    artist = _t(" and ").join(meta.get("names") or []) or cfg.get("podcast_name")
    if artist:
        tags.add(TPE1(encoding=3, text=[artist]))
        tags.add(TALB(encoding=3, text=[cfg.get("podcast_name") or artist]))
    tags.add(TDRC(encoding=3, text=[time.strftime("%Y-%m-%d", time.localtime(meta.get("created", time.time())))]))
    if description:
        tags.add(COMM(encoding=3, lang="rus", desc="", text=[description]))
    cover = find_asset(cfg, "cover")
    if cover:
        tags.add(APIC(encoding=3, mime="image/png" if cover.suffix.lower() == ".png" else "image/jpeg",
                      type=3, desc="Cover", data=cover.read_bytes()))
    ids = []
    for i, (t, name) in enumerate(chapters):
        end = chapters[i + 1][0] if i + 1 < len(chapters) else total
        ids.append(f"chp{i}")
        tags.add(CHAP(element_id=ids[-1], start_time=int(t * 1000), end_time=int(end * 1000),
                      sub_frames=[TIT2(encoding=3, text=[name])]))
    if ids:
        tags.add(CTOC(element_id="toc", flags=CTOCFlags.TOP_LEVEL | CTOCFlags.ORDERED, child_element_ids=ids,
                      sub_frames=[TIT2(encoding=3, text=[_t("Chapters")])]))
    tags.save(mp3, v2_version=3)


# ───────────────────────── orchestration ─────────────────────────

def run(work, cfg, progress=lambda *a, **k: None, key=None, preview=True):
    """Full run; stages already done (sync/transcript/plan) are not repeated."""
    work = Path(work)
    meta = jload(work / "meta.json")
    key = key if key is not None else os.environ.get("ASSEMBLYAI_API_KEY", "")
    if not (work / "sync.json").exists():
        progress("sync", 0.0)
        jsave(work / "sync.json", analyze(*meta["tracks"], marks=meta.get("sync_marks"),
                                          marks_meta=meta.get("sync_marks_meta")))
    sync = jload(work / "sync.json")
    ce = ((sync.get("signal") or {}).get("clock_err") or [])
    progress("sync", 1.0, info=_t("offset {offset:+.2f} s", offset=sync["offset"]) + (
        _t(" · by sync signal") + (_t(" (clocks ±{errs} ms)", errs="/±".join(str(x) for x in ce if x is not None)) if any(x is not None for x in ce) else "")
        if sync["mode"] == "signal" else ""))
    if key and not (work / "transcript.json").exists():
        progress("transcribe", 0.0)
        prev = work / "stereo.mp3"
        make_stereo_preview(*meta["tracks"], sync, prev)
        try:
            tr = transcribe(prev, key, cfg["language"], cfg.get("aai_extra"))
        finally:
            prev.unlink(missing_ok=True)
        jsave(work / "transcript.json", apply_spelling(tr, cfg.get("spelling") or {}))
    tr = jload(work / "transcript.json")
    if tr:
        progress("transcribe", 1.0, info=_t("{n} utterances", n=len(tr.get("utterances") or [])))
    if not (work / "plan.json").exists():
        progress("plan", 0.0)
        make_plan(work, cfg, key)
    plan = jload(work / "plan.json")
    progress("plan", 1.0, info=_t("{n} chapters", n=len(plan.get("chapters") or [])) if plan.get("source") == "llm" else _t("no LLM"))
    return render(work, cfg, progress, preview=preview)


def new_job(work, tracks, names, title="", n=None):
    """Create the episode folder with meta.json (track paths, host names in A/B channel order, number)."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    jsave(work / "meta.json", {"tracks": [str(Path(t).resolve()) for t in tracks], "names": names,
                               "title": title, "created": time.time(), "n": n})
    return work


def apply_edits(work, start=None, end=None, chapters_text=None):
    """Manual edit: trim and/or chapters ("mm:ss Title" per line, time on the FINISHED episode scale)."""
    work = Path(work)
    plan, sync, rep = jload(work / "plan.json"), jload(work / "sync.json"), jload(work / "report.json") or {}
    cfg = load_cfg(work)
    intro = find_asset(cfg, "intro")
    t_main = (duration(intro) - min(cfg["intro_overlap"], duration(intro) / 2)) if intro else 0.0
    old_S = plan["start"]
    to_raw = lambda t: t - t_main + old_S                       # episode scale → raw
    if chapters_text is not None:
        ch = []
        for line in chapters_text.splitlines():
            m = re.match(r"\s*(\d{1,2}(?::\d{2}){1,2})\s*[-–—]?\s*(.+)", line)
            if m:
                ch.append({"start": round(max(old_S, to_raw(parse_ts(m.group(1)))), 2), "title": m.group(2).strip()})
        plan["chapters"] = ch
    if start is not None or end is not None:
        # the trim is given on the FINISHED episode scale (what you hear in the MP3)
        s = to_raw(parse_ts(start)) if start is not None else plan["start"]
        e = to_raw(parse_ts(end)) if end is not None else plan["end"]
        lim = sync.get("timeline_end") or sync["dur_a"]
        plan["start"], plan["end"] = round(max(0.0, s), 2), round(min(lim, e), 2)
    plan["source"] = (plan.get("source") or "") + "+manual"
    jsave(work / "plan.json", plan)
    return plan


def main():
    """CLI: run (full build without preview), render (rebuild with edits), analyze (sync only)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("a")
    r.add_argument("b")
    r.add_argument("--names", nargs=2, default=[_t("Host 1"), _t("Host 2")])
    r.add_argument("--work")
    r.add_argument("--title", default="")
    e = sub.add_parser("render")
    e.add_argument("work")
    e.add_argument("--start")
    e.add_argument("--end")
    e.add_argument("--chapters", help="file with \"mm:ss Title\" lines (finished episode time)")
    a = sub.add_parser("analyze")
    a.add_argument("a")
    a.add_argument("b")
    args = ap.parse_args()
    if args.cmd == "analyze":
        print(json.dumps({k: v for k, v in analyze(args.a, args.b).items() if k != "segments"}, ensure_ascii=False, indent=1))
    elif args.cmd == "run":
        work = args.work or f"work_{time.strftime('%y%m%d_%H%M%S')}"
        new_job(work, [args.a, args.b], args.names, args.title)
        print(json.dumps(run(work, load_cfg(work), preview=False), ensure_ascii=False, indent=1))
    else:
        ch = Path(args.chapters).read_text() if args.chapters else None
        if args.start or args.end or ch:
            apply_edits(args.work, args.start, args.end, ch)
        print(json.dumps(render(args.work, load_cfg(args.work)), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
