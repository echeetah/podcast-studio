"""Track ingest: what the file is (ffprobe) → common working format, 48 kHz mono float32 WAV.

- 44.1 ↔ 48 kHz is resampling only (soxr); clock drift is measured on the converted tracks.
- Codec padding: AAC/M4A (iPhone etc.) — priming via the edit list, MP3 — LAME/Xing header; ffmpeg
  drops it while decoding. The report keeps what the file had (start_time, initial_padding)
  so it is visible when a file lacks these tags and its start may be shifted.
- Stereo: near-identical channels are summed; different ones (two mics in one file) — the louder channel plus a warning.
"""
import json
import subprocess
from pathlib import Path

import numpy as np

from .show import match_hosts

SR = 48000
ASR = 8000                       # sample rate for quick checks
AUDIO_EXT = {".wav", ".aif", ".aiff", ".flac", ".mp3", ".m4a", ".aac", ".mp4", ".mov", ".ogg", ".opus", ".wv", ".caf", ".webm", ".mka"}


def ff(args):
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *map(str, args)], check=True)


_SOXR = None


def resampler():
    """soxr if ffmpeg is built with it (Linux builds usually are, Homebrew is not), otherwise swr with a long filter."""
    global _SOXR
    if _SOXR is None:
        r = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=44100:d=0.1", "-af",
                            f"aresample={SR}:resampler=soxr", "-f", "null", "-"], capture_output=True)
        _SOXR = r.returncode == 0
    return f"aresample={SR}:resampler=soxr:precision=28" if _SOXR else f"aresample={SR}:filter_size=64:phase_shift=12:cutoff=0.97"


def probe(path):
    """Format of the file's first audio stream."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                        "stream=codec_name,sample_rate,channels,channel_layout,bit_rate,start_time,duration,initial_padding"
                        ":format=format_name,duration,bit_rate", "-of", "json", str(path)],
                       capture_output=True, text=True)
    if r.returncode:
        raise ValueError(f"{Path(path).name}: unreadable ({r.stderr.strip()[:200]})")
    j = json.loads(r.stdout)
    if not j.get("streams"):
        raise ValueError(f"{Path(path).name}: no audio stream")
    s, f = j["streams"][0], j.get("format", {})
    num = lambda x, t=float: t(x) if x not in (None, "N/A", "") else None
    return {"codec": s.get("codec_name"), "container": f.get("format_name"),
            "sample_rate": num(s.get("sample_rate"), int), "channels": num(s.get("channels"), int),
            "layout": s.get("channel_layout"), "bit_rate": num(s.get("bit_rate") or f.get("bit_rate"), int),
            "duration": num(s.get("duration")) or num(f.get("duration")),
            "start_time": num(s.get("start_time")) or 0.0, "initial_padding": num(s.get("initial_padding"), int) or 0}


def decode(path, sr=ASR, channels=None):
    """File → float32 [channels, samples] (for analysis)."""
    ch = ["-ac", str(channels)] if channels else []
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0", *ch,
                        "-ar", str(sr), "-f", "f32le", "-"], capture_output=True, check=True)
    n = channels or probe(path)["channels"] or 1
    x = np.frombuffer(r.stdout, np.float32)
    return x[: len(x) // n * n].reshape(-1, n).T


def mono_plan(path, info):
    """How to downmix to mono: (ffmpeg filter, note)."""
    n = info["channels"] or 1
    if n == 1:
        return "anull", ""
    x = decode(path, ASR, n)
    rms = np.sqrt((x.astype(np.float64) ** 2).mean(axis=1)) + 1e-12
    if n == 2:
        c = float(np.corrcoef(x[0], x[1])[0, 1]) if rms.min() > 1e-6 else 0.0
        if c > 0.9 and rms.max() / rms.min() < 2:
            return "pan=mono|c0=0.5*c0+0.5*c1", ""
    k = int(rms.argmax())
    db = 20 * np.log10(rms / rms.max())
    return f"pan=mono|c0=c{k}", (f"{n} channels differ (levels {', '.join(f'{d:.0f}' for d in db)} dB) — took channel {k + 1}; "
                                 "if the file holds two mics, split them into separate tracks")


def stats(path):
    """Peak, clipped share and silence of the working file."""
    x = decode(path, ASR, 1)[0]
    if not len(x):
        return {"peak_db": None, "clip": 0.0, "silent": True}
    peak = float(np.abs(x).max())
    return {"peak_db": round(20 * np.log10(peak + 1e-12), 1), "clip": round(float((np.abs(x) > 0.999).mean()), 5),
            "silent": peak < 1e-4}


def normalize(src, dst):
    """Source → working WAV, 48 kHz mono float32. Returns the track report."""
    info = probe(src)
    pan, note = mono_plan(src, info)
    rs = [] if info["sample_rate"] == SR else [resampler()]
    af = ",".join([pan, *rs, "aformat=sample_fmts=flt:channel_layouts=mono"])
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    ff(["-i", src, "-map", "0:a:0", "-af", af, "-ar", SR, "-c:a", "pcm_f32le", dst])
    warn = [note] if note else []
    st = stats(dst)
    if st["silent"]:
        warn.append("track is nearly silent")
    elif st["peak_db"] > 0:
        warn.append(f"peak {st['peak_db']:+.1f} dB above 0 dBFS (float recording): the track itself is intact, but exporting "
                    "to WAV/MP3 without lowering the level would clip — the build lowers it automatically")
    elif st["clip"] > 1e-4:
        warn.append(f"clipping: {st['clip'] * 100:.2f}% of samples at full scale")
    # MP4/M4A: priming is in the edit list and ffmpeg honours it (start_time 0 there is normal). Untagged: MP3 without
    # a LAME header (start_time 0) and bare AAC (ADTS) — the start may be shifted by one or two codec frames.
    bare = (info["codec"] == "mp3" and not info["start_time"]) or (info["codec"] == "aac" and "aac" in (info["container"] or ""))
    if bare and not info["initial_padding"]:
        warn.append("no codec padding tags — the start may be shifted by ~25–50 ms (sync will correct it)")
    return {"src": str(src), "work": str(dst), "source": info, "duration": round(probe(dst)["duration"] or 0, 3),
            "resampled": info["sample_rate"] != SR, **st, "warnings": warn}


def pick_reference(tracks, first_id, tol=1.0):
    """Reference — the first host's track; if it is shorter than the longest (> tol s) — the longest plus a warning."""
    by = {t["host"]: t for t in tracks if t.get("host")}
    longest = max(tracks, key=lambda t: t["duration"])
    first = by.get(first_id)
    if first and first["duration"] >= longest["duration"] - tol:
        return first["host"], None
    who = first["host"] if first else first_id
    return longest["host"], (f"the first host's track ({who}) is shorter than the longest by "
                             f"{longest['duration'] - (first['duration'] if first else 0):.0f} s — using {longest['host']} as reference")


def ingest(episode, hosts):
    """All audio files in tracks/ → work/<id>.wav, ingest.json. Host — by file name (show.match_hosts)."""
    ep = Path(episode)
    files = sorted(str(p) for p in (ep / "tracks").iterdir() if p.suffix.lower() in AUDIO_EXT and not p.name.startswith("."))
    if not files:
        raise FileNotFoundError(f"no audio files in {ep / 'tracks'}")
    if not 2 <= len(files) <= 5:
        raise ValueError(f"{len(files)} tracks, need 2 to 5")
    match = match_hosts(files, hosts)
    tracks = []
    for f in files:
        hid, guessed = match[f]
        t = normalize(f, ep / "work" / f"{hid or Path(f).stem}.wav")
        tracks.append({"host": hid, "guessed": guessed, **t})
    names = {h["id"]: h["name"] for h in hosts}
    ref, why = pick_reference(tracks, hosts[0]["id"])
    res = {"sample_rate": SR, "reference": ref, "tracks": tracks,
           "warnings": ([why] if why else []) + [f"{Path(t['src']).name}: host guessed as {names.get(t['host'], '?')} — please check"
                                                  for t in tracks if t["guessed"]]}
    (ep / "ingest.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return res
