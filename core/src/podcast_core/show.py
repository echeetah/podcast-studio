"""Show (show.json), episodes (<show>/episodes/N/) and the credentials file.

show.json — once per show: hosts and their mics, how they record (remote/room/mixed), jingle folder,
episode structure, title template, transcription engine, presets. An episode is a folder with tracks/
(the user's originals, never touched) and the stage files next to it.
"""
import json
import os
import re
import stat
import time
from pathlib import Path

SHOW = "show.json"
RECORDING = ("remote", "room", "mixed")
ENGINES = ("assemblyai", "whisper")
CRED_KEYS = ("ASSEMBLYAI_API_KEY",)
CRED_TEMPLATE = """# podcast-studio: API keys. Only your user can read this file (chmod 600).
# Paste the token after "=" without quotes. There is no need to show this file to the agent.
# AssemblyAI: https://www.assemblyai.com/app/api-keys
ASSEMBLYAI_API_KEY=
"""

DEFAULT_SHOW = {
    "version": 1,
    "name": "",
    "language": "en",
    "hosts": [],                 # [{id, name, gender: m|f, mic, match: [file name substrings]}]
    "recording": "remote",
    "assets": {"dir": "", "intro": "", "outro": "", "stings": [], "beds": []},
    "structure": ["intro", "main", "outro"],
    "naming": {"file": "{show} #{n}", "title": "{show} #{n}: {title}"},
    "transcribe": {"engine": "assemblyai", "whisper_model": "large-v3-turbo"},
    "spelling": {},
    "presets": {},               # own mic presets; extend and override PRESETS
}

# Mic presets by device model: eq — ffmpeg filter chain per track (after denoise and loudness).
# "iPhone" on a lossy recording (AAC/MP3/Opus) automatically becomes "iPhone compressed".
PRESETS = {
    "default": {"eq": ""},
    "H4essential": {"eq": "highpass=f=85:p=2,equalizer=f=250:t=q:w=1:g=-2,equalizer=f=3500:t=q:w=1.2:g=2,highshelf=f=10000:g=1.5",
                    "note": "Zoom H4essential recorder, ~20 cm from the mouth"},
    "H4essential close": {"eq": "highpass=f=120:p=2,lowshelf=f=200:g=-5,equalizer=f=250:t=q:w=1:g=-2,"
                                 "equalizer=f=3500:t=q:w=1.2:g=4,highshelf=f=8000:g=3",
                           "note": "recorder up close: tame the boom (proximity effect)"},
    "iPhone": {"eq": "highpass=f=80:p=2,equalizer=f=300:t=q:w=1:g=-2,highshelf=f=10000:g=1",
               "note": "built-in mic, lossless recording"},
    "iPhone compressed": {"eq": "equalizer=f=180:t=q:w=1:g=2,equalizer=f=3500:t=q:w=1:g=1",
                      "note": "built-in mic, AAC: bring back lows and clarity"},
}
LOSSY = {"aac", "mp3", "opus", "vorbis", "amr_nb", "amr_wb"}


def jload(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def jsave(p, obj):
    p = Path(p)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    tmp.replace(p)


# ───────────────────────── credentials ─────────────────────────

def cred_path():
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return Path(os.environ.get("PODCAST_STUDIO_CREDENTIALS") or base / "podcast-studio" / "credentials")


def cred_init():
    """Create the credentials template (600) if missing. An existing file is left alone."""
    p = cred_path()
    created = not p.exists()
    if created:
        p.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(CRED_TEMPLATE)
    return {"path": str(p), "created": created, **cred_status()}


def cred_load():
    """Keys: environment variables override the file."""
    vals = {}
    try:
        for line in cred_path().read_text().splitlines():
            m = re.match(r"\s*([A-Z0-9_]+)\s*=\s*(.*?)\s*$", line)
            if m and not line.lstrip().startswith("#") and m.group(2):
                vals[m.group(1)] = m.group(2)
    except FileNotFoundError:
        pass
    for k in CRED_KEYS:
        if os.environ.get(k):
            vals[k] = os.environ[k]
    return vals


def cred_status():
    """What is set — without the values (they are never printed)."""
    p = cred_path()
    mode = stat.S_IMODE(p.stat().st_mode) if p.exists() else None
    vals = cred_load()
    return {"exists": p.exists(), "private": mode is not None and not mode & 0o077,
            "keys": {k: bool(vals.get(k)) for k in CRED_KEYS}}


# ───────────────────────── show ─────────────────────────

def parse_host(spec, i):
    """\"Name[:m|f[:mic[:substring,substring]]]\" → host record."""
    name, gender, mic, match = (spec.split(":") + ["", "", ""])[:4]
    gender = gender if gender in ("m", "f") else ""
    return {"id": f"h{i + 1}", "name": name.strip(), "gender": gender, "mic": mic.strip() or "default",
            "match": [x.strip() for x in match.split(",") if x.strip()] or [name.strip().lower()]}


def show_init(root, name, hosts, recording="remote", assets="", engine="assemblyai", language="en"):
    """Create show.json. hosts — parse_host strings. Never overwrites."""
    root = Path(root)
    if (root / SHOW).exists():
        raise FileExistsError(f"{root / SHOW} already exists")
    if not 2 <= len(hosts) <= 5:
        raise ValueError("a show needs 2 to 5 hosts")
    if recording not in RECORDING:
        raise ValueError(f"recording: {'/'.join(RECORDING)}")
    if engine not in ENGINES:
        raise ValueError(f"engine: {'/'.join(ENGINES)}")
    show = json.loads(json.dumps(DEFAULT_SHOW))
    show.update(name=name, language=language, recording=recording,
                hosts=[parse_host(h, i) for i, h in enumerate(hosts)])
    show["assets"]["dir"] = assets
    show["transcribe"]["engine"] = engine
    root.mkdir(parents=True, exist_ok=True)
    (root / "episodes").mkdir(exist_ok=True)
    jsave(root / SHOW, show)
    unknown = [h["name"] for h in show["hosts"] if h["mic"] not in PRESETS]
    return {"show": str(root / SHOW), "hosts": show["hosts"], "default_preset_for": unknown}


def find_show(start):
    """Look for show.json in start and above (episode commands can run from its folder)."""
    p = Path(start).resolve()
    for d in [p, *p.parents]:
        if (d / SHOW).exists():
            return d
    raise FileNotFoundError("no show.json — run `show-init` first")


def load_show(root):
    s = json.loads(json.dumps(DEFAULT_SHOW))
    s.update(jload(Path(root) / SHOW, {}))
    s["presets"] = {**PRESETS, **(s.get("presets") or {})}
    return s


def preset_for(show, host_id, codec=None):
    """Host's preset name given the source codec; unknown mic → default."""
    h = next((h for h in show["hosts"] if h["id"] == host_id), None)
    name = (h or {}).get("mic") or "default"
    if name == "iPhone" and codec in LOSSY and "iPhone compressed" in show["presets"]:
        name = "iPhone compressed"
    return name if name in show["presets"] else "default"


def episode_new(root, n, title=""):
    root = find_show(root)
    ep = root / "episodes" / str(n)
    (ep / "tracks").mkdir(parents=True, exist_ok=True)
    if not (ep / "episode.json").exists():
        jsave(ep / "episode.json", {"n": n, "title": title, "created": time.time()})
    return {"episode": str(ep), "tracks_dir": str(ep / "tracks"), "show": str(root)}


TRANSLIT = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
                    "a b v g d e e zh z i y k l m n o p r s t u f h ts ch sh sch _ y _ e yu ya".split()))


def latin(s):
    return "".join(TRANSLIT.get(c, c) for c in s.lower()).replace("_", "")


def match_hosts(files, hosts):
    """File → host by name substrings (also transliterated: \"боря\" ~ \"borya\"); no match — next free host in order. guessed=True — ask the user."""
    out, free = {}, [h["id"] for h in hosts]
    for f in files:
        low = Path(f).stem.lower()
        hit = next((h["id"] for h in hosts if h["id"] in free and
                    any(m and (m.lower() in low or latin(m) in latin(low)) for m in h["match"])), None)
        if hit:
            out[f] = (hit, False)
            free.remove(hit)
    for f in files:
        if f not in out:
            out[f] = (free.pop(0), True) if free else (None, True)
    return out
