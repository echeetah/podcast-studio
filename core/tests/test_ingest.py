"""Track ingest on synthetic files: a click at a known moment must stay in place through any codec."""
import os
import stat
import subprocess

import numpy as np
import pytest

from podcast_core import media, show

CLICK = 1.5


def make(path, sr=48000, ch=1, codec=None, extra=(), dur=4.0, click=CLICK, swap=False):
    """Silence with −60 dB noise and a click at `click` (swap — the click is in the right channel only)."""
    n = int(sr * dur)
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1e-3, (n, ch)).astype(np.float32)
    k = int(sr * click)
    x[k:k + 48, 1 if swap else 0] = 0.8
    if ch == 2 and not swap:
        x[k:k + 48, 1] = 0.8
    raw = path.with_suffix(".f32")
    raw.write_bytes(x.tobytes())
    args = ["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(sr), "-ac", str(ch), "-i", raw]
    args += (["-c:a", codec] if codec else []) + list(extra) + [path]
    subprocess.run(list(map(str, args)), check=True)
    raw.unlink()
    return path


def click_at(wav):
    x = media.decode(wav, 48000, 1)[0]
    return np.argmax(np.abs(x) > 0.3) / 48000


@pytest.mark.parametrize("name,sr,codec,extra,tol", [
    ("a.wav", 48000, None, (), 0.0005),
    ("b.wav", 44100, None, (), 0.0005),
    ("c.flac", 44100, "flac", (), 0.0005),
    ("d.m4a", 44100, "aac", ("-b:a", "128k"), 0.003),
    ("e.mp3", 44100, "libmp3lame", ("-q:a", "4"), 0.003),      # VBR
])
def test_click_stays_in_place(tmp_path, name, sr, codec, extra, tol):
    src = make(tmp_path / name, sr=sr, codec=codec, extra=extra)
    r = media.normalize(src, tmp_path / "w" / (name + ".wav"))
    info = media.probe(r["work"])
    assert info["sample_rate"] == 48000 and info["channels"] == 1 and info["codec"] == "pcm_f32le"
    assert abs(click_at(r["work"]) - CLICK) < tol, (name, click_at(r["work"]))
    assert r["resampled"] == (sr != 48000)


def test_stereo_same_is_summed_different_takes_louder(tmp_path):
    same = media.normalize(make(tmp_path / "s.wav", ch=2), tmp_path / "s48.wav")
    assert not same["warnings"]
    diff = media.normalize(make(tmp_path / "d.wav", ch=2, swap=True), tmp_path / "d48.wav")
    assert any("channel 2" in w for w in diff["warnings"])
    assert abs(click_at(diff["work"]) - CLICK) < 0.0005


def test_pick_reference():
    tr = [{"host": "h1", "duration": 3600.0}, {"host": "h2", "duration": 3600.5}]
    assert media.pick_reference(tr, "h1") == ("h1", None)
    ref, why = media.pick_reference([{"host": "h1", "duration": 3000.0}, {"host": "h2", "duration": 3600.0}], "h1")
    assert ref == "h2" and "shorter" in why


def test_show_episode_ingest(tmp_path):
    r = show.show_init(tmp_path / "show", "Test", ["Ann:f:sm7b", "Боря:m"], recording="room")
    assert r["default_preset_for"] == ["Ann"]      # no mic given for Боря — default right away
    with pytest.raises(FileExistsError):
        show.show_init(tmp_path / "show", "Test", ["A", "B"])
    ep = show.episode_new(tmp_path / "show", 7)
    tracks = tmp_path / "show" / "episodes" / "7" / "tracks"
    make(tracks / "zoom_borya.m4a", sr=44100, codec="aac", dur=5.0)
    make(tracks / "ann_iphone.wav", sr=48000, dur=3.0)
    res = media.ingest(ep["episode"], show.load_show(tmp_path / "show")["hosts"])
    by = {t["host"]: t for t in res["tracks"]}
    assert set(by) == {"h1", "h2"} and not any(t["guessed"] for t in res["tracks"])
    assert res["reference"] == "h2" and any("shorter" in w for w in res["warnings"])
    assert (tracks.parent / "work" / "h1.wav").exists()


def test_match_hosts_guess():
    hosts = [show.parse_host("Ann", 0), show.parse_host("Боря", 1)]    # Cyrillic name vs latin file name
    m = show.match_hosts(["x/Track 1.wav", "x/боря.m4a"], hosts)
    assert m["x/боря.m4a"] == ("h2", False) and m["x/Track 1.wav"] == ("h1", True)


def test_credentials(tmp_path, monkeypatch):
    p = tmp_path / "cfg" / "credentials"
    monkeypatch.setenv("PODCAST_STUDIO_CREDENTIALS", str(p))
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    r = show.cred_init()
    assert r["created"] and r["private"] and r["keys"] == {"ASSEMBLYAI_API_KEY": False}
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o600
    p.write_text(p.read_text().replace("ASSEMBLYAI_API_KEY=", "ASSEMBLYAI_API_KEY=abc"))
    assert show.cred_init() == {**show.cred_init(), "created": False}
    assert show.cred_status()["keys"]["ASSEMBLYAI_API_KEY"] and "abc" not in str(show.cred_status())
