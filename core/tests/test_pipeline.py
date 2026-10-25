"""Processing tests on synthetic data (need numpy and ffmpeg). Russian words test Cyrillic handling."""
import shutil
import wave

import numpy as np
import pytest

from podcast_core import pipeline as P

SR = 16000
needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


def write_wav(path, x, sr=SR):
    y = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(y.tobytes())


@pytest.mark.parametrize("s,v", [("105", 105), ("01:23", 83), ("1:45.0", 105), ("105s", 105), ("1:02:13", 3733),
                                 ("-0:30", -30), ("95,5 с", 95.5), (" 62:13 ", 3733)])
def test_parse_ts(s, v):
    assert P.parse_ts(s) == pytest.approx(v)


def test_find_phrase_picks_nearest_and_handles_split_words():
    words = [(10.0, 10.3, "Поехали."), (11.0, 11.4, "Всем"), (11.4, 11.8, "привет,"),
             (95.0, 95.3, "Всем"), (95.3, 95.7, "привет,"), (95.8, 96.2, "мой подкаст")]
    assert P.find_phrase(words, "Всем привет", 90) == pytest.approx(95.0)
    assert P.find_phrase(words, "Всем привет", 12) == pytest.approx(11.0)
    assert P.find_phrase(words, "привет, мой подкаст", 96, last=True) == pytest.approx(96.2)
    assert P.find_phrase(words, "нет такой фразы", 50) is None


def test_apply_spelling():
    tr = {"words": [{"text": "Мойподкаст,"}], "utterances": [{"text": "с вами Мойподкаст", "words": []}]}
    P.apply_spelling(tr, {"Мойподкаст": "мой подкаст"})
    assert tr["words"][0]["text"] == "мой подкаст,"
    assert tr["utterances"][0]["text"] == "с вами мой подкаст"


def test_match_marks_uses_spacing():
    hits = [(100.0, 0.9), (40.0, 0.95), (3640.0, 0.92)]            # a stray chirp at 100 s
    pairs = P.match_marks(hits, [1000.0, 4600.0])                    # presses 3600 s apart
    assert [p for _, p in pairs] == [40.0, 3640.0]


@needs_ffmpeg
def test_find_chirps_in_noise(tmp_path):
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.05, SR * 150).astype(np.float32)             # 2.5 min of noise, longer than one search chunk
    ch = P.chirp_wave(SR) * 0.2
    for t in (12.5, 130.0):
        i = int(t * SR)
        x[i:i + len(ch)] += ch
    write_wav(tmp_path / "a.wav", x)
    hits = sorted(h[0] for h in P.find_chirps(tmp_path / "a.wav"))
    assert hits == pytest.approx([12.5, 130.0], abs=0.01)


def dialog(seconds, seed):
    """Two "tracks" with alternating turns (noise bursts), like a remote recording on headphones."""
    rng = np.random.default_rng(seed)
    a, b = np.zeros(SR * seconds, np.float32), np.zeros(SR * seconds, np.float32)
    t, who = 2.0, 0
    while t < seconds - 8:
        d = rng.uniform(1.0, 5.0)
        i, j = int(t * SR), int((t + d) * SR)
        (a if who == 0 else b)[i:j] += rng.normal(0, 0.2, j - i)
        t += d + rng.uniform(0.3, 0.9)
        who ^= 1
    a += rng.normal(0, 0.003, len(a))
    b += rng.normal(0, 0.003, len(b))
    return a, b


@needs_ffmpeg
def test_analyze_finds_offset_by_turns(tmp_path):
    a, b = dialog(600, seed=7)
    lag = 3.3                                                         # the second track started 3.3 s earlier
    b = np.concatenate([np.random.default_rng(3).normal(0, 0.003, int(lag * SR)).astype(np.float32), b])
    write_wav(tmp_path / "a.wav", a)
    write_wav(tmp_path / "b.wav", b)
    r = P.analyze(tmp_path / "a.wav", tmp_path / "b.wav", max_offset=60)
    assert r["mode"] == "turns"
    assert r["offset"] == pytest.approx(lag, abs=0.15)


def test_keep_segments_and_to_edit():
    from podcast_core import pipeline as P
    assert P.keep_segments(10, 100) == [(10, 100, "")]
    assert P.to_edit([(10, 100, "")], 5) == -5 and P.to_edit([(10, 100, "")], 120) == 110   # as before: t − S
    segs = P.keep_segments(10, 100, [{"from": 30, "to": 40}, {"from": 0, "to": 12}, {"from": 95, "to": 200}])
    assert segs == [(12, 30, ""), (40, 95, "")]
    assert P.to_edit(segs, 20) == 8
    assert P.to_edit(segs, 35) == P.to_edit(segs, 40) == 18 - P.XF   # inside a cut — start of the next piece
    assert abs(P.to_edit(segs, 50) - (18 - P.XF + 10)) < 1e-9


def test_blind_signal_sync(tmp_path):
    """No timestamps: two chirps in each track → offset and drift; a single stray chirp does no harm."""
    rng = np.random.default_rng(3)
    off, drift = 3.2, 1e-4
    a = rng.normal(0, 0.05, SR * 150).astype(np.float32)
    b = rng.normal(0, 0.05, SR * 160).astype(np.float32)
    ch = P.chirp_wave(SR) * 0.2
    weak = P.chirp_wave(SR) * 0.05      # NCC ≈ 0.4, like a neighbour phone's speaker caught by a recorder (seen in a real recording)
    for x, ts in ((a, (12.5, 130.0)), (b, [off + (1 + drift) * t for t in (12.5, 130.0)] + [70.0])):
        for k, t in enumerate(ts):
            i = int(t * SR)
            x[i:i + len(ch)] += weak if x is a and k == 1 else ch
    write_wav(tmp_path / "a.wav", a)
    write_wav(tmp_path / "b.wav", b)
    sig = P.blind_signal_sync(tmp_path / "a.wav", tmp_path / "b.wav")
    assert sig and sig["blind"] and sig["has_drift"]
    assert sig["offset"] == pytest.approx(off, abs=0.005)
    assert sig["drift"] == pytest.approx(drift, abs=3e-5)
    write_wav(tmp_path / "c.wav", rng.normal(0, 0.05, SR * 150).astype(np.float32))
    assert P.blind_signal_sync(tmp_path / "a.wav", tmp_path / "c.wav") is None


@needs_ffmpeg
def test_preview_name_without_episode_number(tmp_path, monkeypatch):
    a, b = dialog(40, 5)
    write_wav(tmp_path / "a.wav", a)
    write_wav(tmp_path / "b.wav", b)
    w = P.new_job(tmp_path / "ep", [tmp_path / "a.wav", tmp_path / "b.wav"], ["Ann", "Bob"])
    P.jsave(w / "sync.json", P.analyze(tmp_path / "a.wav", tmp_path / "b.wav"))
    P.jsave(w / "plan.json", {"start": 1, "end": 30, "title": "t", "chapters": []})
    monkeypatch.setattr(P, "LANG", "en")
    assert P.render(w, P.load_cfg(w), preview=True)["file"] == "preview.mp3"
    monkeypatch.setattr(P, "LANG", "ru")                 # Russian locale
    assert P.render(w, P.load_cfg(w), preview=True)["file"] == "превью.mp3"


def test_signal_gaps_agree():
    sig = lambda a, b: {"pairs": [[(0, x) for x in a], [(0, x) for x in b]]}
    ok, why = P.signal_gaps_agree(sig([26.712, 3921.692], [26.989, 3922.017]))   # real recording: +48 ms over 3895 s
    assert ok and "agree" in why
    assert not P.signal_gaps_agree(sig([10.0, 1000.0], [10.0, 1000.3]))[0]        # 300 ms is not drift
    assert P.signal_gaps_agree(sig([10.0, 3610.0], [10.0, 3610.4]))[0]             # 100 ppm·3600 s + 40 ms
    assert not P.signal_gaps_agree(sig([10.0], [10.0]))[0]


def test_analyze_trust_signal_only_when_asked(monkeypatch, tmp_path):
    """The signal is 1 s off the talk: by default talk wins, with trust_signal the signal wins."""
    a, b = dialog(200, seed=11)
    write_wav(tmp_path / "a.wav", a)
    write_wav(tmp_path / "b.wav", b)
    talk = P.analyze(tmp_path / "a.wav", tmp_path / "b.wav", max_offset=30)
    fake = {"offset": talk["offset"] + 1.0, "drift": 0.0, "has_drift": True, "blind": True, "ncc": [0.9, 0.9],
            "pairs": [[(20, 20), (180, 180)], [(20, 21 + talk["offset"]), (180, 181 + talk["offset"])]]}
    monkeypatch.setattr(P, "blind_signal_sync", lambda *x, **k: dict(fake))
    default = P.analyze(tmp_path / "a.wav", tmp_path / "b.wav", max_offset=30, blind=True)
    assert default["mode"] != "signal" and "rejected" in default["signal"]
    st = P.analyze(tmp_path / "a.wav", tmp_path / "b.wav", max_offset=30, blind=True, trust_signal=True)
    assert st["mode"] == "signal" and st["offset"] == pytest.approx(talk["offset"] + 1.0, abs=1e-3)
    assert "agree" in st["signal"]["trust"]


def test_denoise_filter(tmp_path):
    base = {"denoise": True, "denoise_nr": 12, "rnnoise_model": ""}
    assert P.denoise_filter(base) == "afftdn=nr=12:tn=1"
    assert P.denoise_filter({**base, "denoise": False}) == ""
    assert P.denoise_filter({**base, "denoise_nr": 0}) == ""
    model = tmp_path / "bd.rnnn"
    model.write_text("rnnoise-nu model file version 1\n")
    assert P.denoise_filter({**base, "rnnoise_model": str(model)}) == f"arnndn=m={model}"
    assert P.denoise_filter({**base, "rnnoise_model": str(model), "denoise": False}) == ""
    weird = tmp_path / "a:b,c'd.rnnn"                       # символы, ломающие цепочку фильтров, экранируются
    weird.write_text("x")
    f = P.denoise_filter({**base, "rnnoise_model": str(weird)})
    assert f.startswith("arnndn=m=") and "\\:" in f and "\\'" in f
    assert P.denoise_filter({**base, "rnnoise_model": str(tmp_path / "missing.rnnn")}) == "afftdn=nr=12:tn=1"   # нет файла — запасной
