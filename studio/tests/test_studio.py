"""Studio: preview window with cuts, find-all, ¶N.M paragraphs, presets from show.json. Russian text tests ё/е."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("studio", Path(__file__).resolve().parent.parent / "studio.py")
st = importlib.util.module_from_spec(spec)
spec.loader.exec_module(st)
P = st.P


def total(segs):
    return sum(b - a for a, b, _ in segs) - P.XF * (len(segs) - 1)


def test_preview_window_counts_only_kept_audio():
    cuts = [{"from": 1230.22, "to": 1247.68}]
    segs = st.preview_window(1230, 30, 0, 4000, cuts)          # real-recording case: used to give 12.3 s instead of 30
    assert total(segs) == pytest.approx(30)
    assert all(not (b > 1230.22 and a < 1247.68) for a, b, _ in segs)
    segs = st.preview_window(1215, 30, 0, 4000, cuts)          # window across a join
    assert total(segs) == pytest.approx(30) and len(segs) == 2
    assert total(st.preview_window(100, 30, 0, 4000, None)) == pytest.approx(30)
    assert total(st.preview_window(3990, 30, 0, 4000, None)) == pytest.approx(10)   # hits the end


def u(start, who, text):
    ws, t = [], start
    for x in text.split():
        ws.append({"start": int(t * 1000), "end": int((t + 0.3) * 1000), "text": x})
        t += 0.4
    return [start, t, who, text, ws]


US = [u(10, "Ann", "Привет. Это наш подкаст! Давай ещё раз."),
      u(20, "Bob", "Давай ещё раз, я сбился. Ну ладно"),
      u(30, "Ann", "Давай, ещё   раз! Последний.")]


def test_paragraphs_numbered_by_sentence():
    ps = st.paragraphs(US)
    assert [p["p"] for p in ps] == [1, 2, 3]
    assert [s["id"] for s in ps[0]["sentences"]] == ["1.1", "1.2", "1.3"]
    assert ps[0]["sentences"][1]["text"] == "Это наш подкаст!"
    assert ps[0]["sentences"][1]["t"] == pytest.approx(10.4)
    assert ps[1]["sentences"][-1]["text"] == "Ну ладно"                       # a tail without a period is a sentence too


def test_find_all_returns_every_hit_with_context():
    hits = st.find_all(US, "давай ещё раз")
    assert [h["para"] for h in hits] == ["1.3", "2.1", "3.1"]
    assert [h["who"] for h in hits] == ["Ann", "Bob", "Ann"]
    assert hits[1]["t"] == pytest.approx(20) and "сбился" in hits[1]["context"]
    assert st.find_all(US, "нет такого") == []
    assert [h["para"] for h in st.find_all(US, "ПОДКА")] == ["1.2"]          # by prefix, case-insensitive
    assert [h["para"] for h in st.find_all(US, "давай ещё ра")] == ["1.3", "2.1", "3.1"]
    assert len(st.find_all(US, "давай еще")) == 3                            # ё = е
    assert st.find_all(US, "дав ещё раз") == []                              # not the last word — whole word only


def test_presets_from_show(tmp_path):
    from podcast_core import show
    show.show_init(tmp_path, "T", ["Ann:f:H4essential", "Bob:m:iPhone"])
    sh = show.load_show(tmp_path)
    assert show.preset_for(sh, "h1") == "H4essential"
    assert show.preset_for(sh, "h2", "aac") == "iPhone compressed"
    assert show.preset_for(sh, "h2", "alac") == "iPhone"
    assert show.preset_for(sh, "h9") == "default"


def run_doctor(capsys):
    st.cmd_doctor(None)
    import json
    r = json.loads(capsys.readouterr().out)
    return r, {c["name"]: c for c in r["checks"]}


def test_doctor_reports_missing_ffmpeg_with_fix(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("PODCAST_STUDIO_CREDENTIALS", str(tmp_path / "credentials"))
    real_which = st.shutil.which
    monkeypatch.setattr(st.shutil, "which", lambda x: None if x in ("ffmpeg", "ffprobe") else real_which(x))
    r, c = run_doctor(capsys)
    assert r["ok"] is False
    assert c["ffmpeg"]["ok"] is False and "ffmpeg" in c["ffmpeg"]["fix"]
    assert c["AssemblyAI key"]["required"] is False and c["AssemblyAI key"]["ok"] is False
    assert "fix" not in c["python"]


def test_doctor_key_is_optional(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("PODCAST_STUDIO_CREDENTIALS", str(tmp_path / "credentials"))
    monkeypatch.setattr(st.shutil, "which", lambda x: "/bin/" + x)
    monkeypatch.setattr(st.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": "ffmpeg version 8\n"})())
    r, c = run_doctor(capsys)
    assert r["ok"] is True and c["ffmpeg"]["detail"] == "ffmpeg version 8"


def test_missing_ffmpeg_error_points_to_doctor():
    with pytest.raises(SystemExit) as e:
        st.fail(FileNotFoundError(2, "No such file or directory", "ffprobe"))
    assert "studio.py doctor" in str(e.value)


def test_bad_time_is_a_clear_error():
    with pytest.raises(ValueError, match="bad time"):
        P.parse_ts("abc")
    assert P.parse_ts("1:30") == 90


def test_bad_arguments_are_json(capsys, monkeypatch):
    monkeypatch.setattr(st.sys, "argv", ["studio.py", "set", "x", "--comp", "abc"])
    with pytest.raises(SystemExit) as e:
        st.main()
    import json
    assert "error" in json.loads(str(e.value))


def test_init_tracks_from_ingest_and_relative(tmp_path):
    ep = tmp_path / "ep"
    (ep / "work").mkdir(parents=True)
    for n in ("h1", "h2", "h3"):
        (ep / "work" / f"{n}.wav").write_bytes(b"")
    ing = {"reference": "h2", "tracks": [{"host": h, "work": str(ep / "work" / f"{h}.wav")} for h in ("h1", "h2", "h3")]}
    sh = {"hosts": [{"id": "h1"}, {"id": "h2"}, {"id": "h3"}]}
    assert [Path(p).stem for p in st.init_tracks(ep, None, None, sh, ing)] == ["h2", "h1"]
    assert Path(st.init_tracks(ep, "work/h3.wav", "work/h1.wav", sh, ing)[0]).name == "h3.wav"
    with pytest.raises(SystemExit):
        st.init_tracks(ep, "nope.wav", "work/h1.wav", sh, ing)
    with pytest.raises(SystemExit):
        st.init_tracks(ep, None, None, sh, None)
