#!/usr/bin/env python3
"""studio.py — podcast editing CLI for AI agents (Claude Code, Codex, Cursor…). A thin layer over podcast_core.

Every stage is a separate command; the agent decides what to do next and makes targeted edits. All output is
JSON on stdout (logs go to stderr) so the agent can parse it. State is files in the episode folder
(meta/sync/transcript/plan/levels) plus edits.json — the agent's edit log.

Show and episode:
  studio.py credentials                        create/check ~/.config/podcast-studio/credentials (the key is never printed)
  studio.py show-init DIR --name … --host "Name:f:mic:substring" ×2–5 [--recording remote|room|mixed]
                      [--assets DIR] [--engine assemblyai|whisper] [--language en]
  studio.py episode-new N [--show DIR] [--title …]   folder <show>/episodes/N/tracks/
  studio.py ingest EPISODE_DIR                 tracks → 48 kHz mono float, hosts, reference track → ingest.json

Two-track build. Times in arguments are "raw" (track A timeline): seconds or mm:ss / hh:mm:ss.
  studio.py init DIR A.m4a B.m4a [--names Ann Bob] [--title …] [--n N]
  studio.py status DIR                         what is done, what is not, next step
  studio.py sync DIR [--redo] [--no-signal]    sync (finds a sync chirp in the audio by itself) → sync.json
  studio.py transcribe DIR [--redo]            AssemblyAI → transcript.json (ASSEMBLYAI_API_KEY)
  studio.py plan DIR [--redo]                  LLM: start/end/chapters/description → plan.json
  studio.py find DIR "phrase"                  every hit: time, ¶N.M, speaker, context (last word matches by prefix)
  studio.py text DIR [--from T] [--to T]       paragraphs ¶N of sentences ¶N.M with times — read in chunks
  studio.py set DIR [--start T] [--end T] [--b-delay MS] [--eq on|off] [--comp R] [--denoise N]
                    [--profile-a NAME] [--profile-b NAME]   mic presets (init applies them from show.json)
  studio.py cut DIR FROM TO [--note …]         cut a passage (plan.json → cuts; joins crossfade 0.4 s)
  studio.py uncut DIR N                        restore cut #N (index from status/plan)
  studio.py preview DIR [--at T] [--len S]     a piece of finished audio → preview/
  studio.py render DIR                         full episode → out/
"""
import argparse
import json
import os
import sys
from pathlib import Path

try:
    import podcast_core  # noqa: F401  (installed: pip install -e core)
except ImportError:      # next to it in the same repository
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core" / "src"))
from podcast_core import media, show  # noqa: E402
from podcast_core import pipeline as P  # noqa: E402

STAGES = [("sync", "sync.json"), ("transcribe", "transcript.json"), ("plan", "plan.json"),
          ("preview", "preview"), ("render", "out")]


def out(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=1))


def T(s):
    return None if s is None else P.parse_ts(str(s))


def need(work, name):
    if not (work / name).exists():
        sys.exit(json.dumps({"error": f"no {name} — run the previous stage first", "hint": "studio.py status"}, ensure_ascii=False))


def log_edit(work, **e):
    ed = P.jload(work / "edits.json", {"log": []})
    ed["log"].append(e)
    P.jsave(work / "edits.json", ed)
    return ed


def fail(e):
    sys.exit(json.dumps({"error": str(e)}, ensure_ascii=False))


def cmd_credentials(a):
    r = show.cred_init()
    r["hint"] = ("open the file and paste your AssemblyAI key after \"=\"; no need to show it to the agent"
                 if not all(r["keys"].values()) else "keys are in place")
    out(r)


def cmd_show_init(a):
    try:
        out(show.show_init(a.dir, a.name, a.host, a.recording, a.assets or "", a.engine, a.language))
    except (FileExistsError, ValueError) as e:
        fail(e)


def cmd_episode_new(a):
    try:
        out(show.episode_new(a.show, a.n, a.title))
    except FileNotFoundError as e:
        fail(e)


def cmd_ingest(a):
    try:
        root = show.find_show(a.episode)
        r = media.ingest(a.episode, show.load_show(root)["hosts"])
    except (FileNotFoundError, ValueError) as e:
        fail(e)
    out({"reference": r["reference"], "warnings": r["warnings"],
         "tracks": [{"file": Path(t["src"]).name, "host": t["host"], "guessed": t["guessed"],
                     "format": f"{t['source']['codec']} {t['source']['sample_rate']} Hz × {t['source']['channels']}",
                     "duration": t["duration"], "peak_db": t["peak_db"], "warnings": t["warnings"]} for t in r["tracks"]]})


def episode_info(w):
    """Show and episode if the folder is inside <show>/episodes/N: (show, ingest, episode) or (None, None, None)."""
    try:
        root = show.find_show(w)
    except FileNotFoundError:
        return None, None, None
    for d in [Path(w).resolve(), *Path(w).resolve().parents]:
        if (d / "episode.json").exists() or (d / "ingest.json").exists():
            return show.load_show(root), P.jload(d / "ingest.json"), P.jload(d / "episode.json")
        if d == root:
            break
    return show.load_show(root), None, None


def apply_presets(w):
    """Mic presets from show.json → <work>/settings.json (mic_profiles + track_profile for tracks A and B, language)."""
    sh, ing, _ = episode_info(w)
    if not sh:
        return None
    meta = P.jload(w / "meta.json")
    by_path = {}
    for t in (ing or {}).get("tracks") or []:
        for k in ("src", "work"):
            by_path[str(Path(t[k]).resolve())] = t
    names = []
    for i, path in enumerate(meta["tracks"]):
        t = by_path.get(str(Path(path).resolve()))
        hid = t["host"] if t else (sh["hosts"][i]["id"] if i < len(sh["hosts"]) else None)
        names.append(show.preset_for(sh, hid, (t or {}).get("source", {}).get("codec")))
    js = P.jload(w / "settings.json", {})
    js["mic_profiles"] = {**(js.get("mic_profiles") or {}), **{n: sh["presets"][n].get("eq", "") for n in names}}
    js["track_profile"] = names
    js["language"] = sh.get("language") or js.get("language") or "en"
    P.jsave(w / "settings.json", js)
    return names


def cmd_init(a):
    sh, _, ep = episode_info(a.work)
    n = a.n if a.n is not None else (ep or {}).get("n")
    names = a.names or ([h["name"] for h in sh["hosts"][:2]] if sh else ["Host 1", "Host 2"])
    w = P.new_job(a.work, [a.a, a.b], names, a.title or (ep or {}).get("title", ""), n)
    out({"work": str(w), "n": n, "names": names, "presets": apply_presets(w), "next": "sync"})


def cmd_status(a):
    w = Path(a.work)
    done = {k: (w / f).exists() for k, f in STAGES}
    nxt = next((k for k, _ in STAGES if not done[k]), None)
    info = {"work": str(w), "done": done, "next": nxt}
    if done["sync"]:
        s = P.jload(w / "sync.json")
        info["sync"] = {k: s.get(k) for k in ("mode", "offset", "drift", "b_delay", "confidence")}
    if done["plan"]:
        p = P.jload(w / "plan.json")
        info["plan"] = {k: p.get(k) for k in ("start", "end", "title")} | {"chapters": len(p.get("chapters") or [])}
        info["cuts"] = p.get("cuts") or []
    out(info)


def cmd_stage(a):
    w = Path(a.work)
    meta = P.jload(w / "meta.json")
    cfg = P.load_cfg(w)
    files = dict(STAGES)
    if a.redo and a.cmd in ("sync", "transcribe", "plan"):
        (w / files[a.cmd]).unlink(missing_ok=True)
    if a.cmd == "sync":
        if not (w / "sync.json").exists():
            P.jsave(w / "sync.json", P.analyze(*meta["tracks"], marks=meta.get("sync_marks"),
                                               marks_meta=meta.get("sync_marks_meta"), blind=not a.no_signal, trust_signal=True))
        s = P.jload(w / "sync.json")
        out({k: v for k, v in s.items() if k != "segments"})
    elif a.cmd == "transcribe":
        need(w, "sync.json")
        key = os.environ.get("ASSEMBLYAI_API_KEY") or show.cred_load().get("ASSEMBLYAI_API_KEY", "")
        if not key:
            sys.exit(json.dumps({"error": "no ASSEMBLYAI_API_KEY", "hint": "studio.py credentials"}, ensure_ascii=False))
        if not (w / "transcript.json").exists():
            st = w / "stereo.mp3"
            P.make_stereo_preview(*meta["tracks"], P.jload(w / "sync.json"), st)
            try:
                tr = P.transcribe(st, key, cfg["language"], cfg.get("aai_extra"))
            finally:
                st.unlink(missing_ok=True)
            P.jsave(w / "transcript.json", P.apply_spelling(tr, cfg.get("spelling") or {}))
        out({"utterances": len(P.jload(w / "transcript.json").get("utterances") or [])})
    elif a.cmd == "plan":
        need(w, "sync.json")
        if not (w / "plan.json").exists():
            P.make_plan(w, cfg, os.environ.get("ASSEMBLYAI_API_KEY") or show.cred_load().get("ASSEMBLYAI_API_KEY", ""))
        out(P.jload(w / "plan.json"))


def words(w):
    tr = P.jload(w / "transcript.json") or {}
    return [(x["start"] / 1000, x["end"] / 1000, x["text"]) for u in tr.get("utterances") or [] for x in u.get("words") or []]


def paragraphs(us):
    """Turns → paragraphs ¶N (N — turn number across the whole episode, from 1) of sentences ¶N.M with start and end."""
    out = []
    for n, u in enumerate(us, 1):
        ws = [(w["start"] / 1000, w["end"] / 1000, w["text"]) for w in u[4]] or [(u[0], u[1], u[3])]
        sents, cur = [], []
        for w in ws:
            cur.append(w)
            if w[2].rstrip().endswith((".", "?", "!", "…")):
                sents.append(cur)
                cur = []
        if cur:
            sents.append(cur)
        out.append({"p": n, "who": u[2], "t": round(u[0], 2), "end": round(u[1], 2),
                    "sentences": [{"id": f"{n}.{m}", "t": round(c[0][0], 2), "end": round(c[-1][1], 2),
                                   "text": " ".join(x[2] for x in c)} for m, c in enumerate(sents, 1)]})
    return out


def find_all(us, phrase, ctx=8):
    """Every hit of a phrase (ignoring case, ё/е and punctuation) → [{t, end, ¶, who, context}] by time.
    The last word matches by prefix ("edit" finds "editing", "edited"), the others as whole words."""
    toks = [P._norm(x) for x in phrase.split() if P._norm(x)]
    hits = []
    if not toks:
        return hits
    for para in paragraphs(us):
        ws = [(s["id"], x) for s in para["sentences"] for x in s["text"].split()]
        times = [(w["start"] / 1000, w["end"] / 1000) for w in us[para["p"] - 1][4]]
        norm = [P._norm(x) for _, x in ws]
        for i in range(len(ws) - len(toks) + 1):
            if norm[i:i + len(toks) - 1] == toks[:-1] and norm[i + len(toks) - 1].startswith(toks[-1]):
                j = i + len(toks) - 1
                t0, t1 = (times[i][0], times[j][1]) if len(times) == len(ws) else (para["t"], para["end"])
                hits.append({"t": round(t0, 2), "end": round(t1, 2), "ts": P.fmt_ts(t0), "para": ws[i][0],
                             "who": para["who"], "context": " ".join(x for _, x in ws[max(0, i - ctx): j + 1 + ctx])})
    return sorted(hits, key=lambda h: h["t"])


def transcript_us(w):
    need(w, "transcript.json")
    return P.utterances(P.jload(w / "transcript.json"), P.jload(w / "meta.json")["names"])


def cmd_find(a):
    hits = find_all(transcript_us(Path(a.work)), a.phrase)
    out({"phrase": a.phrase, "count": len(hits), "hits": hits})


def cmd_text(a):
    lo, hi = T(a.from_) or 0, T(a.to) or 1e9
    out([p for p in paragraphs(transcript_us(Path(a.work))) if p["end"] >= lo and p["t"] <= hi])


def cmd_set(a):
    w = Path(a.work)
    changed = {}
    if a.start is not None or a.end is not None:
        need(w, "plan.json")
        plan = P.jload(w / "plan.json")
        if a.start is not None:
            plan["start"] = changed["start"] = round(T(a.start), 2)
        if a.end is not None:
            plan["end"] = changed["end"] = round(T(a.end), 2)
        plan["source"] = (plan.get("source") or "") + "+agent"
        P.jsave(w / "plan.json", plan)
    if a.b_delay is not None:
        need(w, "sync.json")
        s = P.jload(w / "sync.json")
        s["b_delay"] = changed["b_delay"] = round(a.b_delay / 1000, 4)
        P.jsave(w / "sync.json", s)
    js = P.jload(w / "settings.json", {})
    for i, prof in enumerate((a.profile_a, a.profile_b)):
        if prof is not None:
            sh = episode_info(w)[0]
            known = {**(js.get("mic_profiles") or {}), **({k: v.get("eq", "") for k, v in sh["presets"].items()} if sh else {})}
            if prof not in known and prof not in P.AUTO:
                fail(f"no preset \"{prof}\"; available: {', '.join(sorted(known))}, auto")
            tp = list(js.get("track_profile") or ["", ""])
            tp[i] = prof
            js["track_profile"] = tp
            js["mic_profiles"] = {**(js.get("mic_profiles") or {}), **({prof: known[prof]} if prof in known else {})}
            changed[f"profile_{'ab'[i]}"] = prof
    if a.eq is not None:
        js["eq"] = changed["eq"] = a.eq == "on"
    if a.comp is not None:
        js["comp_ratio"] = changed["comp_ratio"] = a.comp
    if a.denoise is not None:
        js["denoise"], js["denoise_nr"] = a.denoise > 0, a.denoise
        changed["denoise_nr"] = a.denoise
    if js:
        P.jsave(w / "settings.json", js)
    if not changed:
        sys.exit(json.dumps({"error": "nothing to change"}, ensure_ascii=False))
    log_edit(w, op="set", **changed)
    out({"changed": changed, "hint": "listen again: studio.py preview"})


def cmd_cut(a):
    w = Path(a.work)
    f, t = T(a.from_), T(a.to)
    if t <= f:
        sys.exit(json.dumps({"error": "end is before start"}, ensure_ascii=False))
    need(w, "plan.json")
    log_edit(w, op="cut", **{"from": round(f, 2), "to": round(t, 2), "note": a.note or ""})
    plan = P.jload(w / "plan.json")
    plan["cuts"] = sorted((plan.get("cuts") or []) + [{"from": round(f, 2), "to": round(t, 2), "note": a.note or ""}],
                          key=lambda c: c["from"])
    P.jsave(w / "plan.json", plan)
    out({"cuts": plan["cuts"], "hint": f"listen to the join: studio.py preview DIR --at {max(0, f - 10):.0f} --len 20"})


def cmd_uncut(a):
    w = Path(a.work)
    need(w, "plan.json")
    plan = P.jload(w / "plan.json")
    cuts = plan.get("cuts") or []
    if not 0 <= a.index < len(cuts):
        sys.exit(json.dumps({"error": f"no cut #{a.index}", "cuts": cuts}, ensure_ascii=False))
    gone = cuts.pop(a.index)
    P.jsave(w / "plan.json", plan)
    log_edit(w, op="uncut", **{"from": gone["from"], "to": gone["to"]})
    out({"removed": gone, "cuts": cuts})


def preview_window(at, ln, S, E, cuts):
    """Preview pieces from at, ln seconds of AUDIO: cut material does not count, the window extends past it."""
    segs, left = [], ln
    for x, y, _ in P.keep_segments(max(S, at), E, cuts):
        if left <= 0:
            break
        y = min(y, x + left + (P.XF if segs else 0))        # a join eats XF — make up for it
        segs.append((x, y, "excerpt"))
        left -= (y - x) - (P.XF if len(segs) > 1 else 0)
    return segs or [(S, min(E, S + ln), "excerpt")]


def cmd_render(a):
    w = Path(a.work)
    for f in ("sync.json", "plan.json"):
        need(w, f)
    cfg = P.load_cfg(w)
    prog = lambda k, f=None, info=None: print(f"[{k}] {f or 0:.0%} {info or ''}", file=sys.stderr)
    if a.cmd == "preview" and a.at is not None:
        at, ln = T(a.at), a.len
        cuts = P.jload(w / "plan.json").get("cuts")
        P.preview_segments = lambda work, S, E, length=60: preview_window(at, ln, S, E, cuts)
    out(P.render(w, cfg, prog, preview=a.cmd == "preview"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("credentials"); p.set_defaults(fn=cmd_credentials)
    p = sub.add_parser("show-init"); p.add_argument("dir"); p.add_argument("--name", required=True)
    p.add_argument("--host", action="append", required=True, help="Name[:m|f[:mic[:substring,…]]]")
    p.add_argument("--recording", default="remote", choices=show.RECORDING); p.add_argument("--assets")
    p.add_argument("--engine", default="assemblyai", choices=show.ENGINES); p.add_argument("--language", default="en")
    p.set_defaults(fn=cmd_show_init)
    p = sub.add_parser("episode-new"); p.add_argument("n", type=int); p.add_argument("--show", default=".")
    p.add_argument("--title", default=""); p.set_defaults(fn=cmd_episode_new)
    p = sub.add_parser("ingest"); p.add_argument("episode"); p.set_defaults(fn=cmd_ingest)
    p = sub.add_parser("init"); p.add_argument("work"); p.add_argument("a"); p.add_argument("b")
    p.add_argument("--names", nargs=2, help="default: the first two hosts from show.json")
    p.add_argument("--title", default=""); p.add_argument("--n", type=int, help="episode number (otherwise from episode.json)")
    p.set_defaults(fn=cmd_init)
    p = sub.add_parser("status"); p.add_argument("work"); p.set_defaults(fn=cmd_status)
    for name in ("sync", "transcribe", "plan"):
        p = sub.add_parser(name); p.add_argument("work"); p.add_argument("--redo", action="store_true")
        if name == "sync":
            p.add_argument("--no-signal", action="store_true", help="do not search for a sync chirp")
        p.set_defaults(fn=cmd_stage)
    p = sub.add_parser("find"); p.add_argument("work"); p.add_argument("phrase"); p.set_defaults(fn=cmd_find)
    p = sub.add_parser("text"); p.add_argument("work"); p.add_argument("--from", dest="from_"); p.add_argument("--to")
    p.set_defaults(fn=cmd_text)
    p = sub.add_parser("set"); p.add_argument("work"); p.add_argument("--start"); p.add_argument("--end")
    p.add_argument("--b-delay", type=float, help="second track shift, ms (> 0 — B later)")
    p.add_argument("--eq", choices=["on", "off"]); p.add_argument("--comp", type=float)
    p.add_argument("--denoise", type=int, help="denoise strength, 0 — off")
    p.add_argument("--profile-a", help="mic preset for track A (from show.json, or auto)")
    p.add_argument("--profile-b", help="mic preset for track B")
    p.set_defaults(fn=cmd_set)
    p = sub.add_parser("cut"); p.add_argument("work"); p.add_argument("from_", metavar="FROM"); p.add_argument("to")
    p.add_argument("--note"); p.set_defaults(fn=cmd_cut)
    p = sub.add_parser("uncut"); p.add_argument("work"); p.add_argument("index", type=int); p.set_defaults(fn=cmd_uncut)
    p = sub.add_parser("preview"); p.add_argument("work"); p.add_argument("--at")
    p.add_argument("--len", type=float, default=60); p.set_defaults(fn=cmd_render)
    p = sub.add_parser("render"); p.add_argument("work"); p.set_defaults(fn=cmd_render)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
