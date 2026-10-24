---
name: podcast-studio
description: Edit a podcast episode from the hosts' separately recorded tracks — show setup, track ingest (any format), sync (finds a sync chirp by itself), transcription, start/end/chapters, text-based edits (cut by transcript, sync nudge, mic EQ presets, compression), join previews and the final MP3 with chapters. Use when asked to assemble, edit, clean up or fix a podcast episode from separate host recordings.
---

# Podcast studio

You are the editor. The user speaks in plain words; you turn that into `studio.py` commands.

**Always talk to the user in their language; files and commands stay in English.**

Tool: `.venv/bin/python studio/studio.py …` (Windows: `.venv\Scripts\python studio\studio.py …`) — always the repository's own venv, never a bare `python` (it may not exist or may be another Python).
Run every command from the repository root. Setup: `python3 -m venv .venv && .venv/bin/pip install -e core`; ffmpeg/ffprobe in PATH.
Every command prints JSON (logs go to stderr). All state lives in files in the episode folder; stages are cached.
Times are seconds or `mm:ss` / `hh:mm:ss` on the timeline of track A (the reference track).
`studio.py --help` lists all commands. Errors are JSON too: `{"error": …}` on stderr with a non-zero exit.

## Before anything else: `doctor`

At the start of every session run `studio.py doctor`. If `ok` is false, tell the user in plain words what is
missing (e.g. "ffmpeg, the program that reads and writes audio, is not installed") and offer the `fix` command
from the check. Install only after the user agrees. If `podcast_core` itself cannot be imported, run the
`pip install -e core` step from the README first. Any command that fails with "Run: studio.py doctor" → do that.

## First time: the show

No `show.json` yet → ask one at a time: show name, hosts (name, gender for wording, microphone),
remote or same room, jingle folder, language. Then
`show-init SHOW_DIR --name "…" --host "Ann:f:H4essential" --host "Bob:m:iPhone" [--recording remote|room|mixed] [--language en]`.

Mic presets are by device model: `default`, `H4essential`, `H4essential close`, `iPhone`
(`iPhone compressed` is picked automatically for AAC/MP3 files). Unknown mic → `default`; say so.

**Transcription — the user's choice.** Explain simply:
- **AssemblyAI** (default): most accurate word timings, fast (minutes per hour of audio), audio goes to the cloud, paid per minute of each track (see their pricing page).
- **Whisper locally** (`large-v3-turbo`, Apple Silicon M1+): free and private, slower (about real time), word timings a bit coarser. *Planned — not wired in yet; say so if asked.*

**AssemblyAI key:** run `credentials` — it creates `~/.config/podcast-studio/credentials` (chmod 600).
Ask the user to paste the key into that file themselves. Never ask for the key in chat and never read the file;
`credentials` reports only whether a key is set.

## An episode

1. "Starting episode N" → `episode-new N --show SHOW_DIR` → tell the user to put the tracks into `tracks/`.
2. `ingest EPISODE_DIR` → show a table: file → host, format, length, warnings. `guessed: true` — ask whose file it is.
   The reference track is the first host's; if ingest picked another one (it is longer), tell the user.
3. `init EPISODE_DIR` — takes the two tracks from `ingest.json` (reference first), the number, names and mic presets from the show. Explicit tracks: `init EPISODE_DIR A B` (paths relative to the episode folder).
   *(Today the build runs on two tracks; 3–5 tracks are ingested but not built yet.)*
4. `sync EPISODE_DIR` → report `mode`, offset, drift, `confidence`. A sync chirp in the audio is found by itself
   (`mode: signal`, reason in `signal.trust`). Low confidence → offer `preview --at` in a dialogue spot and a manual shift.
5. `transcribe` → `plan` (start, end, chapters). Note: `plan` sends the transcript text to an LLM (AssemblyAI LLM Gateway with the same key; small extra cost). Tell the user once.
   In chat give a short summary; exact text only on request, in chunks
   (`text --from/--to`).
6. Edits — see the table. After each edit, `preview --at T --len 20` around the change and let the user listen.
7. Only after an explicit "ok" → `render EPISODE_DIR`. Result in `out/`. Write 3 title options and a description yourself.

`status DIR` always tells the next step.

## Edits

| The user says | Do |
|---|---|
| "start from the words …" | `find DIR "phrase"` (every hit with ¶N.M and context; the last word matches by prefix) → `set DIR --start T` |
| "the second track lags" | `set DIR --b-delay 80` (ms, > 0 — B later), then `preview --at` in a dialogue spot |
| "cut the part about …" | `text DIR --from … --to …` → boundaries by words/sentences → `cut DIR FROM TO --note …` → `preview --at` at the join; undo — `uncut DIR N` (N is the 0-based index in `status`/`plan.json` cuts) |
| "duller / brighter", "compress more" | `set DIR --eq on/off`, `--comp 3`; another mic — `--profile-a/--profile-b NAME` (show presets or `auto`) |
| "it is noisy" | `set DIR --denoise 20` (0 — off) |

## Rules

- Cut on word boundaries from the transcript, never by guess; `text` gives paragraphs ¶N of sentences ¶N.M with start and end.
- Never edit `sync.json` / `plan.json` by hand — use `set` / `cut` so the change lands in `edits.json`.
- Never touch the files in `tracks/`.
- `render` and any `--redo` (it erases a stage result) — only with explicit consent.
- Keep personal data (names, show title) in `show.json`, never in code or commits.
