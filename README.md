# podcast-studio

AI-agent podcast editor: sync 2–5 tracks, transcription, text-based editing, jingles, mastering.
Works with Claude Code, Codex, Cursor and other coding agents.

Each host records themselves (a recorder, a phone, a call app's local track). You drop the files into a folder
and talk to your agent: *"start episode 12"*, *"cut the part where we argue about coffee"*,
*"the second voice is a bit late"*. The agent runs the `studio.py` commands, shows you previews of every
change and builds the final MP3 with chapters.

> **Status: early.** Working today: show setup, ingest of 2–5 tracks in any format, two-track sync
> (by turn-taking, shared sound or a sync chirp found in the audio), AssemblyAI transcription, start/end/chapters,
> cuts by transcript, mic EQ presets, previews, mastering to −16 LUFS, MP3 with ID3 chapters.
> Planned: 3–5-track build, edit list with jingles and music beds, local Whisper.

## Requirements

- Python 3.10+
- ffmpeg and ffprobe in `PATH` (`brew install ffmpeg`, `apt install ffmpeg`)
- An AI coding agent with a shell (Claude Code, Codex CLI, Cursor, Gemini CLI…)
- For transcription: an [AssemblyAI](https://www.assemblyai.com/) API key (local Whisper is planned)

## Quick start (5 minutes)

```bash
git clone https://github.com/echeetah/podcast-studio && cd podcast-studio
python3 -m venv .venv && . .venv/bin/activate
pip install -e core
python studio/studio.py credentials     # creates ~/.config/podcast-studio/credentials (chmod 600)
```

1. Open `~/.config/podcast-studio/credentials` and paste your AssemblyAI key after `ASSEMBLYAI_API_KEY=`.
   Do not paste the key into the agent chat.
2. Connect the skill to your agent — see [studio/adapters](studio/adapters/README.md).
   For Claude Code: `mkdir -p .claude/skills && ln -s ../../studio .claude/skills/podcast-studio`.
3. Start the agent in this folder and say: *"Set up my podcast show"*. It asks about hosts, mics and jingles
   and writes `show.json`.
4. Say *"Start episode 1"*, put the hosts' files into `episodes/1/tracks/` and let the agent take it from there.

## The flow

```
show.json (once per show)
 └─ episodes/N/tracks/        ← your original files, never modified
     ingest     → work/*.wav (48 kHz mono float), hosts matched by file name, reference track
     sync       → sync.json  (offset + clock drift; a sync chirp in the audio is found by itself)
     transcribe → transcript.json (word timings)
     plan       → plan.json  (start, end, chapters)
     edit       → cuts, sync nudges, EQ/compression — by transcript, in plain words
     preview    → preview/   (only the changed spots)
     render     → out/       MP3 with ID3 chapters, chapters.txt, transcript.txt/.srt
```

Every command prints JSON, so the agent always knows the state. `python studio/studio.py --help` lists them all.

## AssemblyAI or Whisper?

| | AssemblyAI (default) | Whisper, local (planned) |
|---|---|---|
| Accuracy of words and timings | best | good, timings a bit coarser |
| Privacy | audio is uploaded to the cloud | stays on your computer |
| Cost | paid per audio minute, per track | free |
| Speed | minutes per hour of audio | about real time on an Apple Silicon Mac (`large-v3-turbo`) |

## Credentials

The AssemblyAI key lives in `~/.config/podcast-studio/credentials` (permissions 600), or in the
`ASSEMBLYAI_API_KEY` environment variable. The studio never prints it; `studio.py credentials` only says
whether it is set.

## Layout

- `core/` — the `podcast-core` Python package: ingest, sync (incl. sync-chirp search), audio processing, rendering.
- `studio/` — the agent CLI (`studio.py`), the skill (`SKILL.md`) and agent adapters.

Need more accurate sync? — see [podcast-studio-bot](https://github.com/echeetah/podcast-studio-bot): a Telegram mini app that plays a sync chirp on every host's device at the same moment.

## License

MIT
