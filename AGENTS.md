# AGENTS.md

This repository is a podcast editor for AI agents.

When the user asks to set up a show, start an episode, sync, transcribe, edit, preview or render a podcast,
read and follow `studio/SKILL.md`. Always talk to the user in their language; files and commands stay in English.

- CLI: `python studio/studio.py --help` (JSON output on stdout, logs on stderr).
- Core library: `core/` (package `podcast_core`), tests: `python -m pytest -q core studio/tests`.
