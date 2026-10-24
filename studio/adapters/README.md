# Agent adapters

`studio/SKILL.md` is the single source of truth. Each adapter only points an agent at it.

| Agent | How to connect |
|---|---|
| Claude Code | `mkdir -p .claude/skills && ln -s ../../studio .claude/skills/podcast-studio` (project) or `ln -s "$PWD/studio" ~/.claude/skills/podcast-studio` (all projects). Claude Code reads `SKILL.md` as is. Windows (no `ln -s`): `xcopy /E /I studio .claude\skills\podcast-studio`. |
| Codex CLI | Nothing to do: `AGENTS.md` is already in the repository root. |
| Cursor | Copy `adapters/cursor/podcast-studio.mdc` to `.cursor/rules/`. |
| Gemini CLI | Copy `adapters/AGENTS.md` to the root as `GEMINI.md`. |
| Any other agent with a shell | Tell it: "Follow studio/SKILL.md to edit the podcast." |
