# podcast-core

Building a podcast from the hosts' tracks, without a UI: shows and episodes, ingest of any format
(→ 48 kHz mono float32), sync (by turn-taking, by shared sound, by a sync chirp if one is in the recording),
audio processing, transcription, MP3 rendering with chapters.

```bash
pip install -e core[test]     # needs ffmpeg and ffprobe in PATH
python -m pytest -q core
```

- `podcast_core.show` — `show.json`, `episodes/N/`, credentials file `~/.config/podcast-studio/credentials` (600), files → hosts.
- `podcast_core.media` — `probe`, `normalize` (resampling, stereo → mono, codec padding), `ingest`, reference track.
- `podcast_core.pipeline` — two-track sync (`analyze`, `find_chirps`, `blind_signal_sync`), transcription, loudness, `render`.
  `configure(app_dir, title, lang)` — where shared `settings.json` and `assets/` live, and the UI language (`en`, `ru`).

Neural denoiser (optional): put the path of an [RNNoise model](https://github.com/GregorR/rnnoise-models) (e.g. `bd.rnnn`) into the
`rnnoise_model` setting — each track then goes through ffmpeg `arnndn` instead of the FFT denoiser `afftdn`
(`denoise: false` still turns denoising off). Models are not bundled; try them on a minute of your own voice first —
a neural denoiser removes hiss between words much better, but can colour the voice.

Use from another project: `podcast-core @ git+https://github.com/echeetah/<repo>@<tag>#subdirectory=core`.
