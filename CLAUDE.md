# UCCD3094 Assignment 2 — Emotion-Aware Call Routing

Application-based project: from the single short utterance a caller speaks to an
automated system at the start of a call, decide whether they need fast-tracking to a
human agent.

The original goal was continuous escalation monitoring — watching a whole call and
flagging rising anger over time. CREMA-D cannot support that: it is 7,442 isolated
2–3 second clips, with no connected conversation and no time-based ground truth.

The purpose is now single-utterance call routing: from one short thing a caller says
to an automated system, decide whether to fast-track them to a human agent. This
matches what CREMA-D actually provides.

**Framing, motivation, references and limitations live in `docs/REPORT_NOTES.md`; do
not restate them elsewhere.** The streaming escalation view survives as a
demonstration only.

## See also

This file holds the rules I follow every session. For everything else:

- Task list, decisions log, results table → `PROJECT_PLAN.md`
- Report framing, motivation, references, limitations → `docs/REPORT_NOTES.md`
- Diagrams of how the pieces connect → `docs/ARCHITECTURE.md`
- How to run the code → `README.md`

## Hard constraints (from the assignment brief)

- **PyTorch only.** HuggingFace / libraries built on PyTorch are allowed. No TensorFlow, no Keras.
- **Submission zip must stay under 5MB** and must NOT contain the dataset or model weights.
  Those go to Google Drive; the link goes in `README.md`.
- Report is max 10 pages, uses `report_template.docx`, IEEE citation style.
- Every AI-assisted edit must be declared per file/function in `AI_USE.md`.
  Do not write report prose — formatting, editing and proofreading only.
- Deadlines: report + code **6 Sept 2026**; slides + video **9 Sept 2026**.

## Commands

```bash
python scripts/smoke_test.py                                             # fast sanity check (<60s)
python scripts/prepare_crema_d.py --audio-dir <path>/AudioWAV            # build manifest
python scripts/audit_dataset.py --audio-dir <path>/AudioWAV              # open every clip, find corrupt/silent/duplicate
python scripts/make_upload_bundle.py                                     # clean copy for Google Drive
python scripts/train.py --config configs/crema_d_baseline.yaml           # train one configuration
```

Never hand-prune the folder before uploading to Drive — use `make_upload_bundle.py`.
`data` must be excluded at the top level but `src/data/` must be kept, and a name-based
rule gets that wrong, producing `No module named 'src.data'` in Colab.

Always run `scripts/smoke_test.py` after touching anything under `src/data/` or `src/models/`.

## Layout

- `src/data/` — dataset parsing, label mapping, speaker-independent splits
- `src/models/` — backbone wrapper, pooling, heads, fine-tuning strategies
- `scripts/` — entry points (prepare, train, smoke test, routing analysis)
- `configs/` — one baseline YAML plus `--override key.path=value` for sweeps (see
  `scripts/train.py`); never hardcode paths or hyperparameters in code
- `results/` — `master_table.md` plus every run JSON. New run JSONs are written to `--results-dir`
  at run time; the final run JSONs are committed in `results/`
- `data/` — **gitignored**, never write anything here from code except under `data/processed/`

## How to explain things to this user

Assume no background in machine learning, deep learning, audio processing or statistics —
the user is meeting these ideas for the first time, so an unfamiliar term or abbreviation
left unexplained is the most common failure here. (Tone itself is set by the active output
style, not this file.)

## Conventions

- Audio is always 16 kHz mono float32 in `[-1, 1]`.
- Splits are **speaker-independent**. An actor ID must never appear in two splits.
  If you touch splitting logic, re-run the leakage assertion in `smoke_test.py`.
- Development and all ablations use the single 70/15/15 split. The final chosen config also
  gets 5-fold speaker-independent cross-validation, reported as mean and standard deviation.
- Primary metric is **unweighted average recall** (average of per-emotion recall). Never
  report plain accuracy as the headline — CREMA-D is imbalanced and accuracy would flatter a
  model that ignores rare emotions.
- The test split is opened once, at the end. Every decision is made on validation.
- Set seeds via `src.utils.seed.set_seed(cfg.seed)` at the top of every entry point.
- Never report a number that has no `results/<run_id>.json` file behind it.
- Prefer `.py` modules; notebooks stay thin (clone repo, install, call script).

## Label policy — read before touching `src/data/labels.py`

**Train on CREMA-D's native labels: 6-way emotion, plus intensity where available.**

The arousal/valence values in `labels.py` are computed *from* (emotion, intensity) by a
fixed table, so they carry no information the native labels don't already have. They are a
**display and scoring convenience**, not training targets:

- Derive arousal/valence and the escalation score from predicted class probabilities at
  inference time. The escalation score drives the live-demo status display; arousal/valence
  are derivable the same way but are not currently shown anywhere (`docs/ARCHITECTURE.md` §6).
- Never describe proxy arousal/valence as ground truth in results or the report.
- The dimensional regression heads are **out of scope**. Licensed corpora with real human
  arousal/valence ratings (MSP-Podcast, IEMOCAP) were considered and dropped: they need a
  data transfer agreement signed by the university's research office, and MSP-Podcast v2.0
  is 409 hours. All corpora used are freely downloadable — CREMA-D, RAVDESS, MELD.

Intensity is confirmed unusable as a supervised target: 81.7% of CREMA-D clips are tagged
`XX` (unspecified), so the intensity head stays built but switched off.

## Fine-tuning

**LoRA is the adopted method** (`peft`), with `full`, `partial` and `frozen` kept behind the
same `finetune_strategy` config switch so the comparison is a one-line change. Report all
four as an ablation; do not present LoRA itself as the project's contribution.

## Review process

**Codex reviews this code after each task is finished.** Write for that reviewer:

- Explain *why* in comments wherever a choice is non-obvious (a proxy label, a magic
  constant, a fallback path). Do not comment what the code already says.
- Keep functions small and independently testable; a reviewer should be able to check
  one function without reading the whole module.
- State assumptions and limitations in docstrings rather than leaving them implicit —
  especially anything that would be wrong to report as a result.
- Every change must leave `scripts/smoke_test.py` passing, and new behaviour needs a
  new check in it. "It ran once in a notebook" is not evidence.

## Gotchas

- No audio backend is installed locally. `src/data/audio.py` falls back through
  `soundfile` → `scipy.io.wavfile` → stdlib `wave`. Don't add a hard import of any one
  backend to shared code.
- Local machine is CPU-only. Training happens on Colab.
  Anything run locally must work on CPU with a tiny subset.
- Colab sessions die. Checkpoint to Drive every epoch (already built).
