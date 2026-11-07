# Emotion-Aware Call Routing from a Single Caller Utterance

Automated call routing decides where to send a caller from *what* they say — transcribed
keywords — and ignores *how* they say it. A calm enquiry and an angry complaint about the
same billing issue are routed identically and wait identically.

This system listens to the single short utterance a caller speaks to the automated system
at the start of a call, predicts one of six emotions, and decides whether the caller should
be fast-tracked to a human agent. It was built for UCCD3094 Principles of Deep Learning
(Assignment 2) on the CREMA-D corpus, using a WavLM-Base+ speech backbone in PyTorch.

## Results

Best model: `crema_d_partial_top4` (top four transformer layers fine-tuned).

| Split | 6-way UAR | Routing balanced acc | Escalate recall | Routing false alarm |
| --- | --- | --- | --- | --- |
| Validation | 0.7511 | 0.8596 | 0.9346 | 0.2155 |
| Test (speaker-held-out) | 0.6961 | 0.8729 | 0.9107 | 0.1648 |

Routing numbers use the distress grouping (Angry/Fearful/Sad/Disgusted → human agent).
Full comparison, including LoRA and frozen-backbone runs:
[`results/master_table.md`](results/master_table.md). Every number there has a run JSON in
`results/` behind it.

## Quick start (Windows, macOS, Linux)

Needs Python 3.11+ and about 1 GB of disk for the backbone and checkpoint. CPU is enough.

```bash
python -m venv .venv
```

Activate it — Windows: `.venv\Scripts\activate` · macOS/Linux: `source .venv/bin/activate`.

```bash
pip install -r requirements.txt
```

```bash
python scripts/download_checkpoint.py
```

```bash
python app.py --inbrowser
```

The checkpoint (114 MB) lives in the GitHub Release because GitHub refuses files over
100 MB in the repository itself; the download script checks its SHA-256. The first app
start also downloads the `microsoft/wavlm-base-plus` backbone (~360 MB) into the normal
Hugging Face cache. Then open <http://127.0.0.1:7860> if the browser does not open itself.

On Windows, `Launch Speech Emotion Demo.bat` does the last step with a double-click once
the setup above is done.

## The demo

Record from the microphone or upload a file. The app converts the audio to mono, removes DC
offset, trims edge silence, matches the speech level to −24 dBFS, resamples to 16 kHz, and
pads or centre-crops to four seconds. It shows the predicted emotion, the routing decision,
and the full probability breakdown. Behaviour and limits: [`docs/UI_demo_spec.md`](docs/UI_demo_spec.md).

## Tests

```bash
python scripts/smoke_test.py
```

```bash
python -m pytest tests
```

The smoke test builds a tiny synthetic corpus and a dummy backbone, so it checks data
splitting, model wiring, every fine-tuning strategy, metrics and the training loop in
under a minute on CPU, with no dataset or download. `tests/` covers audio preparation,
inference and the Gradio layout.

## Training from scratch

Training needs a GPU and was run on Google Colab via
[`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb). The same scripts run anywhere:

```bash
git clone https://github.com/CheyneyComputerScience/CREMA-D.git data/raw/CREMA-D
```

```bash
python scripts/prepare_crema_d.py --audio-dir data/raw/CREMA-D/AudioWAV
```

```bash
python scripts/train.py --config configs/crema_d_baseline.yaml --run-id my_run --override model.finetune_strategy=partial
```

`prepare_crema_d.py` builds speaker-independent 70/15/15 splits — no actor appears in two
splits. `audit_dataset.py` opens every clip to find corrupt, silent or duplicate audio.
`routing_analysis.py` re-scores a finished run's saved predictions as a binary routing
decision, with no model or GPU needed.

## Layout

| Path | Contents |
| --- | --- |
| `app.py` | Gradio web app: record/upload, analyse, show emotion and route |
| `src/inference.py` | checkpoint loading, audio validation and cleanup, prediction |
| `src/data/` | filename parsing, labels, speaker-independent splits, `Dataset` |
| `src/models/` | WavLM wrapper, weighted layer pooling, heads, four fine-tuning strategies |
| `src/train/` | training loop, loss, evaluation |
| `src/evaluation/` | six-way metrics and the binary routing collapse |
| `scripts/` | prepare, audit, train, routing analysis, smoke test, checkpoint download |
| `configs/` | baseline YAML; `--override key.path=value` changes one setting per run |
| `results/` | comparison table and every run/routing JSON |
| `docs/` | architecture diagrams, demo spec, design notes |

Start with [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) before reading `src/`. The
decisions log — why the project is shaped this way — is in [`PROJECT_PLAN.md`](PROJECT_PLAN.md).

## Metrics

| Metric | Question it answers |
| --- | --- |
| Unweighted average recall (6-way) | how good is the model, giving rare emotions equal weight |
| Routing balanced accuracy (binary) | how good is the escalate / don't-escalate decision |
| Routing false alarm rate | what share of ordinary callers are wrongly fast-tracked |

Plain accuracy is never the headline: CREMA-D is imbalanced, and accuracy flatters a model
that ignores rare emotions.

## Limitations

- CREMA-D is acted speech from 91 actors reading 12 fixed sentences, not real calls.
- Each prediction is one 2–4 second utterance; there is no conversation-level tracking.
- Happy is most often confused with Angry, which would fast-track a delighted caller.
- Intensity labels are unusable (81.7% unspecified), so only the six emotions are trained.

More in [`docs/REPORT_NOTES.md`](docs/REPORT_NOTES.md) §7.

## Data and licence

CREMA-D is free from <https://github.com/CheyneyComputerScience/CREMA-D> under the Open
Database License and is not stored here. AI assistance (Claude Code) was used for parts of
the code; the per-file log is in [`AI_USE.md`](AI_USE.md).
