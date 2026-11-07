# Results — model comparison

**Canonical comparison table for the project.** Add new models here rather than starting a
new table elsewhere. Per-run history and the decisions behind each run:
[`PROJECT_PLAN.md`](../PROJECT_PLAN.md).

All rows use the same manifest and validation split (1,142 clips, 14 held-out actors,
speaker-independent). 6-way metric is unweighted average recall (UAR) — never plain accuracy,
because CREMA-D is imbalanced. Routing metrics come from `scripts/routing_analysis.py`; two
numbers per cell = the two escalate groupings (**distress**: ANG/FEA/SAD/DIS · **narrow**:
ANG/DIS).

> The **false alarm** column below is the *routing* definition: anyone outside the escalate
> group who got escalated. A run's own `metrics` block reports a different, narrower figure
> (calm `NEU` predicted as `ANG`/`DIS`) which is always lower. Do not mix the two.

| Model | 6-way UAR | Routing balanced acc | Escalate recall | False alarm (routing) | Notes |
| --- | --- | --- | --- | --- | --- |
| `crema_d_partial_top4` — top 4 layers unfrozen | **0.7511** | 0.8596 / 0.8799 | 0.9346 / 0.8436 | 0.2155 / 0.0838 | **current best**, epoch 14; test UAR 0.6961 |
| `crema_d_lora_r16` — LoRA rank 16 | **0.6230** | 0.7694 / 0.8266 | 0.8923 / 0.8846 | 0.3536 / 0.2314 | best epoch 25; test UAR 0.6093 |
| `lora_e30` — LoRA fine-tuned | **0.6272** | 0.7702 / 0.8313 | 0.8885 / 0.8846 | 0.3481 / 0.2221 | best epoch 27, ran all 30 |
| `frozen_probe` — backbone frozen | **0.6265** | 0.7686 / 0.8273 | 0.8936 / 0.8846 | 0.3564 / 0.2301 | best epoch 24, early-stopped at 29 |
| published — audeering | n/a | _tbd_ | _tbd_ | _tbd_ | not started |

The published model outputs arousal/valence/dominance, not emotion categories, so it can
never fill the 6-way column — the routing decision is the only common ground.

Backing run and routing JSONs are consolidated in this `results/` directory. The hosted copy
can remain in Google Drive, but it is no longer the only source of the reported evidence.

## Current best model

`crema_d_partial_top4` is the checkpoint used by the web demo. It unfreezes the top four
transformer layers and improved validation UAR from roughly 0.62–0.63 to **0.7511**. Its
speaker-held-out test UAR is **0.6961**. Under the demo's distress grouping, test routing
balanced accuracy is **0.8729**, escalate recall is **0.9107**, and routing false-alarm rate
is **0.1648**. These test values are final confirmation numbers, not tuning targets.

## Per-emotion recall — `lora_e30`

| NEU | HAP | SAD | ANG | FEA | DIS |
| --- | --- | --- | --- | --- | --- |
| 0.707 | 0.533 | 0.369 | 0.939 | 0.626 | 0.590 |

Top confusions: HAP→ANG 51, SAD→FEA 45, DIS→ANG 44. Anger is saturated at 0.94 and has been
in every run; sadness is consistently weakest. **Happy→angry being the largest confusion is
the valence limitation showing up directly** (`docs/REPORT_NOTES.md` §7) — and it is the
commercially expensive error, since it fast-tracks a delighted customer as if they were
furious.

_(`frozen_probe`'s per-emotion breakdown is in its JSON; the notebook cell that prints it has
not been run. Pull it before writing the report.)_

## Fine-tuning vs. the frozen probe — resolved

`lora_e30` (0.6272) and `frozen_probe` (0.6265) are **statistically tied** — a 0.0007 gap on
1,142 clips is noise, and the routing numbers are similarly close in both directions.

**Verified as a controlled comparison.** Same config file, same manifest, same seed (1337),
same 30-epoch budget, same `early_stopping_patience: 5`, same batch size, learning rate,
warmup, window length and augmentation. The only difference is
`model.finetune_strategy` (`lora` → `frozen`), which is the variable under test. Trainable
parameters differ as expected: 493,331 (0.52%) vs 198,419 (0.21%).

**Safe to write in the report:** with a matched training budget, LoRA fine-tuning provides
**no meaningful improvement** over a frozen backbone with only the head and layer-pooling
weights trained. That is a real finding, not a null result — it says the emotional signal is
already linearly available in WavLM-Base+'s frozen representations.

**Two caveats to state alongside it:**

1. `lora_e30` used all 30 epochs with its best at epoch 27, so early stopping never fired —
   it hit the budget ceiling rather than converging, and might still improve given longer.
   `frozen_probe` did converge. The comparison is fair at *"30 epochs each"*, which is what
   was intended, but it is not *"both trained to convergence"*.
2. LoRA here is rank 8 on `q_proj`/`v_proj` only. A different rank or target-module set is a
   Phase 5 ablation and might change this conclusion.

## Historical note — a mismatched-patience bug, and why it matters

An earlier version of this table showed `frozen_probe` (0.6265) clearly **beating**
`lora_e30` (0.5859). That was not a finding: `lora_e30` had run under
`early_stopping_patience: 3` while the probe got 5, so it was stopped before it finished
improving. After the config default was raised to 5 and `lora_e30` was rerun, the two tied.

**Do not quote 0.5859.** The rerun overwrote that JSON, so the number has no results file
behind it.

Worth carrying into every Phase 5 ablation: an ablation only means something if every setting
except the one under test is identical. The `config` block saved inside each
`results/<run_id>.json` records exactly what was used — check it before comparing two runs.
