# Project Plan

**What we're building:** a system that listens to one thing a caller says to the automated
menu, and decides whether they need a real human instead of the robot.
**Team:** 3 people (Team 19). **Deadlines:** report + code **6 Sept 2026** · slides + video
**9 Sept 2026**.

Full detail behind every line here lives in other files, linked where it matters — this page
is the short version. Update it as things change.

---

## What's done ✅

*(Phase numbers in brackets are just anchors other docs point to — ignore them if unhelpful.)*

**Getting and checking the data** *(Phase 1)*
- Downloaded CREMA-D: 7,442 short clips, 91 actors, 6 emotions.
- Split people (not clips) into train/validation/test groups, so the same voice never shows
  up in two groups — otherwise the model could cheat by recognising *who's* talking instead
  of *how* they feel.
- Opened every single clip and checked it: 1 near-silent clip, 3 accidental duplicates, no
  broken files. Also found most clips are much shorter than our 4-second window, so over a
  third of that window is empty padding — a reason to try shorter windows later.

**Building the model** *(Phase 3)*
- Used a pretrained speech model (WavLM) as the base, with a small trainable piece on top
  that turns it into 6 emotion scores.
- Built 4 switchable ways to train it: freeze everything, freeze most of it, unfreeze
  everything, or a lightweight adapter method called LoRA. Details:
  [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §5.
- Balanced training so common emotions don't drown out rare ones.
- Saves progress every round, so a crash doesn't lose everything.

**Testing trained versions**
- Fine-tuned (`lora_e30`): scored **0.627** on naming the right emotion (out of 1.0).
- Not fine-tuned, just the small trainable piece (`frozen_probe`): scored **0.627** too —
  basically tied. (`lora_e30` used its whole training budget without levelling off, so this
  isn't fully final — it might pull ahead with more training time.)
- **Finding worth stating plainly in the report:** fine-tuning barely helped here.
- Turned both into the real yes/no decision ("send to a human or not") — both land around
  77–83% right on that decision, better than the raw 63% emotion-naming score, because some
  wrong emotion guesses still land on the right decision anyway.
- Unfreezing the top four WavLM layers produced the current best run: validation UAR
  **0.7511**, held-out test UAR **0.6961**.

**Local web demo** *(Phase 6)*
- Built one local Gradio page with microphone and upload inputs, shared preprocessing,
  six-emotion probabilities, and a clear human-agent/standard-menu route.
- Added checkpoint loading plus validation for empty, short, silent, non-finite, stereo,
  different-sample-rate, and longer-than-four-second audio.
- Verified the actual partial-top-four checkpoint on CPU and tested the rendered browser UI.

**Housekeeping** *(Phase 0)*
- Google Drive folder set up and shared with the team.
- Team registered on the course spreadsheet.

Numbers and reasoning in full: [`results/master_table.md`](results/master_table.md).

---

## What's not done yet

*This list is "what's left and why it matters." For **what has to be built before what** —
e.g. the demo can't start until the model-loading helper exists — see the build-order table
in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §8.*

- **Get a third model to compare against** *(Phase 2)* — download an existing,
  already-trained emotion model and see how it does, as an outside reference point, not just
  our own two versions. Not started.

- **Run the bigger batch of experiments** *(Phase 5 — worth over a third of the grade,
  hasn't started)*. Try the 2 training styles we haven't tested yet; try shorter/longer audio
  clips; flip settings on/off one at a time to see what actually matters; repeat the best
  setup on different people-splits to check the score isn't a fluke; try the model on a
  completely different dataset it's never seen, to see if it still holds up.

- **Explain the happy/angry mix-up** *(part of Phase 5)* — two small side experiments showing
  this confusion is a known hard problem in the field, not a bug in our model.

- **Streaming inference + escalation state machine** *(Phase 4)* — a rolling status light
  (calm/caution/escalated) for the presentation video. A demo only, not something we score.
  It's really two jobs under one name:

  | Piece | What problem it solves |
  | --- | --- |
  | **Streaming inference** — keep the last 4 seconds in a buffer, take a window every 1 s | This is what lets audio **longer than 4 seconds** be used at all. Our model only ever accepts a fixed 4-second window, so long audio has to be chopped into overlapping 4-second pieces first. |
  | **Escalation state machine** — calm → caution → escalated, with the wait-a-few-seconds timers | Nothing to do with length. It's what you need *because* of streaming: once you get a new prediction every second instead of one per call, something has to stop the status light flickering and decide when the alert genuinely fires. |

  The single-utterance app handles longer recordings by centre-cropping one four-second slice
  and reporting that choice to the user. Streaming is still needed if every part of a long
  call must be analysed. Diagram and detail:
  [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §6.

- **Write the report** *(Phase 6)* — framing and source notes are ready; the writing itself
  hasn't started.

- **Record the demo video and finish the slides.** *(Phase 6)*

- **Fill in the AI-use declaration** *(Phase 6)* — who on the team did what.

- **Package for submission** *(Phase 6)* — zip under 5MB, no dataset or trained weights inside.

- **Put real names on the ownership table below** *(Phase 0)* — still placeholders.

---

## Who's doing what

| Job | Person | Status |
| --- | --- | --- |
| Data + model + both baseline results | Member 1 | done |
| Single-utterance web demo | Member 2 | done |
| Optional streaming inference/escalation state machine | Member 2 | not started |
| Bigger experiments + report | Member 3 | not started |

The bigger-experiments job carries the most marks and hasn't started — if anyone needs a
task, that's it.

---

## Decisions worth remembering

- **CREMA-D** as the main dataset — free and ready now, no licence wait.
- **Score by average-per-emotion, not plain accuracy** — otherwise a model that ignores rare
  emotions looks artificially good.
- **LoRA** as the main training method, with the other 3 kept as easy one-setting comparisons.
- **No "watch the whole call continuously" system** — the dataset is short isolated clips with
  no real conversations, so that can't be fairly tested. Instead: one decision per one thing
  said, and the live "watch it happen" view stays a demo, not a scored result.
- **Code lives in Google Drive, not GitHub** — team preference.
- **Compute runs on Google Colab, not Kaggle** — Kaggle needed phone verification we couldn't
  complete.
- **Not training on "how happy / how energetic" numbers** — those are just a fixed calculation
  from the 6 emotions, so training on them separately wouldn't add anything new.

---

*Note: "false alarm rate" is reported two slightly different ways depending on the file — see
[`results/master_table.md`](results/master_table.md) before quoting either.*

More detail: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) (how it works) ·
[`results/master_table.md`](results/master_table.md) (all the numbers).
