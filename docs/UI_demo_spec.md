# Local web demo

Status: **implemented and tested** in `app.py` and `src/inference.py`.

The demo is one local Gradio process, so there is no separate frontend service to configure.
Its initial state is deliberately minimal: a centred title and explanation, the two routing
groups, and separate **Record audio** and **Upload audio** buttons. Choosing one opens a
floating panel anchored below the buttons; the other panel remains closed. Both routes
produce the same decoded `(sample rate, samples)` value and therefore use the same validation
and preprocessing path.

## Processing and inference

For each utterance the backend:

1. rejects missing, empty, shorter-than-0.25-second, silent, non-finite, or unsupported audio;
2. converts ordinary mono/stereo PCM to finite float32 mono audio;
3. removes DC microphone bias and trims leading/trailing silence while retaining a short
   natural margin around speech;
4. matches the trimmed utterance to an approximate -24 dBFS dataset-style speech level, with
   bounded amplification and a 0.98 peak ceiling;
5. resamples to 16 kHz when needed;
6. right-pads short utterances or centre-crops long ones to the trained four-second window;
7. runs `crema_d_partial_top4_best.pt` through the reconstructed WavLM model; and
8. returns all six probabilities, the highest-probability emotion, and the routing decision.

The raw CREMA-D WAV files are not included in the local demo bundle, so -24 dBFS is an
explicit conservative approximation, not a claimed measurement of the corpus median. The
details accordion reports the input RMS, output RMS, and applied gain for every request so the
assumption remains visible and can later be replaced by a measured target.

The model loads once at startup. On the tested CPU, a warm two-second synthetic request took
about 0.4 seconds for inference; startup took roughly 36 seconds with the backbone cached.

## Routing rule and display

After a successful analysis both input panels close and a new result section appears below
the two choice buttons. It displays the predicted emotion, confidence, and one of two routes:

- **Connect to a human agent** for the evaluated distress group: angry, fearful, sad, or
  disgusted.
- **Continue with the standard menu** for neutral or happy.

The distress grouping is explicit because routing is a product choice rather than ground
truth. On the partial-top-four held-out test split it reached 0.8729 balanced accuracy, versus
0.8258 for the narrower angry/disgusted grouping. The route cards use dark burgundy or dark
blue surfaces with high-contrast text instead of pale status colours. Probability and audio
details stay collapsed by default so the main decision remains readable during a presentation.

Invalid input does not close the selected panel. It displays an actionable message in that
panel so the user can record or upload another clip. Each panel can also be minimised without
running the model.

## Scope and limitations

Each recording is one independent utterance. This demo does not stream, smooth predictions,
or keep an escalation state between requests. Long audio is visibly reported as centre-cropped
rather than silently truncated. The page also states that CREMA-D is acted speech and that a
prediction is not a factual reading of a person's feelings.

Automated coverage lives in `tests/test_inference.py`; the project-wide synthetic regression
suite remains `scripts/smoke_test.py`.
