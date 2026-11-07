"""Local browser demo for the speech-emotion routing model."""

from __future__ import annotations

import argparse
import html
import sys
import time
from pathlib import Path
from typing import Any, Tuple

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference import (  # noqa: E402
    DATASET_MATCH_RMS_DBFS,
    DEFAULT_WINDOW_SECONDS,
    EmotionPredictor,
    Prediction,
)

DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "crema_d_partial_top4_best.pt"

# Layout ("Focus result", chosen from prototype_wide_layout.py): a compact
# header, then a narrower input pane (5) beside a wider result pane (7) with the
# probabilities open, so the whole task fits in one viewport. The page grows with
# the screen up to 1600px instead of sitting in a narrow centre column, and steps
# type and padding up at 1440px. Columns stack below 760px.
# Spacing uses a 4px base: 8 / 12 / 16 / 24 / 32.
APP_CSS = """
:root {
    --ink: #111827;
    --ink-2: #334155;
    --ink-3: #5b6678;
    --page: #ffffff;
    --surface: #ffffff;
    --line: #d8dee9;
    --navy: #16324f;
    --navy-deep: #0b2239;
    --escalate: #4a1022;
    --escalate-line: #be4668;
    --standard: #12364a;
    --standard-line: #4d91ad;
}
/* One plain page colour everywhere: Gradio paints html, body, the app element and
   the container separately, which left lighter strips down the page edges. */
html, body, gradio-app, .gradio-container, .gradio-container .main, .gradio-container .wrap {
    background: var(--page) !important;
}
.gradio-container {
    max-width: min(1600px, 100%) !important;
    width: 100% !important;
    margin: 0 auto !important;
    padding: 32px clamp(16px, 4vw, 72px) 40px !important;
}
footer { display: none !important; }
::selection { background: #c7d6ea; color: var(--ink); }

/* Header row: title and intro on the left, view dropdown pinned top-right. */
.header-row {
    align-items: flex-start !important;
    flex-wrap: nowrap !important;
    gap: 24px !important;
    margin-bottom: 24px;
}
/* Gradio wraps the header HTML in a block sized to its text and the dropdown in
   a <form>; let the text shrink and give the form a fixed width so the row never
   wraps the dropdown underneath on desktop. */
.header-row > :first-child { flex: 1 1 0 !important; min-width: 0 !important; }
.header-row > .form {
    flex: 0 0 200px !important;
    overflow: visible !important;
    background: transparent !important;
    border: none !important;
}
.header-row > .form > .view-switch {
    flex: 1 1 auto !important;
    width: 100% !important;
    min-width: 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
.pane-title ul, .limitation ul { margin: 6px 0 0; padding-left: 18px; }
.pane-title li, .limitation li { margin: 2px 0; }
.app-header { margin: 0; }
.app-header h1 {
    color: var(--ink);
    font-size: 2rem;
    letter-spacing: -.02em;
    line-height: 1.15;
    margin: 0 0 8px;
}
.app-header p { color: var(--ink-3); font-size: 1rem; line-height: 1.6; margin: 0; max-width: 68ch; }

.workspace { align-items: stretch !important; gap: 24px !important; }
.pane {
    background: var(--surface) !important;
    border: 1px solid var(--line) !important;
    border-radius: 14px !important;
    box-shadow: 0 8px 24px rgba(15, 23, 42, .06) !important;
    padding: 20px !important;
    gap: 16px !important;
}
.pane-title h2 { color: var(--ink); font-size: 1.125rem; margin: 0 0 4px; }
.pane-title p { color: var(--ink-3); font-size: .9rem; line-height: 1.5; margin: 0; }

.primary-action {
    background: var(--navy) !important;
    border: 1px solid var(--navy) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    min-height: 48px;
}
.primary-action:hover:not(:disabled) { background: var(--navy-deep) !important; }
.primary-action:focus-visible { outline: 3px solid #f59e0b !important; outline-offset: 2px !important; }
.primary-action:disabled { background: #94a3b8 !important; border-color: #94a3b8 !important; cursor: not-allowed; }

.result-card, .route-card, .empty-card { border-radius: 12px; padding: 18px 20px; }
.empty-card {
    background: #eef1f6;
    border: 1px dashed #94a3b8;
    color: var(--ink-2);
    line-height: 1.55;
    min-height: 196px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.empty-card strong { color: var(--ink); display: block; font-size: 1.05rem; margin-bottom: 4px; }
.result-card { background: var(--ink); color: #ffffff; margin-bottom: 12px; }
.result-card.error { background: #4c1d1d; }
.result-label { color: #93c5fd; font-size: .78rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; }
.result-card.error .result-label { color: #fca5a5; }
.result-value { color: #ffffff; font-size: 2.25rem; font-weight: 800; line-height: 1.1; margin-top: 6px; }
.result-confidence { color: #dbeafe; margin-top: 4px; }
.route-card { color: #fff7ed; }
.route-escalate { background: var(--escalate); border: 1px solid var(--escalate-line); }
.route-standard { background: var(--standard); border: 1px solid var(--standard-line); }
.route-title { font-size: 1.2rem; font-weight: 800; }
.route-copy { line-height: 1.5; margin-top: 4px; opacity: .9; }

.routing-legend { color: var(--ink-3); font-size: .875rem; line-height: 1.7; margin: 0; }
.routing-legend strong { color: var(--ink-2); }
.dot { border-radius: 50%; display: inline-block; height: 8px; margin-right: 6px; vertical-align: 1px; width: 8px; }
.dot-escalate { background: var(--escalate-line); }
.dot-standard { background: var(--standard-line); }

.limitation { color: var(--ink-3); font-size: .85rem; line-height: 1.6; margin: 24px 0 0; max-width: 90ch; }
.limitation strong { color: var(--ink-2); }

/* Gradio's ".gradio-container-X .prose *" rule paints every text node in gr.HTML
   with its body colour, which beats the single-class rules above and made the
   routing card dark-on-dark. Re-assert text colours at higher specificity. */
.gradio-container .prose .app-header h1,
.gradio-container .prose .pane-title h2,
.gradio-container .prose .empty-card strong { color: var(--ink); }
.gradio-container .prose .app-header p,
.gradio-container .prose .pane-title p,
.gradio-container .prose .routing-legend,
.gradio-container .prose .limitation { color: var(--ink-3); }
.gradio-container .prose .empty-card,
.gradio-container .prose .routing-legend strong,
.gradio-container .prose .limitation strong { color: var(--ink-2); }
.gradio-container .prose .result-label { color: #93c5fd; }
.gradio-container .prose .result-card.error .result-label { color: #fca5a5; }
.gradio-container .prose .result-value { color: #ffffff; }
.gradio-container .prose .result-confidence { color: #dbeafe; }
.gradio-container .prose .route-title,
.gradio-container .prose .route-copy { color: #fff7ed; }

@media (min-width: 1440px) {
    .app-header h1 { font-size: 2.4rem; }
    .app-header p { font-size: 1.1rem; }
    .pane { padding: 28px !important; }
    .empty-card { min-height: 260px; }
}

@media (max-width: 760px) {
    .gradio-container { padding: 20px 16px 32px !important; }
    .app-header h1 { font-size: 1.6rem; }
    .header-row { flex-direction: column-reverse !important; gap: 12px !important; }
    .header-row > .form { flex: 0 0 auto !important; width: 100%; }
    .workspace { flex-direction: column !important; }
}
"""

# APP_CSS paints a light page. Gradio follows the OS dark-mode setting by adding a
# "dark" class to <body>, which re-colours its own text light-on-light and blacks out
# the page edges. Removing the class on load keeps the one designed colour scheme.
FORCE_LIGHT_JS = "() => { document.body.classList.remove('dark'); }"

ROUTING_LEGEND = (
    '<p class="routing-legend">'
    '<span class="dot dot-escalate"></span><strong>Human agent:</strong> '
    "Angry · Fearful · Sad · Disgusted<br>"
    '<span class="dot dot-standard"></span><strong>Automated menu:</strong> Neutral · Happy'
    "</p>"
)


VIEW_CHOICES = [("General visitor", "visitor"), ("Technical", "technical")]

# Page wording per view. Both views describe the same model and pipeline; the
# visitor view drops signal-processing terms, the technical view lists them.
PAGE_COPY = {
    "visitor": {
        "intro": "Say one short sentence, the way a caller would. The model guesses how the caller "
        "feels and decides if they should skip the phone menu and talk to a person.",
        "input": "<p>Record or upload 1–4 seconds of speech.</p>",
        "empty": "<strong>No result yet</strong>Record or upload a clip, then press Analyse.",
        "limits": "<strong>Good to know</strong><ul>"
        "<li>The model learned from actors, so treat the result as a guess, not a fact.</li>"
        "<li>Your audio never leaves this computer.</li></ul>",
    },
    "technical": {
        "intro": "Single-utterance call routing. A fine-tuned WavLM-Base+ classifies six CREMA-D "
        "emotions; Angry, Fearful, Sad and Disgusted route to a human agent.",
        "input": "<p>Microphone or file, any sample rate. Preprocessing matches training:</p><ul>"
        "<li>mono, DC offset removed, resampled to 16 kHz</li>"
        f"<li>edge silence trimmed, speech level set to {DATASET_MATCH_RMS_DBFS:.0f} dBFS</li>"
        f"<li>padded or centre-cropped to {DEFAULT_WINDOW_SECONDS:.1f} s</li></ul>",
        "empty": "<strong>No prediction yet</strong>Provide audio and press Analyse.",
        # Test figures: results/master_table.md, backed by crema_d_partial_top4.json
        # (test UAR 0.6961) and crema_d_partial_top4_routing_test.json (recall 0.9107).
        "limits": "<strong>Limitations</strong><ul>"
        "<li>Trained on acted speech (CREMA-D: 91 actors, 12 sentences), not real calls.</li>"
        "<li>Speaker-held-out test: UAR 0.696, escalate recall 0.91.</li>"
        "<li>Runs locally; clips over 4 s are centre-cropped.</li></ul>",
    },
}

NO_DETAILS = "No audio has been processed yet."


def _header_html(view: str = "visitor") -> str:
    return (
        '<header class="app-header"><h1>Speech Emotion Detection Demo</h1>'
        f'<p>{PAGE_COPY[view]["intro"]}</p></header>'
    )


def _input_intro_html(view: str = "visitor") -> str:
    return f'<div class="pane-title"><h2>Caller audio</h2>{PAGE_COPY[view]["input"]}</div>'


def _limitation_html(view: str = "visitor") -> str:
    return f'<div class="limitation">{PAGE_COPY[view]["limits"]}</div>'


def _empty_result(view: str = "visitor") -> str:
    # The empty state teaches the one task on the page instead of saying "nothing here".
    return f'<div class="empty-card">{PAGE_COPY[view]["empty"]}</div>'


def _prediction_html(result: Prediction) -> str:
    return (
        '<div class="result-card">'
        '<div class="result-label">Detected emotion</div>'
        f'<div class="result-value">{html.escape(result.emotion_name)}</div>'
        f'<div class="result-confidence">{result.confidence:.1%} model confidence</div>'
        "</div>"
    )


VIEWS = ("visitor", "technical")
DEFAULT_VIEW = "visitor"

# Route-card explanation per view. The visitor wording avoids the project's
# internal term "distress routing group"; the decision itself is identical.
ROUTE_COPY = {
    "visitor": {
        True: "The caller sounds upset, so a person should take this call.",
        # "calm" would be wrong: this branch also covers Happy callers.
        False: "The caller doesn't sound upset, so the automated menu can handle this call.",
    },
    "technical": {
        True: "The detected emotion falls inside the project’s distress routing group.",
        False: "The detected emotion falls outside the project’s distress routing group.",
    },
}


def _route_html(result: Prediction, view: str = DEFAULT_VIEW) -> str:
    copy = ROUTE_COPY[view][result.should_escalate]
    if result.should_escalate:
        css_class = "route-escalate"
        title = "Connect to a human agent"
    else:
        css_class = "route-standard"
        title = "Continue with the standard menu"
    return (
        f'<div class="route-card {css_class}">'
        f'<div class="route-title">{title}</div>'
        f'<div class="route-copy">{copy}</div>'
        "</div>"
    )


def _details_markdown(result: Prediction, view: str = DEFAULT_VIEW) -> str:
    """Audio-processing details as a markdown bullet list.

    Visitors get three plain-language lines; technical readers get every
    preprocessing number, so the two views differ in depth, not in facts.
    """
    audio = result.audio
    if view == "visitor":
        lines = [
            f"Clip length: {audio.original_duration_seconds:.1f} s",
            "Cleaned up: quiet edges trimmed, volume evened out",
            f"Analysis took {result.inference_seconds:.1f} s",
        ]
    else:
        # Wording matches the approved prototype (prototype/copy-variants branch).
        window = "centre-cropped" if audio.was_cropped else "padded"
        lines = [
            f"Input: {audio.original_duration_seconds:.2f} s at {audio.original_sample_rate:,} Hz",
            f"Edge silence trimmed: {audio.trimmed_seconds:.2f} s",
            f"Level: {audio.input_rms_dbfs:.1f} → {audio.output_rms_dbfs:.1f} dBFS "
            f"({audio.level_gain_db:+.1f} dB)",
            f"Window: {window} to {DEFAULT_WINDOW_SECONDS:.1f} s",
            f"Inference: {result.inference_seconds:.2f} s",
        ]
    return "\n".join(f"- {line}" for line in lines)


def analyse_audio(
    predictor: EmotionPredictor, audio: Any, view: str = DEFAULT_VIEW
) -> Tuple[str, dict[str, float], str]:
    """UI-safe adapter around the predictor, including actionable errors.

    Returns (result HTML, probabilities, details markdown). The result HTML holds
    both the emotion card and the routing card so they always update together.
    `view` picks the wording ("visitor" or "technical"); the prediction is the same.
    """
    return render_outcome(predict_outcome(predictor, audio), view)


def predict_outcome(predictor: EmotionPredictor, audio: Any) -> Prediction | str:
    """Run the model; return the Prediction, or the error message to show the user.

    Kept separate from rendering so the page can re-word the same result when the
    view changes, without running the model again.
    """
    try:
        return predictor.predict(audio)
    except (ValueError, RuntimeError) as exc:
        return str(exc)


def render_outcome(outcome: Prediction | str, view: str = DEFAULT_VIEW) -> Tuple[str, dict[str, float], str]:
    """Turn a model outcome into (result HTML, probabilities, details markdown).

    A str outcome is an error message from predict_outcome. An unknown view falls
    back to the default instead of raising: the view only changes wording, so a
    bad value should never stop the user from seeing their result.
    """
    if view not in VIEWS:
        view = DEFAULT_VIEW
    if isinstance(outcome, str):
        message = html.escape(outcome)
        error = (
            '<div class="result-card error">'
            '<div class="result-label">Unable to analyse</div>'
            f'<div class="result-confidence">{message}</div>'
            "</div>"
        )
        return error, {}, f"**Input needs attention:** {message}"
    return (
        _prediction_html(outcome) + _route_html(outcome, view),
        outcome.probabilities,
        _details_markdown(outcome, view),
    )


def create_app(predictor: EmotionPredictor):
    """Create the Gradio interface without loading a second model."""
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("Gradio is not installed. Run: pip install -r requirements.txt") from exc

    with gr.Blocks(title="Speech Emotion Detection Demo") as demo:
        # The last model outcome (Prediction or error text), so switching the view
        # re-words the current result instead of discarding it.
        last_outcome = gr.State(None)

        with gr.Row(equal_height=False, elem_classes="header-row"):
            header_output = gr.HTML(_header_html(DEFAULT_VIEW))
            view_select = gr.Dropdown(
                choices=VIEW_CHOICES,
                value=DEFAULT_VIEW,
                label="View for",
                scale=0,
                min_width=180,
                elem_classes="view-switch",
            )

        with gr.Row(equal_height=False, elem_classes="workspace"):
            with gr.Column(scale=5, elem_classes=["pane", "input-pane"]):
                input_intro = gr.HTML(_input_intro_html(DEFAULT_VIEW))
                # One component with both sources: the user picks mic or upload
                # inside it, so there is no extra gate before the task starts.
                audio_input = gr.Audio(
                    label="Caller audio",
                    sources=["microphone", "upload"],
                    type="numpy",
                )
                analyse_button = gr.Button(
                    "Analyse emotion",
                    variant="primary",
                    interactive=False,
                    elem_classes="primary-action",
                )

            with gr.Column(scale=7, elem_classes=["pane", "result-pane"]):
                gr.HTML('<div class="pane-title"><h2>Result</h2></div>')
                result_output = gr.HTML(_empty_result(DEFAULT_VIEW), label="Result")
                gr.HTML(ROUTING_LEGEND)
                with gr.Accordion("Emotion probabilities and audio details", open=True):
                    probability_output = gr.Label(label="Emotion probabilities", num_top_classes=6)
                    details_output = gr.Markdown(NO_DETAILS)

        limitation_output = gr.HTML(_limitation_html(DEFAULT_VIEW))

        result_outputs = [result_output, probability_output, details_output]

        def render_or_empty(outcome, view):
            if outcome is None:
                return _empty_result(view), None, NO_DETAILS
            return render_outcome(outcome, view)

        # Analyse is only clickable once there is audio to analyse. Any change to
        # the audio (new clip or cleared) resets the result, so a decision for the
        # previous clip is never shown next to a different one.
        def on_audio_change(audio, view):
            return (gr.update(interactive=audio is not None), None, *render_or_empty(None, view))

        def on_analyse(audio, view):
            outcome = predict_outcome(predictor, audio)
            return (outcome, *render_outcome(outcome, view))

        def on_view_change(view, outcome):
            return (
                _header_html(view),
                _input_intro_html(view),
                _limitation_html(view),
                *render_or_empty(outcome, view),
            )

        audio_input.change(
            on_audio_change,
            inputs=[audio_input, view_select],
            outputs=[analyse_button, last_outcome, *result_outputs],
            api_name=False,
        )
        analyse_button.click(
            on_analyse,
            inputs=[audio_input, view_select],
            outputs=[last_outcome, *result_outputs],
            api_name="analyse",
        )
        view_select.change(
            on_view_change,
            inputs=[view_select, last_outcome],
            outputs=[header_output, input_intro, limitation_output, *result_outputs],
            api_name=False,
        )

    return demo.queue(default_concurrency_limit=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="trained .pt checkpoint")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="ask Gradio for a temporary public link")
    parser.add_argument("--inbrowser", action="store_true", help="open the local URL automatically")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not Path(args.checkpoint).is_file():
        # The checkpoint is too large for git, so a fresh clone never has it.
        print(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Download it first with:  python scripts/download_checkpoint.py",
            file=sys.stderr,
        )
        return 1
    started = time.perf_counter()
    print(f"Loading checkpoint: {Path(args.checkpoint).resolve()}", flush=True)
    predictor = EmotionPredictor.from_checkpoint(args.checkpoint, device=args.device)
    print(
        f"Model ready on {predictor.device} in {time.perf_counter() - started:.1f}s. "
        f"Checkpoint: {predictor.checkpoint_name}",
        flush=True,
    )
    demo = create_app(predictor)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=args.inbrowser,
        show_error=True,
        css=APP_CSS,
        js=FORCE_LIGHT_JS,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
