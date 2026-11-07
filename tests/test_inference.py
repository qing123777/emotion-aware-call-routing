from __future__ import annotations

import gc
import unittest
import warnings

import numpy as np
import torch

from app import analyse_audio, create_app, predict_outcome, render_outcome
from src.inference import DATASET_MATCH_RMS_DBFS, EmotionPredictor, prepare_audio, resolve_device


class FixedEmotionModel(torch.nn.Module):
    """Tiny stand-in that makes Angry the unambiguous top prediction."""

    def forward(self, waveform, num_valid=None):
        assert waveform.shape == (1, 64_000)
        assert num_valid is not None
        return {"emotion": torch.tensor([[0.0, 0.5, 0.2, 3.0, -0.5, 0.1]], device=waveform.device)}


def voiced_pcm(sample_rate: int, seconds: float, channels: int = 1) -> np.ndarray:
    time = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    mono = (np.sin(2 * np.pi * 220.0 * time) * 12_000).astype(np.int16)
    return mono if channels == 1 else np.column_stack([mono] * channels)


class AudioPreparationTests(unittest.TestCase):
    def test_stereo_pcm_is_mono_resampled_and_padded(self):
        prepared = prepare_audio((8_000, voiced_pcm(8_000, 1.0, channels=2)))
        self.assertEqual(prepared.waveform.shape, (64_000,))
        self.assertEqual(prepared.waveform.dtype, np.float32)
        self.assertEqual(prepared.num_valid, 16_000)
        self.assertTrue(prepared.was_resampled)
        self.assertFalse(prepared.was_cropped)
        self.assertLessEqual(float(np.abs(prepared.waveform).max()), 1.0)

    def test_quiet_and_loud_speech_are_matched_to_the_same_level(self):
        sample_rate = 16_000
        time = np.arange(sample_rate, dtype=np.float32) / sample_rate
        tone = np.sin(2 * np.pi * 220.0 * time)

        quiet = prepare_audio((sample_rate, tone * 0.02))
        loud = prepare_audio((sample_rate, tone * 0.70))

        self.assertAlmostEqual(quiet.output_rms_dbfs, DATASET_MATCH_RMS_DBFS, places=1)
        self.assertAlmostEqual(loud.output_rms_dbfs, DATASET_MATCH_RMS_DBFS, places=1)
        self.assertAlmostEqual(quiet.output_rms_dbfs, loud.output_rms_dbfs, places=1)
        self.assertGreater(quiet.level_gain_db, 0.0)
        self.assertLess(loud.level_gain_db, 0.0)
        self.assertLessEqual(float(np.abs(quiet.waveform).max()), 0.98)
        self.assertLessEqual(float(np.abs(loud.waveform).max()), 0.98)

    def test_edge_silence_is_trimmed_but_a_speech_margin_is_retained(self):
        sample_rate = 16_000
        silence = np.zeros(sample_rate // 2, dtype=np.float32)
        time = np.arange(sample_rate, dtype=np.float32) / sample_rate
        speech = 0.2 * np.sin(2 * np.pi * 220.0 * time)

        prepared = prepare_audio((sample_rate, np.concatenate([silence, speech, silence])))

        self.assertGreater(prepared.trimmed_seconds, 0.5)
        self.assertLess(prepared.trimmed_seconds, 1.0)
        self.assertGreater(prepared.analysed_duration_seconds, 1.0)
        self.assertLess(prepared.analysed_duration_seconds, 1.5)

    def test_dc_offset_is_removed_before_level_matching(self):
        sample_rate = 16_000
        time = np.arange(sample_rate, dtype=np.float32) / sample_rate
        biased = 0.15 + 0.02 * np.sin(2 * np.pi * 220.0 * time)

        prepared = prepare_audio((sample_rate, biased))

        valid = prepared.waveform[: prepared.num_valid]
        self.assertAlmostEqual(float(valid.mean()), 0.0, places=5)
        self.assertAlmostEqual(prepared.output_rms_dbfs, DATASET_MATCH_RMS_DBFS, places=1)

    def test_long_audio_is_centre_cropped(self):
        prepared = prepare_audio((16_000, voiced_pcm(16_000, 5.0)))
        self.assertEqual(prepared.num_valid, 64_000)
        self.assertTrue(prepared.was_cropped)
        self.assertAlmostEqual(prepared.original_duration_seconds, 5.0)

    def test_empty_short_silent_and_non_finite_audio_are_rejected(self):
        invalid = [
            None,
            (16_000, np.array([], dtype=np.int16)),
            (16_000, np.ones(1_000, dtype=np.float32)),
            (16_000, np.zeros(16_000, dtype=np.float32)),
            (16_000, np.full(16_000, np.nan, dtype=np.float32)),
        ]
        for audio in invalid:
            with self.subTest(audio_type=type(audio).__name__):
                with self.assertRaises(ValueError):
                    prepare_audio(audio)

    def test_unavailable_cuda_fails_clearly(self):
        if not torch.cuda.is_available():
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                resolve_device("cuda")


class PredictorAndAppTests(unittest.TestCase):
    def setUp(self):
        self.predictor = EmotionPredictor(FixedEmotionModel())
        self.audio = (16_000, voiced_pcm(16_000, 1.0))

    def test_prediction_probabilities_and_distress_route(self):
        result = self.predictor.predict(self.audio)
        self.assertEqual(result.emotion_code, "ANG")
        self.assertEqual(result.emotion_name, "Angry")
        self.assertTrue(result.should_escalate)
        self.assertAlmostEqual(sum(result.probabilities.values()), 1.0, places=6)

    def test_ui_adapter_returns_useful_success_and_error_states(self):
        result, probabilities, details = analyse_audio(self.predictor, self.audio, view="technical")
        self.assertIn("Angry", result)
        self.assertIn("Connect to a human agent", result)
        self.assertEqual(len(probabilities), 6)
        self.assertIn("Input:", details)
        self.assertIn("Level:", details)
        self.assertIn("dBFS", details)

        result, probabilities, details = analyse_audio(self.predictor, None)
        self.assertIn("Unable to analyse", result)
        self.assertNotIn("route-card", result)
        self.assertEqual(probabilities, {})
        self.assertIn("Record an utterance", details)

    def test_visitor_view_hides_signal_jargon(self):
        result, _, details = analyse_audio(self.predictor, self.audio, view="visitor")
        self.assertIn("Connect to a human agent", result)
        for jargon in ("dBFS", "distress routing group", "Hz"):
            self.assertNotIn(jargon, result + details)
        self.assertGreaterEqual(details.count("\n- ") + details.startswith("- "), 3)

    def test_technical_view_lists_preprocessing_numbers(self):
        result, _, details = analyse_audio(self.predictor, self.audio, view="technical")
        self.assertIn("distress routing group", result)
        self.assertIn("dBFS", details)
        self.assertIn("16,000 Hz", details)
        self.assertGreaterEqual(details.count("\n- ") + details.startswith("- "), 5)

    def test_switching_view_rewords_the_same_result_without_rerunning(self):
        outcome = predict_outcome(self.predictor, self.audio)
        visitor = render_outcome(outcome, "visitor")
        technical = render_outcome(outcome, "technical")
        self.assertEqual(visitor[1], technical[1])  # same probabilities
        self.assertIn("Angry", visitor[0])
        self.assertIn("Angry", technical[0])
        self.assertNotEqual(visitor[2], technical[2])

        error = predict_outcome(self.predictor, None)
        self.assertIsInstance(error, str)
        self.assertIn("Unable to analyse", render_outcome(error, "visitor")[0])

    def test_gradio_app_has_one_audio_input_with_mic_and_upload(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            try:
                app = create_app(self.predictor)
            except RuntimeError as exc:
                if "Gradio is not installed" in str(exc):
                    self.skipTest(str(exc))
                raise

            try:
                config = app.get_config_file()
                audio_components = [item for item in config["components"] if item["type"] == "audio"]
                self.assertEqual(len(audio_components), 1)
                self.assertEqual(set(audio_components[0]["props"]["sources"]), {"microphone", "upload"})

                # Analyse starts disabled: there is nothing to analyse until audio arrives.
                buttons = [item for item in config["components"] if item["type"] == "button"]
                analyse = [b for b in buttons if b["props"].get("value") == "Analyse emotion"]
                self.assertEqual(len(analyse), 1)
                self.assertFalse(analyse[0]["props"]["interactive"])

                # Input and result panes are both visible from the start.
                panes = {
                    tuple(item["props"].get("elem_classes", []))
                    for item in config["components"]
                    if item["type"] == "column" and item["props"].get("visible", True)
                }
                self.assertIn(("pane", "input-pane"), panes)
                self.assertIn(("pane", "result-pane"), panes)

                # Layout "Focus result": the result pane is wider than the input
                # pane, and the probabilities/details are open by default.
                scales = {
                    tuple(item["props"].get("elem_classes", [])): item["props"]["scale"]
                    for item in config["components"]
                    if item["type"] == "column"
                }
                self.assertGreater(scales[("pane", "result-pane")], scales[("pane", "input-pane")])
                accordions = [item for item in config["components"] if item["type"] == "accordion"]
                self.assertEqual(len(accordions), 1)
                self.assertTrue(accordions[0]["props"]["open"])

                api_names = {item["api_name"] for item in config["dependencies"]}
                self.assertIn("analyse", api_names)

                # One view switcher, defaulting to the visitor wording.
                dropdowns = [item for item in config["components"] if item["type"] == "dropdown"]
                self.assertEqual(len(dropdowns), 1)
                self.assertEqual(dropdowns[0]["props"]["value"], "visitor")
                choice_values = {choice[1] for choice in dropdowns[0]["props"]["choices"]}
                self.assertEqual(choice_values, {"visitor", "technical"})
            finally:
                app.close()
                del app
                gc.collect()


if __name__ == "__main__":
    unittest.main()
