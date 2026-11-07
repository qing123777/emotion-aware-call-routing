# CREMA-D Notebook Flow

This document explains how the Colab notebook organizes the CREMA-D dataset preparation, inspection, training, and result-reading workflow.

## Section 3: Drive Setup

Purpose: store the dataset in Google Drive so it remains available after the Colab runtime disconnects.

1. Put `crema_d.zip` into Google Drive.
2. Keep the dataset available across Colab sessions.

Output:

```text
crema_d.zip stored in Google Drive
```

## Section 4: Unpack Audio

Purpose: extract the dataset from Google Drive into Colab local storage and locate the raw audio folder.

1. Extract `crema_d.zip` to `/content/crema_d`.
2. Find the `AudioWAV` folder.
3. Store its path as `AUDIO_DIR`.

Output:

```text
AUDIO_DIR = path to the folder containing the raw .wav files
```

## Section 5: Copy Project Code

Purpose: copy the uploaded project code from Google Drive into Colab local storage so scripts and imports run faster and reliably.

1. Copy the uploaded project folder from Drive to `/content/project`.
2. Check required files such as `src/data/crema_d.py`.
3. Set `/content/project` as the working directory.

Output:

```text
Colab can run project scripts and import src.data
```

## Section 6: Dependencies

Purpose: install the required Python packages that Colab does not already provide.

1. Install missing packages.
2. Prepare the Python environment for model training.

Output:

```text
Training libraries are available
```

## Section 7: Currently Missing

Purpose: this section is currently only a numbering issue.

1. There is currently no Section 7 in the notebook.
2. This is only caused by a missed section label.
3. The workflow continues directly from Section 6 to Section 8.

## Section 8: Manifest-Level Preprocessing

Purpose: create a structured manifest CSV from the raw CREMA-D audio filenames.

Run:

```bash
python scripts/prepare_crema_d.py \
    --audio-dir "{AUDIO_DIR}" \
    --out /content/crema_d_manifest.csv
```

Main files used:

```text
scripts/prepare_crema_d.py
src/data/crema_d.py
src/data/labels.py
```

What happens:

1. Read all `.wav` filenames from `AUDIO_DIR`.
2. Parse `actor_id`, sentence, emotion, and intensity.
3. Convert labels into structured values such as `emotion_index`, arousal, valence, and escalation.
4. Create speaker-independent train, validation, and test splits.
5. Check that speakers do not leak across splits.
6. Write the manifest CSV.

Output:

```text
/content/crema_d_manifest.csv
```

The manifest is an index table. It does not contain the audio itself. Instead, each row stores the audio file path, labels, metadata, and split name.

## Section 8: Dataset Audit

Purpose: inspect whether the dataset is usable before training.

Run:

```bash
python scripts/audit_dataset.py \
    --audio-dir "{AUDIO_DIR}" \
    --manifest /content/crema_d_manifest.csv
```

Main files used:

```text
scripts/audit_dataset.py
src/data/crema_d.py
src/data/audio.py
```

What happens:

1. Check whether filenames are valid.
2. Open audio files to verify readability.
3. Inspect duration, silence, duplicates, and label distribution.
4. Check the intensity label distribution.

Important finding:

```text
81.7% of clips are labelled XX
```

This means most clips do not have a clear intensity label. Therefore, intensity is not used as a main fine-tuning target. The clips are still used for emotion fine-tuning.

Output:

```text
dataset_audit.json
```

This file is used for reporting and sanity checks.

## Section 9: Training-Time Audio Preprocessing

Purpose: train or fine-tune the selected model configuration.

Run:

```bash
python scripts/train.py --config configs/crema_d_baseline.yaml
```

Main files used:

```text
scripts/train.py
src/data/dataset.py
src/data/audio.py
src/data/crema_d.py
src/data/labels.py
```

What happens:

1. Read the manifest CSV.
2. For each row, load the actual `.wav` file.
3. Convert audio to mono, resample to 16 kHz, crop or pad to fixed length, and convert to PyTorch tensors.
4. Feed tensors `x` and labels `y` into the model.
5. Fine-tune or train the selected configuration.

Output:

```text
checkpoint files
training result JSON
```

The checkpoint files store the trained model parameters. The result JSON stores metrics, training history, config, and checkpoint path.

## Section 10: Read Results

Purpose: inspect and compare model performance.

1. Load result JSON files.
2. Display validation metrics, routing metrics, and layer weights.
3. Compare the fine-tuned run against the frozen baseline.

Output:

```text
Performance numbers for the report
```

## Section 11: Saved Outputs

Purpose: summarize the important files produced by the workflow.

Important outputs:

```text
manifest CSV
audit JSON
training result JSON
model checkpoints
```

## Overall Flow

```text
crema_d.zip in Google Drive
        ↓
Section 4 extracts audio and finds AUDIO_DIR
        ↓
Section 5 copies project code into Colab
        ↓
Section 6 installs dependencies
        ↓
Section 8 creates the manifest CSV
        ↓
Section 8 audits the dataset
        ↓
Section 9 trains / fine-tunes the model
        ↓
Section 10 reads results
        ↓
Section 11 summarizes saved outputs
```
