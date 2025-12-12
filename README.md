# EmoSteer-TTS

> **Unofficial PyTorch Implementation**
> Fine-Grained and Training-Free Emotion-Controllable Text-to-Speech via Activation Steering

[![arXiv](https://img.shields.io/badge/arXiv-2508.03543-b31b1b.svg)](https://arxiv.org/abs/2508.03543)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

<p align="center">
  <em>This is NOT an official implementation from the original authors</em>
</p>

---

## Overview

**EmoSteer-TTS** is a training-free method that enables fine-grained, continuous emotion control in Text-to-Speech systems by directly steering internal activations of pretrained flow-matching-based TTS models.

### Key Features

- **Training-Free**: No model retraining required
- **Fine-Grained Control**: Continuous emotion intensity adjustment via strength parameter α
- **Versatile**: Supports emotion conversion, interpolation, erasure, and composition
- **Model Agnostic**: Compatible with F5-TTS, CosyVoice2, E2-TTS, and similar architectures

### How It Works

Instead of retraining models with emotion labels, EmoSteer-TTS:

1. **Extracts** emotion-related activations from reference speech
2. **Builds** steering vectors (emotional - neutral)
3. **Applies** them at inference time with user-defined strength

---

## Dataset

This implementation uses the **ESD (Emotion Speech Dataset)** for extracting
neutral and emotional activation statistics required for building steering vectors.

You can download ESD from Kaggle:  
https://www.kaggle.com/datasets/nguyenthanhlim/emotional-speech-dataset-esd

### Directory Structure

After downloading and organizing the dataset, the expected directory structure is:

```

dataset_esd/
│
├── angry/
│   ├── 0001/
│   │   ├── 0001_000351.wav
│   │   ├── 0001_000872.wav
│   │   └── ...
│   ├── 0002/
│   └── ...
│
├── happy/
├── neutral/
├── sad/
├── surprise/
│
└── transcription/
├── 0001.txt
├── 0002.txt
├── 0003.txt
└── ...

````

- Each emotion folder contains multiple speakers (e.g., `0001`, `0002`, …).  
- Each speaker folder contains audio files of the form:  
  `SPEAKERID_UTTERANCEID.wav`
- The `transcription/` directory contains one text file per speaker, where each line maps an utterance ID to its transcription.

---
## Usage

Below is the basic workflow using **F5-TTS + Activation Steering**.


### Inference without steering

```bash
f5-tts_infer-cli \
  --model F5TTS_v1_Base \
  --ref_audio /path/to/ref_neutral.wav \
  --ref_text "Reference text corresponding to the neutral audio." \
  --gen_text "Input text to be synthesized without emotion steering." \
  --output_dir /path/to/output \
  --output_file baseline.wav \
  --steer_mode off
````


### Extract neutral activations

```bash
f5-tts_infer-cli \
  --model F5TTS_v1_Base \
  --steer_mode neutral \
  --steer_layers all \
  --steer_dataset_dir /path/to/dataset_esd_sorted \
  --steer_neutral_dir neutral \
  --steer_speaker_filter <SPEAKER_RANGE> \
  --steer_max_samples <NUM_SAMPLES> \
  --steer_output_dir /path/to/steering_vectors
```

* `--steer_speaker_filter <SPEAKER_RANGE>`
  e.g. `0001-0010` or `0001,0003,0007` (optional)
* `--steer_max_samples <NUM_SAMPLES>`
  e.g. `50` (0 = use all available files)

### Extract emotional activations

```bash
# Example: emotion = angry
f5-tts_infer-cli \
  --model F5TTS_v1_Base \
  --steer_mode emotion \
  --steer_layers all \
  --steer_dataset_dir /path/to/dataset_esd_sorted \
  --steer_emotion_dir angry \
  --steer_speaker_filter <SPEAKER_RANGE> \
  --steer_max_samples <NUM_SAMPLES> \
  --steer_output_dir /path/to/steering_vectors

# Example: emotion = surprise
f5-tts_infer-cli \
  --model F5TTS_v1_Base \
  --steer_mode emotion \
  --steer_layers all \
  --steer_dataset_dir /path/to/dataset_esd_sorted \
  --steer_emotion_dir surprise \
  --steer_speaker_filter <SPEAKER_RANGE> \
  --steer_max_samples <NUM_SAMPLES> \
  --steer_output_dir /path/to/steering_vectors
```

### Apply steering with emotion2vec

```bash
f5-tts_infer-cli \
  --model F5TTS_v1_Base \
  --ref_audio /path/to/ref_neutral.wav \
  --ref_text "Reference text corresponding to the neutral audio." \
  --gen_text "Input text to be synthesized with emotion steering." \
  --output_dir /path/to/output \
  --output_file output_emotion2vec.wav \
  --steer_mode apply \
  --steer_layers all \
  --steer_alpha <ALPHA> \
  --steer_top_k <TOP_K> \
  --steer_use_emotion2vec \
  --steer_target_emotion <TARGET_EMOTION> \
  --steer_emotion_dir <EMOTION_DIR_NAME> \
  --steer_output_dir /path/to/steering_vectors
```

* `--steer_alpha <ALPHA>`: steering strength, e.g. `5.0` or `10.0`
* `--steer_top_k <TOP_K>`: number of most important tokens to keep, e.g. `200`
* `--steer_target_emotion <TARGET_EMOTION>`: one of `angry`, `disgusted`, `fearful`, `happy`, `neutral`, `sad`, `surprised`
* `--steer_emotion_dir <EMOTION_DIR_NAME>`: name of the emotion subdirectory used in `--steer_mode emotion` (e.g. `angry`, `surprise`)


---

## Supported Backbones

- **F5-TTS** - Flow-matching based TTS model

---

## References

- Original Paper: [EmoSteer-TTS (arXiv:2508.03543)](https://arxiv.org/abs/2508.03543)
- Based on [F5-TTS](https://github.com/SWivid/F5-TTS)

---

## Disclaimer

This is an **unofficial implementation** for research and educational purposes only. Results may differ from the original paper. This project is not affiliated with or endorsed by the original authors.
