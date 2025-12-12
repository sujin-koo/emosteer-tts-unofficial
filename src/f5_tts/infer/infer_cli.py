'''
https://github.com/SWivid/F5-TTS + Activation Steering

#steer_mode = "off"
f5-tts_infer-cli --model F5TTS_v1_Base \
  --ref_audio "./inference_input/0020_000337_neutral.wav" \
  --ref_text "But mom, I'm not certain about." \
  --gen_text "When we studied the Riemann integral the integrability of sums was not quite trivial because we had to use common refinement but it looks easier than the proof here using countable unions." \
  --output_dir ./inference_output \
  --output_file original.wav \
  --steer_mode off

#steer_mode = "neutral"
f5-tts_infer-cli \
  --model F5TTS_v1_Base \
  --steer_mode neutral \
  --steer_layers all \
  --steer_dataset_dir ./dataset_esd_sorted \
  --steer_neutral_dir neutral \
  --steer_speaker_filter 0011-0020 \
  --steer_max_samples 50 \
  --steer_output_dir ./steering_vectors

#steer_mode = "emotion"
f5-tts_infer-cli \
  --model F5TTS_v1_Base \
  --steer_mode emotion \
  --steer_layers all \
  --steer_dataset_dir ./dataset_esd_sorted \
  --steer_emotion_dir angry \
  --steer_speaker_filter 0011-0020 \
  --steer_max_samples 50 \
  --steer_output_dir ./steering_vectors

f5-tts_infer-cli \
  --model F5TTS_v1_Base \
  --steer_mode emotion \
  --steer_layers all \
  --steer_dataset_dir ./dataset_esd_sorted \
  --steer_emotion_dir surprise \
  --steer_speaker_filter 0011-0020 \
  --steer_max_samples 50 \
  --steer_output_dir ./steering_vectors
  
#steer_mode = "apply"
# Without emotion2vec
f5-tts_infer-cli --model F5TTS_v1_Base \
  --ref_audio "./inference_input/0020_000337_neutral.wav" \
  --ref_text "But mom, I'm not certain about." \
  --gen_text "When we studied the Riemann integral the integrability of sums was not quite trivial because we had to use common refinement but it looks easier than the proof here using countable unions." \
  --output_dir ./inference_output \
  --output_file test_output_surprise_without_emotion2vec.wav \
  --steer_mode apply \
  --steer_layers all \
  --steer_alpha 10 \
  --steer_top_k 0 \
  --steer_emotion_dir surprise \
  --steer_output_dir ./steering_vectors
  
# With emotion2vec (slow, paper method)
f5-tts_infer-cli --model F5TTS_v1_Base \
  --ref_audio "./inference_input/0020_000337_neutral.wav" \
  --ref_text "But mom, I'm not certain about." \
  --gen_text "When we studied the Riemann integral the integrability of sums was not quite trivial because we had to use common refinement but it looks easier than the proof here using countable unions." \
  --output_dir ./inference_output \
  --output_file test_output_emotion2vec.wav \
  --steer_mode apply \
  --steer_layers all \
  --steer_alpha 10 \
  --steer_top_k 200 \
  --steer_use_emotion2vec \
  --steer_target_emotion surprised \
  --steer_emotion_dir surprise \
  --steer_output_dir ./steering_vectors


##

0020_001046.wav I am going to back home.
0020_000692_angry.wav  Why should I purchase my own?
0020_000337_neutral.wav  But mom, I'm not certain about.
'''

import argparse
import codecs
import os
import re
from datetime import datetime
from importlib.resources import files
from pathlib import Path

import numpy as np
import soundfile as sf
import tomli
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf
from unidecode import unidecode

import torch 
import torch.nn.functional as F 

from f5_tts.infer.utils_infer import (
    cfg_strength,
    cross_fade_duration,
    device,
    fix_duration,
    infer_process,
    load_model,
    load_vocoder,
    mel_spec_type,
    nfe_step,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    speed,
    sway_sampling_coef,
    target_rms,
)


parser = argparse.ArgumentParser(
    prog="python3 infer-cli.py",
    description="Commandline interface for E2/F5 TTS with Advanced Batch Processing.",
    epilog="Specify options above to override one or more settings from config.",
)
parser.add_argument(
    "-c",
    "--config",
    type=str,
    default=os.path.join(files("f5_tts").joinpath("infer/examples/basic"), "basic.toml"),
    help="The configuration file, default see infer/examples/basic/basic.toml",
)


# Note. Not to provide default value here in order to read default from config file

parser.add_argument(
    "-m",
    "--model",
    type=str,
    help="The model name: F5TTS_v1_Base | F5TTS_Base | E2TTS_Base | etc.",
)
parser.add_argument(
    "-mc",
    "--model_cfg",
    type=str,
    help="The path to F5-TTS model config file .yaml",
)
parser.add_argument(
    "-p",
    "--ckpt_file",
    type=str,
    help="The path to model checkpoint .pt, leave blank to use default",
)
parser.add_argument(
    "-v",
    "--vocab_file",
    type=str,
    help="The path to vocab file .txt, leave blank to use default",
)
parser.add_argument(
    "-r",
    "--ref_audio",
    type=str,
    help="The reference audio file.",
)
parser.add_argument(
    "-s",
    "--ref_text",
    type=str,
    help="The transcript/subtitle for the reference audio",
)
parser.add_argument(
    "-t",
    "--gen_text",
    type=str,
    help="The text to make model synthesize a speech",
)
parser.add_argument(
    "-f",
    "--gen_file",
    type=str,
    help="The file with text to generate, will ignore --gen_text",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="The path to output folder",
)
parser.add_argument(
    "-w",
    "--output_file",
    type=str,
    help="The name of output file",
)
parser.add_argument(
    "--save_chunk",
    action="store_true",
    help="To save each audio chunks during inference",
)
parser.add_argument(
    "--no_legacy_text",
    action="store_false",
    help="Not to use lossy ASCII transliterations of unicode text in saved file names.",
)
parser.add_argument(
    "--remove_silence",
    action="store_true",
    help="To remove long silence found in ouput",
)
parser.add_argument(
    "--load_vocoder_from_local",
    action="store_true",
    help="To load vocoder from local dir, default to ../checkpoints/vocos-mel-24khz",
)
parser.add_argument(
    "--vocoder_name",
    type=str,
    choices=["vocos", "bigvgan"],
    help=f"Used vocoder name: vocos | bigvgan, default {mel_spec_type}",
)
parser.add_argument(
    "--target_rms",
    type=float,
    help=f"Target output speech loudness normalization value, default {target_rms}",
)
parser.add_argument(
    "--cross_fade_duration",
    type=float,
    help=f"Duration of cross-fade between audio segments in seconds, default {cross_fade_duration}",
)
parser.add_argument(
    "--nfe_step",
    type=int,
    help=f"The number of function evaluation (denoising steps), default {nfe_step}",
)
parser.add_argument(
    "--cfg_strength",
    type=float,
    help=f"Classifier-free guidance strength, default {cfg_strength}",
)
parser.add_argument(
    "--sway_sampling_coef",
    type=float,
    help=f"Sway Sampling coefficient, default {sway_sampling_coef}",
)
parser.add_argument(
    "--speed",
    type=float,
    help=f"The speed of the generated audio, default {speed}",
)
parser.add_argument(
    "--fix_duration",
    type=float,
    help=f"Fix the total duration (ref and gen audios) in seconds, default {fix_duration}",
)
parser.add_argument(
    "--device",
    type=str,
    help="Specify the device to run on",
)

parser.add_argument(
    "--steer_mode",
    type=str,
    default="off",
    choices=["off", "neutral", "emotion", "apply"],
    help="Activation steering mode: off | neutral | emotion | apply",
)

parser.add_argument(
    "--steer_alpha",
    type=float,
    default=0.02,
    help="Strength for steering vector when applying residual steering",
)

parser.add_argument(
    "--steer_layers",
    type=str,
    default="all",
    help="Layers to apply steering (e.g., 'all' or '0,1,2,5,7')",
)

parser.add_argument(
    "--steer_top_k",
    type=int,
    default=0,
    help="Top-K tokens to keep from the steering vector (0 = use all tokens)",
)

parser.add_argument(
    "--steer_use_emotion2vec",
    action="store_true",
    help="Use emotion2vec for top-k token selection (paper's Eq. 2-4 method). "
         "WARNING: Very slow (~10-15 min for 250 tokens). Default: use L2 norm (fast)",
)

parser.add_argument(
    "--steer_target_emotion",
    type=str,
    default="happy",
    choices=["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"],
    help="Target emotion for emotion2vec-based token selection",
)

parser.add_argument(
    "--steer_dataset_dir",
    type=str,
    default=None,
    help="Dataset directory for multi-sample steering (e.g., /path/to/dataset_esd_sorted/)",
)

parser.add_argument(
    "--steer_emotion_dir",
    type=str,
    default=None,
    help="Emotion subdirectory name (e.g., 'angry', 'happy') within dataset_dir for emotion mode",
)

parser.add_argument(
    "--steer_neutral_dir",
    type=str,
    default="neutral",
    help="Neutral subdirectory name within dataset_dir for neutral mode (default: 'neutral')",
)

parser.add_argument(
    "--steer_max_samples",
    type=int,
    default=0,
    help="Maximum number of samples to use from dataset (0 = use all)",
)

parser.add_argument(
    "--steer_output_dir",
    type=str,
    default="./steering_vectors",
    help="Directory to save steering vectors (default: ./steering_vectors)",
)

parser.add_argument(
    "--steer_speaker_filter",
    type=str,
    default=None,
    help="Filter specific speakers (e.g., '0011-0020' for speakers 11-20, or '0011,0015,0018' for specific speakers)",
)

args = parser.parse_args()


# config file

config = tomli.load(open(args.config, "rb"))


# command-line interface parameters

model = args.model or config.get("model", "F5TTS_v1_Base")
ckpt_file = args.ckpt_file or config.get("ckpt_file", "")
vocab_file = args.vocab_file or config.get("vocab_file", "")

ref_audio = args.ref_audio or config.get("ref_audio", "infer/examples/basic/basic_ref_en.wav")
ref_text = (
    args.ref_text
    if args.ref_text is not None
    else config.get("ref_text", "Some call me nature, others call me mother nature.")
)
gen_text = args.gen_text or config.get("gen_text", "Here we generate something just for test.")
gen_file = args.gen_file or config.get("gen_file", "")

output_dir = args.output_dir or config.get("output_dir", "tests")
output_file = args.output_file or config.get(
    "output_file", f"infer_cli_{datetime.now().strftime(r'%Y%m%d_%H%M%S')}.wav"
)

save_chunk = args.save_chunk or config.get("save_chunk", False)
use_legacy_text = args.no_legacy_text or config.get("no_legacy_text", False)  # no_legacy_text is a store_false arg
if save_chunk and use_legacy_text:
    print(
        "\nWarning to --save_chunk: lossy ASCII transliterations of unicode text for legacy (.wav) file names, --no_legacy_text to disable.\n"
    )

remove_silence = args.remove_silence or config.get("remove_silence", False)
load_vocoder_from_local = args.load_vocoder_from_local or config.get("load_vocoder_from_local", False)

vocoder_name = args.vocoder_name or config.get("vocoder_name", mel_spec_type)
target_rms = args.target_rms or config.get("target_rms", target_rms)
cross_fade_duration = args.cross_fade_duration or config.get("cross_fade_duration", cross_fade_duration)
nfe_step = args.nfe_step or config.get("nfe_step", nfe_step)
cfg_strength = args.cfg_strength or config.get("cfg_strength", cfg_strength)
sway_sampling_coef = args.sway_sampling_coef or config.get("sway_sampling_coef", sway_sampling_coef)
speed = args.speed or config.get("speed", speed)
fix_duration = args.fix_duration or config.get("fix_duration", fix_duration)
device = args.device or config.get("device", device)


# patches for pip pkg user
if "infer/examples/" in ref_audio:
    ref_audio = str(files("f5_tts").joinpath(f"{ref_audio}"))
if "infer/examples/" in gen_file:
    gen_file = str(files("f5_tts").joinpath(f"{gen_file}"))
if "voices" in config:
    for voice in config["voices"]:
        voice_ref_audio = config["voices"][voice]["ref_audio"]
        if "infer/examples/" in voice_ref_audio:
            config["voices"][voice]["ref_audio"] = str(files("f5_tts").joinpath(f"{voice_ref_audio}"))


# ignore gen_text if gen_file provided

if gen_file:
    gen_text = codecs.open(gen_file, "r", "utf-8").read()


# output path

wave_path = Path(output_dir) / output_file
# spectrogram_path = Path(output_dir) / "infer_cli_out.png"
if save_chunk:
    output_chunk_dir = os.path.join(output_dir, f"{Path(output_file).stem}_chunks")
    if not os.path.exists(output_chunk_dir):
        os.makedirs(output_chunk_dir)


# load vocoder

if vocoder_name == "vocos":
    vocoder_local_path = "../checkpoints/vocos-mel-24khz"
elif vocoder_name == "bigvgan":
    vocoder_local_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"

vocoder = load_vocoder(
    vocoder_name=vocoder_name, is_local=load_vocoder_from_local, local_path=vocoder_local_path, device=device
)


# load TTS model

model_cfg = OmegaConf.load(
    args.model_cfg or config.get("model_cfg", str(files("f5_tts").joinpath(f"configs/{model}.yaml")))
)
model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
model_arc = model_cfg.model.arch

repo_name, ckpt_step, ckpt_type = "F5-TTS", 1250000, "safetensors"

if model != "F5TTS_Base":
    assert vocoder_name == model_cfg.model.mel_spec.mel_spec_type

# override for previous models
if model == "F5TTS_Base":
    if vocoder_name == "vocos":
        ckpt_step = 1200000
    elif vocoder_name == "bigvgan":
        model = "F5TTS_Base_bigvgan"
        ckpt_type = "pt"
elif model == "E2TTS_Base":
    repo_name = "E2-TTS"
    ckpt_step = 1200000

if not ckpt_file:
    ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{model}/model_{ckpt_step}.{ckpt_type}"))

print(f"Using {model}...")
ema_model = load_model(
    model_cls, model_arc, ckpt_file, mel_spec_type=vocoder_name, vocab_file=vocab_file, device=device
)




# Helper functions for activation steering
def parse_layer_selection(selection, total_layers):
    if selection == "all":
        return list(range(total_layers))
    return [int(x.strip()) for x in selection.split(",")]

def load_emotion2vec_model():
    """Load emotion2vec model for SER-based token selection"""
    try:
        from funasr import AutoModel
        print("[SER] Loading emotion2vec model...")
        model = AutoModel(
            model="iic/emotion2vec_plus_large",
            hub="ms"  # "ms" for modelscope (China), "hf" for huggingface (overseas)
        )
        print("[SER] emotion2vec model loaded successfully")
        return model
    except ImportError:
        print("[SER] Warning: funasr not installed. Install with: pip install funasr")
        print("[SER] Falling back to L2 norm-based token selection")
        return None
    except Exception as e:
        print(f"[SER] Warning: Failed to load emotion2vec: {e}")
        print("[SER] Falling back to L2 norm-based token selection")
        return None

def load_transcription(dataset_dir, file_id):
    """
    Load transcription text for a given file ID.

    Args:
        dataset_dir: Base dataset directory
        file_id: File ID like "0001_000351"

    Returns:
        Transcription text or empty string if not found
    """
    # Extract speaker ID (e.g., "0001" from "0001_000351")
    speaker_id = file_id.split("_")[0]

    transcription_file = os.path.join(dataset_dir, "transcription", f"{speaker_id}.txt")

    if not os.path.exists(transcription_file):
        return ""

    try:
        with open(transcription_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2 and parts[0] == file_id:
                    return parts[1]  # Return the transcription text
    except Exception as e:
        print(f"[Warning] Error reading transcription for {file_id}: {e}")

    return ""

def parse_speaker_filter(filter_str):
    """
    Parse speaker filter string.

    Args:
        filter_str: String like "0011-0020" or "0011,0015,0018"

    Returns:
        Set of speaker IDs (e.g., {'0011', '0012', ...})
    """
    if not filter_str:
        return None

    speaker_ids = set()

    # Check if it's a range (e.g., "0011-0020")
    if '-' in filter_str:
        parts = filter_str.split('-')
        if len(parts) == 2:
            try:
                start = int(parts[0])
                end = int(parts[1])
                for i in range(start, end + 1):
                    speaker_ids.add(f"{i:04d}")
            except ValueError:
                print(f"[Warning] Invalid range format: {filter_str}")
    else:
        # Comma-separated list (e.g., "0011,0015,0018")
        for speaker in filter_str.split(','):
            speaker = speaker.strip()
            if speaker:
                speaker_ids.add(speaker)

    return speaker_ids if speaker_ids else None

def extract_activations_from_dataset(
    dataset_dir,
    emotion_subdir,
    ema_model,
    vocoder,
    selected_layers,
    max_samples=0,
    mel_spec_type="vocos",
    device="cuda",
    speaker_filter=None
):
    """
    Extract activations from multiple audio samples in a dataset directory.

    Args:
        dataset_dir: Base dataset directory (e.g., /path/to/dataset_esd_sorted/)
        emotion_subdir: Emotion subdirectory (e.g., 'angry', 'neutral')
        ema_model: TTS model
        vocoder: Vocoder
        selected_layers: List of layer indices to extract
        max_samples: Maximum number of samples to process (0 = all)
        mel_spec_type: Mel spectrogram type
        device: Device to use
        speaker_filter: Set of speaker IDs to include (None = all speakers)

    Returns:
        List of layer activations, where each element is a list of [T, D] tensors
    """
    import glob
    from f5_tts.infer.utils_infer import preprocess_ref_audio_text, infer_process

    emotion_path = os.path.join(dataset_dir, emotion_subdir)

    if not os.path.exists(emotion_path):
        print(f"[Error] Directory not found: {emotion_path}")
        return None

    # Get all .wav files recursively
    audio_files = sorted(glob.glob(os.path.join(emotion_path, "**", "*.wav"), recursive=True))

    if len(audio_files) == 0:
        print(f"[Error] No .wav files found in {emotion_path}")
        return None

    # Filter by speaker if specified
    if speaker_filter:
        print(f"[Dataset] Filtering speakers: {sorted(speaker_filter)}")
        filtered_files = []
        for audio_file in audio_files:
            filename = os.path.basename(audio_file)
            speaker_id = filename.split('_')[0]  # Extract speaker ID from filename
            if speaker_id in speaker_filter:
                filtered_files.append(audio_file)

        audio_files = filtered_files
        print(f"[Dataset] After filtering: {len(audio_files)} files")

    if len(audio_files) == 0:
        print(f"[Error] No audio files found after speaker filtering")
        return None

    if max_samples > 0:
        audio_files = audio_files[:max_samples]

    print(f"\n[Dataset] Found {len(audio_files)} audio files in {emotion_path}")
    print(f"[Dataset] Processing {len(audio_files)} samples for activation extraction...")

    # Enable residual saving for selected layers
    for i, block in enumerate(ema_model.transformer.transformer_blocks):
        if i in selected_layers:
            block.save_residual = True

    # Collect activations from all samples
    # layer_activations[layer_idx] = [activation_sample1, activation_sample2, ...]
    layer_activations = [[] for _ in selected_layers]

    for idx, audio_file in enumerate(audio_files):
        # Extract file ID from filename (e.g., "0001_000351" from "0001_000351.wav")
        filename = os.path.basename(audio_file)
        file_id = os.path.splitext(filename)[0]

        print(f"[Dataset] Processing {idx+1}/{len(audio_files)}: {filename}")

        # Load transcription for this file
        ref_text = load_transcription(dataset_dir, file_id)

        if ref_text:
            print(f"[Dataset]   Transcription: {ref_text[:50]}{'...' if len(ref_text) > 50 else ''}")
        else:
            print(f"[Dataset]   Warning: No transcription found for {file_id}, using empty text")

        ref_audio_path = audio_file
        gen_text = ref_text if ref_text else "This is a sample text for activation extraction."

        try:
            # Preprocess audio
            ref_audio_processed, ref_text_processed = preprocess_ref_audio_text(
                ref_audio_path, ref_text
            )

            # Run inference to extract activations
            _, _, _ = infer_process(
                ref_audio_processed,
                ref_text_processed,
                gen_text,
                ema_model,
                vocoder,
                mel_spec_type=mel_spec_type,
                device=device,
            )

            # Collect activations from each layer
            for layer_list_idx, layer_idx in enumerate(selected_layers):
                block = ema_model.transformer.transformer_blocks[layer_idx]
                if hasattr(block, "first_residual") and block.first_residual is not None:
                    # first_residual: [B, T, D] -> extract [T, D] for single sample
                    activation = block.first_residual[0].cpu()  # [T, D]
                    layer_activations[layer_list_idx].append(activation)

        except Exception as e:
            print(f"[Dataset] Error processing {audio_file}: {e}")
            continue

    # Disable residual saving
    for block in ema_model.transformer.transformer_blocks:
        block.save_residual = False

    print(f"[Dataset] Extraction complete. Collected activations from {len(audio_files)} samples")

    return layer_activations

def evaluate_tokens_with_ser(
    ser_model,
    tts_model,
    vocoder,
    base_activation,
    steering_vector_unnorm,
    target_emotion="happy",
    ref_audio=None,
    ref_text=None,
    gen_text=None,
    use_full_synthesis=False,
    selected_layers=None
):
    """
    Evaluate emotion probability for each token using SER model (Eq. 3-4).

    This follows the paper's method (Section 3.3):
    1. For each token i, repeat token i T times to form ûl (Eq. 2)
    2. Modify activation: x̂l_a ← fr(xl_a + ûl) with L2 renormalization
    3. Synthesize audio Âi with modified activation (if use_full_synthesis=True)
    4. Use emotion2vec to predict Pemotion(Âi) (Eq. 3)
    5. Return probability set P = {Pemotion(Âi) | i ∈ [1,T]}

    Args:
        ser_model: emotion2vec SER model
        tts_model: TTS model for synthesis
        vocoder: Vocoder to convert mel-spec to audio
        base_activation: Base activation [T, D]
        steering_vector_unnorm: Unnormalized steering vector [T, D]
        target_emotion: Target emotion to evaluate
        ref_audio: Reference audio path
        ref_text: Reference text
        gen_text: Generation text
        use_full_synthesis: If True, synthesize T audio samples (expensive but accurate)
                           If False, use L2 norm approximation (fast but less accurate)

    Returns:
        emotion_probs: Tensor of emotion probabilities [T]
    """
    import os

    if ser_model is None:
        # Fallback to L2 norm
        print("[SER] No SER model provided, using L2 norm as importance")
        return steering_vector_unnorm.norm(p=2, dim=-1)

    if not use_full_synthesis:
        # Fast path: use L2 norm approximation
        print(f"[SER] Using L2 norm approximation (fast mode)")
        return steering_vector_unnorm.norm(p=2, dim=-1)

    # Full synthesis path (expensive but paper-accurate)
    print(f"[SER] Using FULL emotion2vec synthesis (paper method)")
    print(f"[SER] This implements paper's Eq. 2-4")

    # Import the full implementation
    import sys
    # Go up 4 levels: infer_cli.py -> infer -> f5_tts -> src -> F5-TTS (project root)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.insert(0, project_root)
    from evaluate_tokens_ser_full import evaluate_tokens_with_ser_full

    # Use a random neutral sample for evaluation (as per paper Section 4.1)
    # Paper uses: "10 random neutral ESD samples"
    # For simplicity, we use one sample here
    ref_audio_path = "./inference_input/0020_000337_neutral.wav"
    ref_text_sample = "But mom, I'm not certain about."
    gen_text_sample = "This is a test sentence for evaluation."

    # Use provided selected_layers, or default to every 5th layer
    if selected_layers is None:
        selected_layers = [i for i in range(0, len(tts_model.transformer.transformer_blocks), 5)]

    emotion_probs = evaluate_tokens_with_ser_full(
        ser_model=ser_model,
        tts_model=tts_model,
        vocoder=vocoder,
        steering_vector_unnorm=steering_vector_unnorm,
        target_emotion=target_emotion,
        ref_audio_path=ref_audio_path,
        ref_text=ref_text_sample,
        gen_text=gen_text_sample,
        selected_layers=selected_layers,
        nfe_step=32,
        cfg_strength=2.0,
        sway_sampling_coef=-1.0,
        mel_spec_type="vocos",
        device=tts_model.device
    )

    return emotion_probs



# inference process


def main():
    main_voice = {"ref_audio": ref_audio, "ref_text": ref_text}
    if "voices" not in config:
        voices = {"main": main_voice}
    else:
        voices = config["voices"]
        voices["main"] = main_voice
    for voice in voices:
        print("Voice:", voice)
        print("ref_audio ", voices[voice]["ref_audio"])
        voices[voice]["ref_audio"], voices[voice]["ref_text"] = preprocess_ref_audio_text(
            voices[voice]["ref_audio"], voices[voice]["ref_text"]
        )
        print("ref_audio_", voices[voice]["ref_audio"], "\n\n")

    generated_audio_segments = []


    # === Activation Steering Setup ===
    # Initialize all transformer blocks with default values
    for block in ema_model.transformer.transformer_blocks:
        block.save_residual = False
        block.steer_vector = None
        block.alpha = 0.0

    steer_mode = args.steer_mode
    steer_alpha = args.steer_alpha
    steer_top_k = args.steer_top_k
    steer_use_emotion2vec = args.steer_use_emotion2vec
    steer_target_emotion = args.steer_target_emotion
    selected_layers = parse_layer_selection(
        args.steer_layers,
        len(ema_model.transformer.transformer_blocks)
    )

    # Load SER model if requested
    ser_model = None
    if steer_use_emotion2vec and steer_mode == "apply":
        ser_model = load_emotion2vec_model()
        if ser_model is None:
            steer_use_emotion2vec = False

    print(f"\n==============================")
    print(f" Steering Mode     : {steer_mode}")
    print(f" Steering Layers   : {selected_layers}")
    print(f" Steering Alpha    : {steer_alpha}")
    print(f" Steering top-k    : {steer_top_k}")
    print(f" Use emotion2vec   : {steer_use_emotion2vec}")
    if steer_use_emotion2vec:
        print(f" Target Emotion    : {steer_target_emotion}")
    print(f"==============================\n")
    
    # 1) Neutral / Emotion Capturing — forward 전 block 설정
    if steer_mode in ["neutral", "emotion"]:
        # Check if using dataset mode
        dataset_dir = args.steer_dataset_dir
        emotion_dir = args.steer_emotion_dir
        neutral_dir = args.steer_neutral_dir
        max_samples = args.steer_max_samples

        if dataset_dir is not None:
            # === DATASET MODE: Extract from multiple samples ===
            if steer_mode == "neutral":
                subdir = neutral_dir
            else:  # emotion mode
                if emotion_dir is None:
                    print("[Error] --steer_emotion_dir must be specified for emotion mode")
                    return
                subdir = emotion_dir

            print(f"\n[Dataset Mode] Extracting activations from: {os.path.join(dataset_dir, subdir)}")

            # Parse speaker filter if provided
            speaker_filter = parse_speaker_filter(args.steer_speaker_filter)

            # Extract activations from dataset
            layer_activations = extract_activations_from_dataset(
                dataset_dir=dataset_dir,
                emotion_subdir=subdir,
                ema_model=ema_model,
                vocoder=vocoder,
                selected_layers=selected_layers,
                max_samples=max_samples,
                mel_spec_type=vocoder_name,
                device=device,
                speaker_filter=speaker_filter
            )

            if layer_activations is None:
                print("[Error] Failed to extract activations from dataset")
                return

            # === Compute average length and interpolate all samples ===
            print(f"\n[Dataset] Computing average length and interpolating...")

            processed_layers = []
            for layer_idx, activations_list in enumerate(layer_activations):
                # activations_list: list of [T_i, D] tensors
                if len(activations_list) == 0:
                    print(f"[Warning] No activations collected for layer {selected_layers[layer_idx]}")
                    continue

                # Step 1: Compute average length across all samples
                lengths = [act.shape[0] for act in activations_list]
                avg_len = int(sum(lengths) / len(lengths))

                print(f"[Dataset] Layer {selected_layers[layer_idx]}: {len(activations_list)} samples, "
                      f"lengths {min(lengths)}-{max(lengths)}, avg={avg_len}")

                # Step 2: Interpolate all samples to average length
                def match_length_single(x, target_len):
                    """x: [T, D] -> [target_len, D]"""
                    x = x.transpose(0, 1).unsqueeze(0)  # [1, D, T]
                    x = F.interpolate(x, size=target_len, mode="nearest")
                    return x.squeeze(0).transpose(0, 1)  # [target_len, D]

                interpolated = [match_length_single(act, avg_len) for act in activations_list]

                # Step 3: Stack and compute mean (Eq. 1)
                stacked = torch.stack(interpolated, dim=0)  # [N, avg_len, D]
                processed_layers.append(stacked)

            # Save processed activations to organized directory
            steer_output_dir = args.steer_output_dir
            os.makedirs(steer_output_dir, exist_ok=True)

            # Create filename with emotion name for better organization
            if steer_mode == "neutral":
                filename = "neutral_residual.pt"
            else:
                filename = f"{emotion_dir}_residual.pt"

            save_path = os.path.join(steer_output_dir, filename)
            torch.save(processed_layers, save_path)
            print(f"\n[Dataset] Saved multi-sample activations → {save_path}")
            print(f"[Dataset] Format: List of [N_samples, avg_len, D] per layer")

            # Skip the normal inference process in dataset mode
            return

        else:
            # === SINGLE SAMPLE MODE: Original behavior ===
            print(f"[Steering] Capturing first residuals for layers: {selected_layers}")
            for i, block in enumerate(ema_model.transformer.transformer_blocks):
                if i in selected_layers:
                    block.save_residual = True

    # 2) APPLY mode — 미리 저장된 residual 불러와 steering vector 준비
    elif steer_mode == "apply":

        print("[Steering] Loading saved residuals...")

        steer_output_dir = args.steer_output_dir
        neutral_path = os.path.join(steer_output_dir, "neutral_residual.pt")

        # Determine emotion file path
        emotion_dir = args.steer_emotion_dir
        if emotion_dir:
            emotion_path = os.path.join(steer_output_dir, f"{emotion_dir}_residual.pt")
        else:
            # Fallback to default emotion_residual.pt
            emotion_path = os.path.join(steer_output_dir, "emotion_residual.pt")

        # Check if files exist
        if not os.path.exists(neutral_path):
            print(f"[Error] Neutral residual file not found: {neutral_path}")
            print(f"[Error] Please run with --steer_mode neutral first")
            return

        if not os.path.exists(emotion_path):
            print(f"[Error] Emotion residual file not found: {emotion_path}")
            print(f"[Error] Please run with --steer_mode emotion --steer_emotion_dir {emotion_dir} first")
            return

        print(f"[Steering] Loading neutral from: {neutral_path}")
        print(f"[Steering] Loading emotion from: {emotion_path}")

        neutral_res = torch.load(neutral_path)  # List of [B, T, D] per layer
        emotion_res = torch.load(emotion_path)  # List of [B, T, D] per layer

        def match_length(x, target_len):
            """
            x: [B, T, D]
            target_len: int
            return: [B, target_len, D]
            """
            B, T, D = x.shape
            x = x.transpose(1, 2)         # [B, D, T]
            x = F.interpolate(
                x,
                size=target_len,
                mode="nearest"
            )
            x = x.transpose(1, 2)         # [B, target_len, D]
            return x

        # -------------------------
        # Step 1: Compute unnormalized steering vectors for all layers
        # -------------------------
        unnormalized_vectors = []
        for n, e in zip(neutral_res, emotion_res):
            # n: [B_n, T_n, D], e: [B_e, T_e, D]

            # Compute average sequence length
            avg_len = (n.shape[1] + e.shape[1]) // 2

            # Interpolate to average length
            n_interp = match_length(n, avg_len)  # [B_n, avg_len, D]
            e_interp = match_length(e, avg_len)  # [B_e, avg_len, D]

            # Compute mean across samples (Eq. 1)
            n_mean = n_interp.mean(dim=0)  # [avg_len, D]
            e_mean = e_interp.mean(dim=0)  # [avg_len, D]

            # Compute activation difference
            ul = e_mean - n_mean  # [avg_len, D]

            unnormalized_vectors.append({
                'ul': ul,
                'avg_len': avg_len,
                'n_mean': n_mean,
                'e_mean': e_mean
            })

        # -------------------------
        # Step 2: Compute token importance ONCE (O(T) complexity)
        # Apply to all layers simultaneously when evaluating
        # -------------------------
        top_k_indices_global = None
        top_k_values_global = None

        if steer_top_k > 0:
            if steer_use_emotion2vec and ser_model is not None:
                # Use emotion2vec for token evaluation (paper's Eq. 2-4)
                # Select representative layer (use middle layer)
                repr_idx = len(unnormalized_vectors) // 2
                repr_data = unnormalized_vectors[repr_idx]

                print(f"[SER] ========================================")
                print(f"[SER] Using emotion2vec for token selection (target: {steer_target_emotion})")
                print(f"[SER] Evaluating with representative layer {selected_layers[repr_idx]}")
                print(f"[SER] This will apply steering to ALL selected layers: {selected_layers}")
                print(f"[SER] Complexity: O(T) = O({repr_data['avg_len']}) TTS + SER evaluations")
                print(f"[SER] ========================================")

                token_importance = evaluate_tokens_with_ser(
                    ser_model=ser_model,
                    tts_model=ema_model,
                    vocoder=vocoder,
                    base_activation=repr_data['n_mean'],
                    steering_vector_unnorm=repr_data['ul'],
                    target_emotion=steer_target_emotion,
                    use_full_synthesis=True,  # Always use full synthesis with emotion2vec
                    selected_layers=selected_layers  # Pass actual layers being used
                )

                # Select top-k tokens (apply this result to all layers)
                if steer_top_k < repr_data['avg_len']:
                    top_k_values_global, top_k_indices_global = torch.topk(token_importance, steer_top_k)
                    # Move to device for later use
                    top_k_values_global = top_k_values_global.to(device)
                    top_k_indices_global = top_k_indices_global.to(device)
                    print(f"[SER] Selected top-{steer_top_k} tokens: {top_k_indices_global.tolist()}")
                    print(f"[SER] Token importance range: [{token_importance.min():.4f}, {token_importance.max():.4f}]")

        # -------------------------
        # Step 3: Apply computed top-k indices to all layers
        # -------------------------
        steering_vectors = []
        weight_vectors = []

        for layer_idx, vec_data in enumerate(unnormalized_vectors):
            ul = vec_data['ul'].to(device)  # Move to device
            avg_len = vec_data['avg_len']

            # L2 normalization (global: preserve relative token importance)
            ul_flat = ul.view(-1)
            ul_norm = ul / (ul_flat.norm(p=2) + 1e-8)  # [avg_len, D]

            # Top-k token filtering (requires emotion2vec)
            if steer_top_k > 0:
                if top_k_indices_global is not None:
                    # Use globally computed top-k indices (emotion2vec method)
                    top_k_indices = top_k_indices_global
                    top_k_values = top_k_values_global
                else:
                    raise ValueError(
                        "--steer_top_k requires --steer_use_emotion2vec. "
                        "Top-k token selection is only available with emotion2vec-based evaluation (paper method)."
                    )

                # Select top-k tokens
                if steer_top_k < avg_len:
                    # Create mask: zero out non-top-k tokens (Eq. 5)
                    mask = torch.zeros_like(ul_norm)  # [avg_len, D]
                    mask[top_k_indices] = 1.0

                    # Apply mask to get steering vector
                    sl = ul_norm * mask  # [avg_len, D]

                    # === Compute weight vector (Eq. 6) ===
                    # Softmax over top-k token importance
                    top_k_weights = F.softmax(top_k_values, dim=0)  # [k]

                    # Create full weight vector (on the same device as top_k)
                    wl = torch.zeros(avg_len, device=device, dtype=top_k_weights.dtype)  # [avg_len]
                    wl[top_k_indices] = top_k_weights
                else:
                    sl = ul_norm
                    wl = torch.ones(avg_len, device=device, dtype=ul_norm.dtype) / avg_len
            else:
                # Use all tokens
                sl = ul_norm
                wl = torch.ones(avg_len, device=device, dtype=ul_norm.dtype) / avg_len

            # === Compute weighted steering vector (Eq. 7) ===
            # Apply adaptive weight to each token based on importance
            # wl: [avg_len] contains softmax weights for top-k tokens (sum=1)
            # sl: [avg_len, D] is the masked, normalized steering vector
            #
            # IMPORTANT: Since wl is softmax-normalized (sum=1), multiplying it
            # would reduce the magnitude by ~1/k. To preserve magnitude while
            # still applying adaptive weights, we scale by k (number of top-k tokens)
            num_nonzero = (wl > 0).sum().item()
            wl_scaled = wl * num_nonzero  # Scale to preserve magnitude
            sl_weighted = sl * wl_scaled.unsqueeze(-1)  # [avg_len, D]

            # Get model dtype and convert steering vector
            model_dtype = next(ema_model.parameters()).dtype
            steering_vectors.append(sl_weighted.unsqueeze(0).to(device, dtype=model_dtype))  # [1, avg_len, D]
            weight_vectors.append(wl.to(device))

        # --------------------------
        # Set steering vector per block
        # --------------------------
        print(f"[Steering] Applying vectors to layers: {selected_layers}")
        print(f"[Steering] Steering vectors computed with L2 normalization and top-k={steer_top_k}")

        # Apply steering vectors to selected layers
        for vec_idx, layer_idx in enumerate(selected_layers):
            block = ema_model.transformer.transformer_blocks[layer_idx]
            block.steer_vector = steering_vectors[vec_idx]
            block.alpha = steer_alpha

            # Debug: Print steering vector statistics
            vec_norm = steering_vectors[vec_idx].norm(p=2).item()
            print(f"[Steering] Layer {layer_idx}: vector norm={vec_norm:.4f}, alpha={steer_alpha}")


    reg1 = r"(?=\[\w+\])"
    chunks = re.split(reg1, gen_text)
    reg2 = r"\[(\w+)\]"
    for text in chunks:
        if not text.strip():
            continue
        match = re.match(reg2, text)
        if match:
            voice = match[1]
        else:
            print("No voice tag found, using main.")
            voice = "main"
        if voice not in voices:
            print(f"Voice {voice} not found, using main.")
            voice = "main"
        text = re.sub(reg2, "", text)
        ref_audio_ = voices[voice]["ref_audio"]
        ref_text_ = voices[voice]["ref_text"]
        local_speed = voices[voice].get("speed", speed)
        gen_text_ = text.strip()
        print(f"Voice: {voice}")
   
        audio_segment, final_sample_rate, spectrogram = infer_process(
            ref_audio_,
            ref_text_,
            gen_text_,
            ema_model,
            vocoder,
            mel_spec_type=vocoder_name,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=local_speed,
            fix_duration=fix_duration,
            device=device,
        )


        # ----------------------------
        # inference 후 residual 저장
        # ----------------------------
        if steer_mode in ["neutral", "emotion"]:
            collected = []
            for i, block in enumerate(ema_model.transformer.transformer_blocks):
                if block.save_residual and hasattr(block, "first_residual"):
                    collected.append(block.first_residual.cpu())

            # Save to organized directory (single sample mode)
            steer_output_dir = args.steer_output_dir
            os.makedirs(steer_output_dir, exist_ok=True)

            if steer_mode == "neutral":
                filename = "neutral_residual.pt"
            else:
                # Use emotion_dir if specified, otherwise default
                emotion_dir = args.steer_emotion_dir
                if emotion_dir:
                    filename = f"{emotion_dir}_residual.pt"
                else:
                    filename = "emotion_residual.pt"

            save_path = os.path.join(steer_output_dir, filename)
            torch.save(collected, save_path)
            print(f"[Steering] Saved residual → {save_path}")


       
        generated_audio_segments.append(audio_segment)

        if save_chunk:
            if len(gen_text_) > 200:
                gen_text_ = gen_text_[:200] + " ... "
            if use_legacy_text:
                gen_text_ = unidecode(gen_text_)
            sf.write(
                os.path.join(output_chunk_dir, f"{len(generated_audio_segments) - 1}_{gen_text_}.wav"),
                audio_segment,
                final_sample_rate,
            )

    if generated_audio_segments:
        final_wave = np.concatenate(generated_audio_segments)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(wave_path, "wb") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            # Remove silence
            if remove_silence:
                remove_silence_for_generated_wav(f.name)
            print(f.name)


if __name__ == "__main__":
    main()
