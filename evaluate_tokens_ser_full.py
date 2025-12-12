"""
Full emotion2vec-based token evaluation (implements paper's Eq. 2-4)

This is a standalone implementation of the paper's method.
Use this when you want paper-accurate token selection.
"""

import torch
import torch.nn.functional as F
import tempfile
import os
import torchaudio


def evaluate_tokens_with_ser_full(
    ser_model,
    tts_model,
    vocoder,
    steering_vector_unnorm,
    target_emotion="happy",
    ref_audio_path="./inference_input/0020_000337_neutral.wav",
    ref_text="But mom, I'm not certain about.",
    gen_text="This is a test sentence.",
    selected_layers=[1, 6, 11, 16, 21],
    nfe_step=32,
    cfg_strength=2.0,
    sway_sampling_coef=-1.0,
    mel_spec_type="vocos",
    device="cuda"
):
    """
    Complete implementation of paper Section 3.3's Eq. 2-4 for token evaluation.

    For each token i:
    1. Repeat token i T times (Eq. 2)
    2. TTS synthesis
    3. emotion2vec evaluation (Eq. 3)
    4. Collect probabilities and select top-k (Eq. 4)

    Args:
        ser_model: emotion2vec model
        tts_model: F5-TTS model
        vocoder: Vocos or BigVGAN
        steering_vector_unnorm: Unnormalized steering vector [T, D]
        target_emotion: Target emotion to evaluate
        ref_audio_path: Path to reference audio (random neutral sample)
        ref_text: Reference text
        gen_text: Generation text (random sentence)
        selected_layers: Which layers to apply steering
        ...

    Returns:
        emotion_probs: Tensor of emotion probabilities [T]
    """
    from f5_tts.infer.utils_infer import preprocess_ref_audio_text

    # Move steering vector to the correct device and dtype
    steering_vector_unnorm = steering_vector_unnorm.to(device)
    T = steering_vector_unnorm.shape[0]

    # Emotion mapping (9-class)
    emotion_map = {
        "angry": 0, "disgusted": 1, "fearful": 2, "happy": 3,
        "neutral": 4, "other": 5, "sad": 6, "surprised": 7, "unknown": 8
    }
    target_idx = emotion_map.get(target_emotion, 3)

    print(f"[SER-FULL] ========================================")
    print(f"[SER-FULL] Evaluating {T} tokens with emotion2vec")
    print(f"[SER-FULL] Target emotion: {target_emotion} (idx={target_idx})")
    print(f"[SER-FULL] WARNING: This will take ~{T * 3} seconds")
    print(f"[SER-FULL] ========================================")

    # Preprocess reference audio (returns file path, not tensor)
    ref_audio_path_processed, ref_text_processed = preprocess_ref_audio_text(
        ref_audio_path, ref_text
    )

    # Load audio tensor and move to device
    ref_audio, sr = torchaudio.load(ref_audio_path_processed)
    ref_audio = ref_audio.to(device)

    # Get reference audio length
    if len(ref_audio.shape) == 2:
        ref_audio_len = ref_audio.shape[-1] // 256  # hop_length=256
    else:
        ref_audio_len = ref_audio.shape[-2] // 256

    # Prepare text
    from f5_tts.model.utils import get_tokenizer
    vocab_char_map, vocab_size = get_tokenizer("Emilia_ZH_EN", "pinyin")
    text_list = [ref_text_processed + " " + gen_text]

    # Calculate duration
    ref_text_len = len(ref_text_processed.encode('utf-8'))
    gen_text_len = len(gen_text.encode('utf-8'))
    duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len)

    emotion_probs = []
    temp_dir = tempfile.mkdtemp(prefix="ser_eval_")

    try:
        for token_idx in range(T):
            try:
                # === Create single-token steering ===
                single_token_steering = torch.zeros_like(steering_vector_unnorm)
                single_token_steering[token_idx] = steering_vector_unnorm[token_idx]

                # Normalize
                single_token_steering = single_token_steering / (single_token_steering.norm(p=2) + 1e-8)

                # === Set steering vectors on blocks (same method as infer_cli.py) ===
                # Prepare steering vector: [1, T, D]
                # single_token_steering is already [T, D], just add batch dimension
                steering_to_apply = single_token_steering.unsqueeze(0)  # [1, T, D]

                # Get model dtype (usually float16/half)
                model_dtype = next(tts_model.parameters()).dtype
                steering_to_apply = steering_to_apply.to(dtype=model_dtype)

                # Apply to selected layers
                for layer_idx in selected_layers:
                    block = tts_model.transformer.transformer_blocks[layer_idx]
                    block.steer_vector = steering_to_apply
                    block.alpha = 1.0  # Use full strength for evaluation

                # === Synthesize audio (Step 3 in paper) ===
                with torch.inference_mode():
                    generated, _ = tts_model.sample(
                        cond=ref_audio,
                        text=text_list,
                        duration=duration,
                        steps=nfe_step,
                        cfg_strength=cfg_strength,
                        sway_sampling_coef=sway_sampling_coef,
                    )

                    # Convert to waveform
                    generated = generated.to(torch.float32)
                    generated = generated[:, ref_audio_len:, :]
                    generated = generated.permute(0, 2, 1)

                    if mel_spec_type == "vocos":
                        generated_wave = vocoder.decode(generated)
                    elif mel_spec_type == "bigvgan":
                        generated_wave = vocoder(generated)

                    # Save to temp file
                    temp_path = os.path.join(temp_dir, f"token_{token_idx:04d}.wav")
                    # Ensure 2D tensor [channels, samples]
                    wave_to_save = generated_wave.squeeze(0).cpu()
                    if wave_to_save.ndim == 1:
                        wave_to_save = wave_to_save.unsqueeze(0)  # Add channel dimension
                    torchaudio.save(
                        temp_path,
                        wave_to_save,
                        24000,
                    )

                # Clear steering vectors from blocks
                for layer_idx in selected_layers:
                    block = tts_model.transformer.transformer_blocks[layer_idx]
                    block.steer_vector = None
                    block.alpha = 0.0

                # === Evaluate with emotion2vec (Eq. 3 in paper) ===
                result = ser_model.generate(
                    temp_path,
                    granularity="utterance",
                    extract_embedding=False
                )

                # Extract probability for target emotion
                if result and len(result) > 0:
                    scores = result[0].get('scores', [0] * 9)
                    emotion_prob = scores[target_idx] if len(scores) > target_idx else 0.0
                else:
                    emotion_prob = 0.0

                emotion_probs.append(emotion_prob)

                # Progress
                if (token_idx + 1) % 10 == 0 or token_idx == T - 1:
                    print(f"[SER-FULL] {token_idx + 1}/{T} | prob={emotion_prob:.4f}")

            except Exception as e:
                print(f"[SER-FULL] Error at token {token_idx}: {e}")
                import traceback
                traceback.print_exc()
                emotion_probs.append(0.0)

                # Clean up steering vectors
                for layer_idx in selected_layers:
                    try:
                        block = tts_model.transformer.transformer_blocks[layer_idx]
                        block.steer_vector = None
                        block.alpha = 0.0
                    except:
                        pass

    finally:
        # Clean up
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    print(f"[SER-FULL] ========================================")
    print(f"[SER-FULL] Complete! Prob range: [{min(emotion_probs):.4f}, {max(emotion_probs):.4f}]")
    print(f"[SER-FULL] ========================================")

    return torch.tensor(emotion_probs, dtype=torch.float32, device=device)
