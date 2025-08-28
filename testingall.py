import os
import sys
import numpy as np
import librosa
import soundfile as sf
import torch
from pathlib import Path

# =====================
# CONFIG
# =====================
REF_WAV_PATH = r"D:/realtimevoice/obama.wav"  # change to your test audio
TEXT = "This is a voice cloning test using pre-trained models only."
OUTPUT_WAV_PATH = "rtvc_test_output.wav"

# Model paths as Path objects
ENCODER_MODEL_PATH = Path(r"D:/realtimevoice/encoder/saved_models/encoder.pt")
SYNTH_MODEL_PATH = Path(r"D:/realtimevoice/synthesizer/saved_models/synthesizer.pt")
VOCODER_MODEL_PATH = Path(r"D:/realtimevoice/vocoder/saved_models/vocoder.pt")

# =====================
# IMPORT MODEL APIs
# =====================
sys.path.append(os.path.abspath("D:/realtimevoice/encoder"))
sys.path.append(os.path.abspath("D:/realtimevoice/synthesizer"))
sys.path.append(os.path.abspath("D:/realtimevoice/vocoder"))

from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder

# =====================
# LOAD MODELS
# =====================
print("Loading models...")
encoder.load_model(ENCODER_MODEL_PATH)
synthesizer = Synthesizer(SYNTH_MODEL_PATH)
vocoder.load_model(VOCODER_MODEL_PATH)

# =====================
# ENCODER: Reference → Embedding
# =====================
print("Loading and preprocessing reference audio...")
original_wav, _ = librosa.load(REF_WAV_PATH, sr=16000)
preprocessed_wav = encoder.preprocess_wav(original_wav)
embed = encoder.embed_utterance(preprocessed_wav)
print("✅ Encoder embedding shape:", embed.shape)

# =====================
# SYNTHESIZER: Text → Mel
# =====================
print("Synthesizing mel spectrogram...")
specs = synthesizer.synthesize_spectrograms([TEXT], [embed])
mel = specs[0]
print("✅ Synthesizer mel shape:", mel.shape)

# =====================
# VOCODER: Mel → Waveform
# =====================
print("Generating waveform...")
generated_wav = vocoder.infer_waveform(mel)
generated_wav = np.pad(generated_wav, (0, 4000), mode="constant")

# Save output
sf.write(OUTPUT_WAV_PATH, generated_wav.astype(np.float32), 16000)
print(f"✅ Output saved to: {OUTPUT_WAV_PATH}")

# =====================
# SIMILARITY CHECK
# =====================
print("Calculating speaker similarity...")
preprocessed_gen = encoder.preprocess_wav(generated_wav)
embed_gen = encoder.embed_utterance(preprocessed_gen)

similarity = torch.cosine_similarity(
    torch.tensor(embed).unsqueeze(0),
    torch.tensor(embed_gen).unsqueeze(0)
).item()

print(f"🔹 Speaker similarity: {similarity * 100:.2f}%")
