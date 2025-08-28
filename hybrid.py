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
REF_WAV_PATH = r"D:/realtimevoice/obama.wav"  # Reference voice
TEXT = "This is a reverse hybrid test using the RTVC encoder and synthesizer with my vocoder."
OUTPUT_WAV_PATH = "reverse_hybrid_output.wav"

# RTVC model paths
RTVC_ENCODER_PATH = Path(r"D:/realtimevoice/encoder/saved_models/encoder.pt")
RTVC_SYNTH_PATH = Path(r"D:/realtimevoice/synthesizer/saved_models/synthesizer.pt")

# Your vocoder checkpoint
MY_VOCODER_PATH = r"D:/gui_app/gui_app/vocoder_epoch50.pth"

# =====================
# IMPORTS
# =====================
# Add RTVC encoder/synth paths
sys.path.append(os.path.abspath("D:/realtimevoice/encoder"))
sys.path.append(os.path.abspath("D:/realtimevoice/synthesizer"))

from encoder import inference as encoder
from synthesizer.inference import Synthesizer

# Add your vocoder path
sys.path.append(os.path.abspath("D:/realtimevoice/utils"))
from utils import models as my_models

# =====================
# LOAD MODELS
# =====================
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading RTVC encoder...")
encoder.load_model(RTVC_ENCODER_PATH)

print("Loading RTVC synthesizer...")
synthesizer = Synthesizer(RTVC_SYNTH_PATH)

print("Loading my trained vocoder...")
my_vocoder = my_models.HiFiGANGenerator(n_mels=80).to(device)
my_vocoder.load_state_dict(torch.load(MY_VOCODER_PATH, map_location=device))
my_vocoder.eval()

# =====================
# ENCODER: Reference → Embedding
# =====================
print("Generating embedding from RTVC encoder...")
original_wav, _ = librosa.load(REF_WAV_PATH, sr=16000)
preprocessed_wav = encoder.preprocess_wav(original_wav)
embed = encoder.embed_utterance(preprocessed_wav)
print("✅ RTVC encoder embedding shape:", embed.shape)

# =====================
# SYNTHESIZER: Text → Mel
# =====================
print("Synthesizing mel spectrogram with RTVC synthesizer...")
specs = synthesizer.synthesize_spectrograms([TEXT], [embed])
mel = specs[0]  # (80, T)
mel_torch = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).to(device)  # [1, T, 80]
print("✅ RTVC mel shape:", mel.shape)

# =====================
# VOCODER: Mel → Waveform (My vocoder)
# =====================
print("Generating waveform with my vocoder...")
with torch.no_grad():
    generated_wav = my_vocoder(mel_torch).squeeze(0).cpu().numpy()

# Save output
sf.write(OUTPUT_WAV_PATH, generated_wav.astype(np.float32), 22050)
print(f"✅ Reverse hybrid output saved to: {OUTPUT_WAV_PATH}")
