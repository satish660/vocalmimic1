# run_clone.py
import numpy as np
import librosa
import soundfile as sf
import os
import sys

# Adjust path to import local encoder, synthesizer, vocoder
sys.path.append(os.path.abspath("D:/realtimevoice/encoder"))
sys.path.append(os.path.abspath("D:/realtimevoice/synthesizer"))
sys.path.append(os.path.abspath("D:/realtimevoice/vocoder"))

# Import model APIs
from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder

# Paths to models
from pathlib import Path

encoder_model_fpath = Path("D:/realtimevoice/encoder/saved_models/encoder.pt")
synthesizer_model_fpath = Path("D:/realtimevoice/synthesizer/saved_models/synthesizer.pt")
vocoder_model_fpath = Path("D:/realtimevoice/vocoder/saved_models/vocoder.pt")


# Step 1: Load models
print("Loading models...")
encoder.load_model(encoder_model_fpath)
synthesizer = Synthesizer(synthesizer_model_fpath)
vocoder.load_model(vocoder_model_fpath)

# Step 2: Load reference voice
print("Loading reference audio...")
ref_wav_path = "D:/realtimevoice/obama.wav"  # You can change this
original_wav, sampling_rate = librosa.load(ref_wav_path, sr=16000)
preprocessed_wav = encoder.preprocess_wav(original_wav)
embed = encoder.embed_utterance(preprocessed_wav)

# Step 3: Synthesize spectrogram from text
text = "This is a voice cloning test using pre-trained models only."
print("Synthesizing spectrogram...")
specs = synthesizer.synthesize_spectrograms([text], [embed])

# Step 4: Vocoder to generate waveform
print("Generating waveform...")
generated_wav = vocoder.infer_waveform(specs[0])
# Optional: Pad to avoid cutting off the end
generated_wav = np.pad(generated_wav, (0, 4000), mode="constant")

# Step 5: Save output
output_path = "D:/realtimevoice/cloned_output.wav"
sf.write(output_path, generated_wav, 16000)
print(f"Voice cloning complete! Output saved to: {output_path}")
