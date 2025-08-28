import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder
from pathlib import Path
import librosa
import soundfile as sf

encoder.load_model(Path("encoder/saved_models/encoder.pt"))
synthesizer = Synthesizer(Path("synthesizer/saved_models/synthesizer.pt"))
vocoder.load_model(Path("vocoder/saved_models/vocoder.pt"))

def visualize(reference_audio, text):
    wav, _ = librosa.load(reference_audio, sr=16000)
    preprocessed = encoder.preprocess_wav(wav)
    embed = encoder.embed_utterance(preprocessed)
    spec = synthesizer.synthesize_spectrograms([text], [embed])[0]
    generated_wav = vocoder.infer_waveform(spec)

    # Plot Embedding
    fig_embed, ax = plt.subplots(figsize=(10, 2))
    ax.bar(range(len(embed)), embed)
    ax.set_title("Speaker Embedding")

    # Plot Spectrogram
    fig_spec, ax = plt.subplots(figsize=(8, 4))
    librosa.display.specshow(spec.T, sr=16000, hop_length=200, x_axis='time', y_axis='mel')
    ax.set_title("Generated Mel Spectrogram")

    # Save generated audio
    output_path = "output.wav"
    sf.write(output_path, generated_wav, 16000)

    return output_path, fig_embed, fig_spec

# Gradio interface
iface = gr.Interface(
    fn=visualize,
    inputs=[
        gr.Audio(type="filepath", label="Reference Voice (.wav)"),
        gr.Textbox(label="Text to Synthesize")
    ],
    outputs=[
        gr.Audio(type="filepath", label="Cloned Voice Output"),
        gr.Plot(label="Speaker Embedding"),
        gr.Plot(label="Generated Mel Spectrogram")
    ],
    title="Voice Cloning with Visualization",
    description="Upload a voice sample and type some text. See the speaker embedding and spectrogram as the system synthesizes speech."
)

iface.launch()
