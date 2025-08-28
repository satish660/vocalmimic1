import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from pathlib import Path
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
import plotly.express as px

# Disable Brotli compression bug in Gradio
import os
os.environ["GRADIO_DISABLE_COMPRESSION"] = "1"

# Import your model code
from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder

# Load models
encoder.load_model(Path("encoder/saved_models/encoder.pt"))
synthesizer = Synthesizer(Path("synthesizer/saved_models/synthesizer.pt"))
vocoder.load_model(Path("vocoder/saved_models/vocoder.pt"))

def clone_and_visualize(reference_audio, comparison_audio, text):
    # Preprocess reference audio
    wav_ref, _ = librosa.load(reference_audio, sr=16000)
    preprocessed_ref = encoder.preprocess_wav(wav_ref)
    embed_ref = encoder.embed_utterance(preprocessed_ref)

    # Generate spectrogram and waveform
    spectrogram = synthesizer.synthesize_spectrograms([text], [embed_ref])[0]
    generated_wav = vocoder.infer_waveform(spectrogram)
    generated_wav = np.pad(generated_wav, (0, 4000), mode="constant")

    # Save generated audio
    output_path = "cloned_output.wav"
    sf.write(output_path, generated_wav, 16000)

    # Plot speaker embedding
    fig_embed, ax = plt.subplots(figsize=(10, 2))
    ax.bar(np.arange(len(embed_ref)), embed_ref)
    ax.set_title("Speaker Embedding")

    # Plot mel spectrogram
    fig_spec, ax2 = plt.subplots(figsize=(8, 4))
    librosa.display.specshow(spectrogram.T, sr=16000, hop_length=200, x_axis='time', y_axis='mel')
    ax2.set_title("Generated Mel Spectrogram")

    # Plot waveform
    fig_wave, ax3 = plt.subplots(figsize=(10, 2))
    ax3.plot(generated_wav)
    ax3.set_title("Generated Audio Waveform")

    # Optional: compare with second audio
    similarity_score = "N/A"
    pca_plot = None
    if comparison_audio:
        wav_cmp, _ = librosa.load(comparison_audio, sr=16000)
        preprocessed_cmp = encoder.preprocess_wav(wav_cmp)
        embed_cmp = encoder.embed_utterance(preprocessed_cmp)

        similarity_score = 1 - cosine(embed_ref, embed_cmp)

        # Dimensionality reduction plot (PCA)
        pca = PCA(n_components=2)
        reduced = pca.fit_transform([embed_ref, embed_cmp])
        pca_plot = px.scatter(
            x=reduced[:, 0], y=reduced[:, 1],
            text=["Reference", "Comparison"],
            labels={"x": "PCA 1", "y": "PCA 2"},
            title="PCA Projection of Speaker Embeddings"
        )
        pca_plot.update_traces(marker=dict(size=10))

    return output_path, fig_embed, fig_spec, fig_wave, similarity_score, pca_plot

# Define the GUI
iface = gr.Interface(
    fn=clone_and_visualize,
    inputs=[
        gr.Audio(type="filepath", label="Reference Voice (.wav)"),
        gr.Audio(type="filepath", label="(Optional) Comparison Voice (.wav)"),
        gr.Textbox(lines=2, label="Text to Synthesize")
    ],
    outputs=[
        gr.Audio(type="filepath", label="Cloned Voice Output"),
        gr.Plot(label="Speaker Embedding (Bar Chart)"),
        gr.Plot(label="Mel Spectrogram"),
        gr.Plot(label="Waveform"),
        gr.Textbox(label="Cosine Similarity Score (0-1)"),
        gr.Plot(label="PCA of Embeddings (2D)")
    ],
    title="VOCALMIMIC: Voice Cloning with deep Learning",
    description="Explore how voice cloning works with speaker embeddings, mel spectrograms, and vocoder output."
)

iface.launch()
