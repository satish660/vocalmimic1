import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from pathlib import Path
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
import plotly.express as px
import os

# Import your model code
from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder

# Load models
encoder.load_model(Path("encoder/saved_models/encoder.pt"))
synthesizer = Synthesizer(Path("synthesizer/saved_models/synthesizer.pt"))
vocoder.load_model(Path("vocoder/saved_models/vocoder.pt"))

st.set_page_config(layout="wide")
st.title("🎙 VOCALMIMIC: Voice Cloning with Deep Learning")
st.markdown("Upload a reference voice and input text to generate a clone. Optionally, compare similarity with another voice.")

# Inputs
ref_audio = st.file_uploader("Upload Reference Voice (.wav)", type=["wav"])
cmp_audio = st.file_uploader("Upload Comparison Voice (.wav, optional)", type=["wav"])
text = st.text_area("Text to Synthesize", height=80)

if st.button("Clone Voice") and ref_audio and text.strip():
    with st.spinner("Processing..."):
        # Preprocess reference audio
        wav_ref, _ = librosa.load(ref_audio, sr=16000)
        preprocessed_ref = encoder.preprocess_wav(wav_ref)
        embed_ref = encoder.embed_utterance(preprocessed_ref)

        # Synthesize and vocode
        spectrogram = synthesizer.synthesize_spectrograms([text], [embed_ref])[0]
        generated_wav = vocoder.infer_waveform(spectrogram)
        generated_wav = np.pad(generated_wav, (0, 4000), mode="constant")

        # Save output
        output_path = "cloned_output.wav"
        sf.write(output_path, generated_wav, 16000)

        # Plot Speaker Embedding
        st.subheader("🔢 Speaker Embedding")
        fig_embed, ax = plt.subplots(figsize=(10, 2))
        ax.bar(np.arange(len(embed_ref)), embed_ref)
        ax.set_title("Speaker Embedding")
        st.pyplot(fig_embed)

        # Plot Mel Spectrogram
        st.subheader("📊 Mel Spectrogram")
        fig_spec, ax2 = plt.subplots(figsize=(8, 4))
        librosa.display.specshow(spectrogram.T, sr=16000, hop_length=200, x_axis='time', y_axis='mel')
        ax2.set_title("Generated Mel Spectrogram")
        st.pyplot(fig_spec)

        # Plot Waveform
        st.subheader("📈 Generated Waveform")
        fig_wave, ax3 = plt.subplots(figsize=(10, 2))
        ax3.plot(generated_wav)
        ax3.set_title("Generated Audio Waveform")
        st.pyplot(fig_wave)

        # Play audio
        st.subheader("🔊 Cloned Voice Output")
        st.audio(output_path, format="audio/wav")

        # Comparison
        if cmp_audio:
            wav_cmp, _ = librosa.load(cmp_audio, sr=16000)
            preprocessed_cmp = encoder.preprocess_wav(wav_cmp)
            embed_cmp = encoder.embed_utterance(preprocessed_cmp)
            similarity = 1 - cosine(embed_ref, embed_cmp)
            st.success(f"✅ Cosine Similarity Score: {similarity:.4f}")

            # PCA Plot
            st.subheader("🧭 PCA of Speaker Embeddings")
            pca = PCA(n_components=2)
            reduced = pca.fit_transform([embed_ref, embed_cmp])
            fig_pca = px.scatter(
                x=reduced[:, 0], y=reduced[:, 1],
                text=["Reference", "Comparison"],
                labels={"x": "PCA 1", "y": "PCA 2"},
                title="PCA Projection of Speaker Embeddings"
            )
            fig_pca.update_traces(marker=dict(size=10))
            st.plotly_chart(fig_pca, use_container_width=True)