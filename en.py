import matplotlib.pyplot as plt

text = """
┌─────────────────────────────┐
│   Input Audio (.wav file)   │
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│  Preprocessing (trim, resample) │
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│     Mel Spectrogram         │
│  (frames × frequency bins)  │
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│     3-layer LSTM Network     │
│ (Temporal speaker features)  │
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│  Linear Projection Layer     │
│    (Down to 256 dims)        │
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│  L2 Normalization (Unit Vec)│
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│     256-D Speaker Embedding │
└─────────────────────────────┘
"""

plt.figure(figsize=(10, 12))
plt.text(0.5, 0.5, text, fontsize=12, fontfamily='monospace', ha='center', va='center')
plt.axis('off')
plt.tight_layout()
plt.savefig("encoder_workflow.png", dpi=300)
plt.show()
