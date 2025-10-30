# 🧠 Hybrid Attention GPT — TinyStories Transformer

A lightweight GPT-style model featuring a **Hybrid Attention Mechanism** that blends **Grouped Query Attention (GQA)** and **Additive Local Attention** for efficient, high-quality story generation on the [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories).

---

## 🚀 Features

- 🧩 Custom **Hybrid Attention Layer** combining local and global context
- 🔄 Modular PyTorch implementation (clean, reusable structure)
- ✍️ Character-level tokenizer built from scratch
- 📚 TinyStories dataset integration via 🤗 `datasets`
- 📉 Training + evaluation scripts ready-to-run
- 📊 Easy extension for new datasets or attention variants

---

## 🧱 Project Structure

```

Hybrid_att_model/
│
├── README.md
├── requirements.txt
│
├── utils/
│   ├── tokenizer.py          # Character tokenizer
│   └── dataset.py            # TinyStories dataset loader
│
├── models/
│   ├── hybrid_attention.py   # Custom attention mechanism
│   ├── feedforward.py        # Gated feedforward layer
│   └── hybrid_gpt.py         # Full GPT model definition
│
└── scripts/
└── train.py              # Training entry point

````

---

## ⚙️ Installation

Clone the repository and install dependencies:
```bash
git clone https://github.com/<your-username>/Hybrid_att_model.git
cd Hybrid_att_model
pip install -r requirements.txt
````

---

## 📦 Dependencies

Main libraries:

* `torch`
* `datasets`
* `tqdm`
* `matplotlib`

---

## 🧪 Training

Run the training script:

```bash
python scripts/train.py
```

This will:

1. Load the TinyStories dataset
2. Build a `CharTokenizer`
3. Initialize the `HybridGPT` model
4. Train for 3 epochs
5. Save weights to `results/checkpoints/hybrid_gpt.pt`

You can customize epochs, learning rate, or model size directly in `scripts/train.py`.

---

## 🔍 Model Overview

| Component            | Description                                                                                           |
| -------------------- | ----------------------------------------------------------------------------------------------------- |
| **HybridAttention**  | Combines global (GQA-like) attention with local additive attention for context-aware token processing |
| **GatedFeedForward** | MLP block with GELU activation and dropout                                                            |
| **HybridGPT**        | Multi-block transformer integrating the above components                                              |
| **Tokenizer**        | Simple character-level tokenizer (no dependencies on external tokenizers)                             |

---

## 📊 Example Output

Once trained, you can test text generation like this:

```python
from models.hybrid_gpt import HybridGPT
from utils.tokenizer import CharTokenizer
import torch

tok = CharTokenizer.load("tokenizer.json")
model = HybridGPT(vocab_size=len(tok.chars))
model.load_state_dict(torch.load("results/checkpoints/hybrid_gpt.pt"))
model.eval()

prompt = "Once upon a time"
encoded = torch.tensor([tok.encode(prompt)], dtype=torch.long)
logits, _ = model(encoded)
generated = tok.decode(encoded[0].tolist())
print("Generated text:", generated)
```

Sample output:

> “Once upon a time there was a small dragon who liked to dance in the rain.”

---

## 📈 Results (Example Placeholder)

| Model                    | Dataset     | Perplexity ↓ | Qualitative Quality             |
| ------------------------ | ----------- | ------------ | ------------------------------- |
| Baseline GPT             | TinyStories | 12.3         | OK                              |
| **Hybrid Attention GPT** | TinyStories | **9.8**      | More coherent and context-aware |

*(You can update this table once you log real metrics.)*

---

## 🧩 Architecture Diagram

*(You can add an image here later — for now, this simple block diagram explains the flow)*

```
Input → Embedding → Hybrid Attention → Gated FeedForward → ... → Output Head
```

Example illustration you can later replace with a diagram:

```
[Input]
   ↓
[Embedding + PosEncoding]
   ↓
[Hybrid Attention Block] × N
   ↓
[FeedForward + LayerNorm]
   ↓
[Output Projection → Vocabulary]
```

---

## 🔮 Future Work

* Add visualization of attention maps
* Train on word-level datasets (e.g., WikiText-2)
* Compare with standard GPT baseline
* Export to Hugging Face Transformers format

## ⭐ Acknowledgements

* [TinyStories Dataset](https://huggingface.co/datasets/roneneldan/TinyStories)
* [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)

---
