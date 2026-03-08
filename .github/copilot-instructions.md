# Copilot Instructions — DeeplearingClassNotes

## Repository Overview

This repository contains notes and exam answers for a 24-session deep learning course.
Sessions cover: PyTorch fundamentals, perceptrons, MLPs, backpropagation, regularisation,
CNNs, RNNs, LSTMs, GRUs, Seq2Seq, attention mechanisms, and Transformers.

---

## Source of Truth

When generating or updating answer files, use **`latex_merged/`** as the primary content source.
These 10 merged LaTeX documents cover all sessions and contain authoritative definitions,
equations, and derivations.

Map topics to files:

| Topic | File(s) |
|---|---|
| Intro, PyTorch, tensors | `latex_merged/01_introduction_and_pytorch_foundations.tex` |
| Perceptron, MLP, activation functions | `latex_merged/02_perceptron_and_mlp.tex` |
| Backprop, chain rule, gradient flow | `latex_merged/03_neural_networks_and_backpropagation.tex` |
| Deep networks, vanishing gradients | `latex_merged/04_deep_networks_vanishing_gradients_and_assessment.tex` |
| Regularisation, Dropout, BatchNorm | `latex_merged/05_ann_hands_on_and_regularisation.tex` |
| CNN, convolution, pooling, parameter sharing | `latex_merged/06_convolutional_neural_networks.tex` |
| RNN, BPTT, vanishing gradient in sequences | `latex_merged/07_recurrent_neural_networks.tex` |
| LSTM, forget/input/output gates | `latex_merged/08_lstm_theory_and_practice.tex` |
| GRU, Seq2Seq, BiRNN | `latex_merged/09_gru_seq2seq_and_birnn.tex` |
| Attention mechanism, Transformers | `latex_merged/10_attention_mechanism_and_transformers.tex` |

Do **not** use `transcripts/` or `ai_systems_lab_transcripts/` as primary sources —
these are raw session recordings and may contain informal or imprecise language.

---

## Output Format

- Always output **GitHub-flavoured markdown** (`.md`).
- Never output LaTeX for answer files.
- Follow the Section A / B / C structure with mark allocations as in existing answer files.

### Answer Structure Template

```
## Section A (4 Marks)
### A1. <Topic> [2]
### A2. <Topic> [2]

## Section B (6 Marks)
### B1. <Topic> [3]
### B2. <Topic> [3]

## Section C (10 Marks)
### C1. <Topic> [5]
### C2. <Topic> [5]
```

---

## Diagram Rules

| Content type | Format | Notes |
|---|---|---|
| Simple flows (training loop, forward/backward pass, supervised vs unsupervised) | Mermaid `flowchart LR` or `flowchart TD` | Works well on GitHub |
| Transformer encoder/decoder architecture | Mermaid `flowchart TB` with `subgraph` blocks | Valid pattern — keep it |
| CNN layer stacks | **ASCII art** | Required — Mermaid cannot handle layer cycles |
| RNN/LSTM/GRU layer diagrams | **ASCII art** | Gate equations are better shown as math |
| Parameter sharing / hierarchy diagrams | Mermaid `graph TD` simple chain | OK for 4-node linear chains |

### Mermaid — what to avoid

- **No cycle arrows**: patterns like `POOL --> CONV` where an earlier node is a target will cause
  a "Diagram syntax error" on github.com. Never create cycles in Mermaid.
- **No `classDef` + `:::` syntax**: `classDef foo fill:#f88` with `node:::foo` is not supported
  in all GitHub Mermaid versions. Use plain nodes instead.
- **No long Unicode math strings in node labels**: avoid `∂ℒ/∂W²` inside `[ ]` labels;
  use short descriptive text and put the math below the diagram in a `$$` block.

### ASCII art layer stack pattern (CNN example)

```
Input (28x28x1)
      |
  Conv2D-32 (3x3, ReLU)   ->  320 params
      |
  MaxPool (2x2)            ->    0 params
      |
  Conv2D-64 (3x3, ReLU)   ->  18,496 params
      |
  MaxPool (2x2)            ->    0 params
      |
  Flatten -> Dense-128 (ReLU) -> Dense-10 (Softmax)
```

---

## Math Rules

- **Inline math**: `$...$`
- **Block math**: `$$...$$`
- Always show parameter calculations in three steps:

```
formula  ->  substitute values  ->  numeric result
```

Example:

$$\text{params} = (k \times k \times C_{in} + 1) \times C_{out} = (3 \times 3 \times 1 + 1) \times 32 = 320$$

---

## Parameter Calculation Convention

For every layer in a CNN/MLP answer, include a table with an explicit formula column:

| Layer | Formula | Params |
|---|---|---|
| Conv2D(1->32, 3x3) | $(3\times3\times1+1)\times32$ | 320 |
| Conv2D(32->64, 3x3) | $(3\times3\times32+1)\times64$ | 18,496 |
| Dense(576->128) | $576\times128+128$ | 73,856 |
| Dense(128->10) | $128\times10+10$ | 1,290 |
| **Total** | | **93,962** |
