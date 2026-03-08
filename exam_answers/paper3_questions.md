# Deep Learning — Model Question Paper 3

**Total Marks: 20 | Time: 2 Hours**
**Topics: LSTM Gates · GRU · Seq2Seq · Attention Mechanism · Transformers**

> Answer **all** questions. Mark allocations are shown in brackets.

---

## Section A (4 Marks)
*Short-answer questions — 2 marks each*

---

### A1. LSTM Gates [2]

(a) Name the **four** gates/components in an LSTM cell and state the role of each in one sentence.  
(b) Write the equation for the LSTM **forget gate** $f_t$. Explain what a value of $f_t \approx 0$ vs $f_t \approx 1$ means for the cell state.

---

### A2. Gated Recurrent Unit (GRU) [2]

(a) Name the **two** gates in a GRU and give the formula for each.  
(b) How does the GRU simplify the LSTM? State one advantage and one disadvantage of GRU compared to LSTM.

---

## Section B (6 Marks)
*Medium-answer questions — 3 marks each*

---

### B1. Sequence-to-Sequence (Seq2Seq) Models [3]

(a) Describe the encoder–decoder architecture of a Seq2Seq model. What is the **context vector** and what information does it carry?  
(b) Draw a diagram showing a Seq2Seq model translating the sentence "I am happy" to "Je suis heureux". Label the encoder, decoder, and context vector.  
(c) What is the **information bottleneck** problem in Seq2Seq? How does increasing encoder hidden size partially address it?

---

### B2. Attention Mechanism [3]

(a) Explain what problem the attention mechanism solves in Seq2Seq models.  
(b) Write the formula for computing the context vector $c_t$ using attention weights $\alpha_{t,s}$ over encoder hidden states $h_s$.  
(c) Describe how the attention weights $\alpha_{t,s}$ are computed (Bahdanau additive attention). Why is a softmax applied?

---

## Section C (10 Marks)
*Long-answer questions — 5 marks each*

---

### C1. Transformer Architecture [5]

(a) Draw the full Transformer encoder–decoder architecture (or provide a reference diagram) and label: positional encoding, multi-head attention, feed-forward sub-layer, add & norm layers.  
(b) Derive the **scaled dot-product attention** formula:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

Explain the purpose of the $\dfrac{1}{\sqrt{d_k}}$ scaling factor.  
(c) Describe **multi-head attention**: how is it computed and why is it beneficial compared to single-head attention?  
(d) Write the positional encoding formulas for even and odd dimensions. Why are sinusoidal functions used?

---

### C2. Transformers vs RNNs — Comparative Analysis [5]

(a) Fill in the comparison table:

| Property | RNN / LSTM | Transformer |
|---|---|---|
| Parallelism during training | ? | ? |
| Handling long-range dependencies | ? | ? |
| Computational complexity per layer | ? | ? |
| Positional information | ? | ? |
| Memory/context window | ? | ? |

(b) Explain why Transformers **parallelise** better than RNNs during training. Use the concepts of sequential dependency vs. matrix operations.  
(c) What is the computational complexity of self-attention with respect to sequence length $n$ and model dimension $d$? Why can this become a bottleneck for very long sequences?  
(d) Name **two** real-world applications where the Transformer architecture has replaced RNN-based approaches, and briefly justify why.

---

*End of Paper 3*
