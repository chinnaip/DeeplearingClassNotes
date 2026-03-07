# Deep Learning — Test Answer Key

> **Format guide**: Diagrams → Mermaid | Math → LaTeX | Tables → Markdown

---

## SECTION A — 4 Marks

---

### A1 · Supervised vs Unsupervised Learning

| Feature | Supervised | Unsupervised |
|---|---|---|
| **Labels** | ✅ Required | ❌ None |
| **Goal** | Learn mapping $f: X \to Y$ | Discover hidden structure |
| **Output** | Class / Value | Cluster / Embedding |
| **Examples** | Image classification, Regression | K-Means, Autoencoders |

```mermaid
flowchart LR
    subgraph SL["Supervised Learning"]
        direction TB
        X1["Input X"] --> M1["Model f(x)"] --> Y1["Label Y"]
    end
    subgraph UL["Unsupervised Learning"]
        direction TB
        X2["Input X"] --> M2["Model g(x)"] --> C2["Structure / Cluster"]
    end
```

**Key equations**

$$\text{Supervised: } \hat{y} = f_\theta(x), \quad \mathcal{L}(\hat{y}, y)$$

$$\text{Unsupervised: minimise } \mathcal{L}(x, \hat{x}) \text{ or maximise cluster cohesion}$$

---

### A2 · Loss Functions

> A **loss function** $\mathcal{L}$ measures how far model predictions are from true values. Training minimises $\mathcal{L}$ via gradient descent.

| Name | Formula | Used For |
|---|---|---|
| **MSE** | $\mathcal{L} = \dfrac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$ | Regression |
| **Cross-Entropy** | $\mathcal{L} = -\sum_{c} y_c \log(\hat{y}_c)$ | Classification |

```mermaid
flowchart LR
    Pred["Prediction ŷ"] & True["True Label y"] --> Loss["Loss ℒ(ŷ, y)"]
    Loss --> Grad["Gradient ∂ℒ/∂θ"]
    Grad --> Update["θ ← θ − η∇ℒ"]
```

---

## SECTION B — 6 Marks

---

### B1 · Overfitting & Regularisation

```mermaid
graph LR
    subgraph Fit["Model Complexity"]
        U["Underfitting\nhigh bias"] --> G["Good Fit"] --> O["Overfitting\nhigh variance"]
    end
```

| Symptom | Training Loss | Validation Loss |
|---|---|---|
| Underfitting | High | High |
| **Overfitting** | **Low** | **High** |
| Good Fit | Low | Low |

#### Technique 1 — L2 Regularisation (Weight Decay)

$$\mathcal{L}_{\text{reg}} = \mathcal{L} + \frac{\lambda}{2}\|\theta\|^2$$

- Penalises large weights → forces simpler model  
- $\lambda$ controls strength

#### Technique 2 — Dropout

$$\tilde{h}_i = h_i \cdot \text{Bernoulli}(p) \quad \text{(during training)}$$

```mermaid
flowchart LR
    H1["h₁"] & H2["h₂"] & H3["h₃"] & H4["h₄"]
    H1 -->|kept| N
    H2 -->|"dropped ✗"| X:::drop
    H3 -->|kept| N
    H4 -->|"dropped ✗"| Y:::drop
    N["Next Layer"]
    classDef drop fill:#f88,stroke:#f00
```

---

### B2 · Backpropagation & Chain Rule

> **Backpropagation** efficiently computes $\frac{\partial \mathcal{L}}{\partial \theta}$ for every parameter by applying the **chain rule** in reverse through the network.

#### Chain Rule (core idea)

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$$

where $z = wx + b$, $a = \sigma(z)$.

#### Forward & Backward Pass

```mermaid
flowchart LR
    x["x"] --> z1["z¹=W¹x+b¹"] --> a1["a¹=σ(z¹)"]
    a1 --> z2["z²=W²a¹+b²"] --> a2["a²=σ(z²)"]
    a2 --> L["ℒ"]

    L -->|"∂ℒ/∂a²"| da2["δ²"]
    da2 -->|"∂ℒ/∂W²"| dW2["∇W²"]
    da2 -->|"∂ℒ/∂a¹ (chain)"| da1["δ¹"]
    da1 -->|"∂ℒ/∂W¹"| dW1["∇W¹"]
```

| Step | Direction | Purpose |
|---|---|---|
| Forward Pass | $x \to \hat{y}$ | Compute predictions & loss |
| Backward Pass | $\hat{y} \to x$ | Compute gradients via chain rule |
| Update | — | $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$ |

**Why chain rule is essential**: Layers are composed functions. Without the chain rule there is no way to propagate error signals back through multiple non-linear transformations.

---

## SECTION C — 10 Marks

---

### C1 · CNN Pipeline for Image Classification

```mermaid
flowchart LR
    IMG["Input Image\nH×W×C"] --> CONV["Conv Layer\n(filters, stride, padding)"]
    CONV --> ACT["Activation\nReLU"]
    ACT --> POOL["Pooling\nMax/Avg"]
    POOL --> CONV
    POOL --> FLAT["Flatten\nH×W×F → vector"]
    FLAT --> FC["Fully-Connected\nLayers"]
    FC --> SOFT["Softmax → Class Probabilities"]
```

#### Key Operations

| Operation | Formula / Description |
|---|---|
| Convolution | $(\mathbf{I} * \mathbf{K})_{i,j} = \sum_m \sum_n \mathbf{I}_{i+m,\, j+n} \cdot \mathbf{K}_{m,n}$ |
| Output size | $O = \dfrac{W - F + 2P}{S} + 1$ |
| Stride $S$ | Step size of filter movement |
| Padding $P$ | Zero-border added to preserve spatial size |
| Max Pooling | $y = \max(\text{patch})$ — reduces spatial dims |
| Flatten | Reshape feature maps to 1-D vector |
| FC + Softmax | $\hat{y}_c = \dfrac{e^{z_c}}{\sum_k e^{z_k}}$ |

#### Parameter Sharing (why CNNs work for images)

```mermaid
graph TD
    A["Same filter slides over ALL positions"] --> B["Fewer parameters than FC"]
    B --> C["Translation invariance"]
    C --> D["Generalises to unseen positions"]
```

- **Local connectivity** — each neuron sees a small patch  
- **Parameter sharing** — same weights detect the same feature anywhere  
- **Hierarchical features** — early layers: edges → deeper layers: shapes → objects

---

### C2 · Transformer Architecture

```mermaid
flowchart TB
    subgraph ENC["Encoder Block (×N)"]
        E1["Input Embeddings + Positional Encoding"]
        E1 --> MHA["Multi-Head Self-Attention"]
        MHA --> AN1["Add & Norm"]
        AN1 --> FFN["Feed-Forward Network"]
        FFN --> AN2["Add & Norm"]
    end
    subgraph DEC["Decoder Block (×N)"]
        D1["Output Embeddings + Positional Encoding"]
        D1 --> MMHA["Masked Multi-Head Self-Attention"]
        MMHA --> AN3["Add & Norm"]
        AN3 --> MHA2["Cross-Attention (Enc-Dec)"]
        MHA2 --> AN4["Add & Norm"]
        AN4 --> FFN2["Feed-Forward Network"]
        FFN2 --> AN5["Add & Norm"]
    end
    AN2 -->|"Key, Value"| MHA2
    AN5 --> LIN["Linear + Softmax → Output"]
```

#### Self-Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

| Symbol | Meaning |
|---|---|
| $Q$ | Query matrix ($n \times d_k$) |
| $K$ | Key matrix ($n \times d_k$) |
| $V$ | Value matrix ($n \times d_v$) |
| $d_k$ | Dimension of keys (scaling factor) |

#### Multi-Head Attention Intuition

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q,\; KW_i^K,\; VW_i^V)$$

- Each head attends to **different aspects** of the sequence (syntax, semantics, co-reference…)

#### Positional Encoding

$$PE_{(pos,\, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos,\, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

- Adds order information to otherwise **position-agnostic** embeddings

#### Transformer vs RNN

| Feature | RNN / LSTM | Transformer |
|---|---|---|
| **Parallelism** | ❌ Sequential (time steps) | ✅ All positions at once |
| **Long-range deps** | ⚠️ Vanishing gradient | ✅ Direct attention |
| **Training speed** | Slow (no GPU parallelism) | Fast |
| **Context window** | Limited by memory | Fixed but large |
| **Complexity** | $O(n \cdot d^2)$ | $O(n^2 \cdot d)$ |

> **Why Transformers parallelise better**: every token attends to every other token simultaneously via matrix operations, whereas RNNs must process tokens one-by-one, creating a sequential bottleneck.

---

*Answer key compiled for Deep Learning exam revision — A4 hand-copy format.*
