# Deep Learning — Model Question Paper 2

**Total Marks: 20 | Time: 2 Hours**
**Topics: Activation Functions · Vanishing Gradients · Regularisation (Dropout, BatchNorm, L2) · RNN · BPTT**

> Answer **all** questions. Mark allocations are shown in brackets.

---

## Section A (4 Marks)
*Short-answer questions — 2 marks each*

---

### A1. Activation Functions [2]

(a) Write the formulas for Sigmoid $\sigma(z)$ and Tanh $\tanh(z)$.  
(b) For a Sigmoid unit, what is the maximum value of $\dfrac{d\sigma}{dz}$? At what value of $z$ does it occur? State why this matters for training.

---

### A2. Vanishing Gradient Problem [2]

(a) Define the vanishing gradient problem in the context of deep networks.  
(b) If the derivative of the Sigmoid activation is at most 0.25, explain (with a brief mathematical argument) why gradients shrink as they are propagated through 10 Sigmoid layers.

---

## Section B (6 Marks)
*Medium-answer questions — 3 marks each*

---

### B1. Dropout and Batch Normalisation [3]

(a) Describe how Dropout works during training and during inference (test time). Why is weight scaling applied at test time?  
(b) State the Batch Normalisation formula for a mini-batch of activations $\{x_i\}$. What are the learnable parameters $\gamma$ and $\beta$?  
(c) List **two** benefits of Batch Normalisation beyond regularisation.

---

### B2. L2 Regularisation (Weight Decay) [3]

(a) Write the regularised loss function for L2 regularisation with coefficient $\lambda$.  
(b) Derive the gradient of the L2-regularised loss with respect to weight $w$.  Show that this gradient update is equivalent to **weight decay**.  
(c) Compare L1 and L2 regularisation: which produces sparser weights, and why?

---

## Section C (10 Marks)
*Long-answer questions — 5 marks each*

---

### C1. Recurrent Neural Networks [5]

(a) Draw and label the unrolled architecture of an RNN over 3 time steps for a sequence input $x_1, x_2, x_3$.  
(b) Write the RNN state-update equation and output equation. Define all symbols.  
(c) An RNN has hidden size 64, input size 10, and output size 5. Compute the total number of trainable parameters (weight matrices $W_{hh}$, $W_{xh}$, $W_{hy}$, and all biases).  
(d) Why is the same set of weights $W_{hh}$ and $W_{xh}$ reused at every time step? What property does this encode?

---

### C2. Backpropagation Through Time (BPTT) [5]

(a) Explain how BPTT differs from standard backpropagation. Why is "unrolling" necessary?  
(b) For an RNN unrolled over $T$ time steps, write the expression for $\dfrac{\partial \mathcal{L}}{\partial W_{hh}}$ as a sum over time steps, applying the chain rule through the hidden states.  
(c) Explain the **vanishing gradient problem** specific to RNNs trained with BPTT. Use a mathematical argument involving products of Jacobians across time steps.  
(d) Name and briefly describe **two** strategies (other than LSTM/GRU) that help mitigate vanishing gradients in RNNs.

---

*End of Paper 2*
