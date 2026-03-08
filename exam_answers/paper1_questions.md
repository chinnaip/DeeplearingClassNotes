# Deep Learning — Model Question Paper 1

**Total Marks: 20 | Time: 2 Hours**
**Topics: PyTorch Tensors · Perceptron · MLP · Backpropagation · CNNs**

> Answer **all** questions. Mark allocations are shown in brackets.

---

## Section A (4 Marks)
*Short-answer questions — 2 marks each*

---

### A1. PyTorch Tensors [2]

(a) What is a PyTorch tensor? How does it differ from a NumPy array?  
(b) Write a one-line PyTorch statement to create a $3 \times 4$ tensor of zeros on the GPU.

---

### A2. Perceptron [2]

State the perceptron learning rule. For a single-input perceptron with weight $w = 0.5$, bias $b = 0$, and learning rate $\eta = 0.1$, compute the updated weight after one misclassified sample where $x = 1$ and $y = 1$ (true label), $\hat{y} = 0$ (predicted).

---

## Section B (6 Marks)
*Medium-answer questions — 3 marks each*

---

### B1. Multi-Layer Perceptron and Activation Functions [3]

(a) Draw the architecture of a 2-hidden-layer MLP with input size 4, hidden sizes 8 and 4, and output size 2.  
(b) Explain why non-linear activation functions are necessary in an MLP. What happens if all activations are linear?  
(c) Write the ReLU activation function formula and sketch its graph.

---

### B2. Backpropagation and the Chain Rule [3]

(a) State the chain rule for a composed function $\mathcal{L}(a(z(w)))$.  
(b) For a single neuron with $z = wx + b$, $a = \sigma(z)$, and loss $\mathcal{L}$, derive the gradient $\dfrac{\partial \mathcal{L}}{\partial w}$.  
(c) Briefly explain the difference between the forward pass and the backward pass in a neural network.

---

## Section C (10 Marks)
*Long-answer questions — 5 marks each*

---

### C1. Convolutional Neural Networks — Convolution and Pooling [5]

(a) Define the convolution operation for a 2-D input. Write the formula for the output at position $(i, j)$ given input $\mathbf{X}$, filter $\mathbf{W}$ of size $F \times F$, and bias $b$.  
(b) An input of size $28 \times 28$ is convolved with 32 filters of size $3 \times 3$, stride $S = 1$, and padding $P = 0$. Compute:
   - (i) the output spatial size,
   - (ii) the number of parameters in this convolutional layer (include bias).  
(c) Describe max pooling with a $2 \times 2$ window and stride 2. What is the output size after pooling the result from (b)?  
(d) Why is parameter sharing in CNNs advantageous over fully connected layers for image data?

---

### C2. CNN Architecture — Full Pipeline [5]

Design a CNN for classifying $28 \times 28$ grayscale images into 10 classes. Your answer must include:

(a) A complete layer-by-layer architecture (at least: Conv → ReLU → Pool → Conv → ReLU → Pool → Flatten → FC → Softmax).  
(b) A parameter count table for **every** layer using the formula:

$$\text{Conv params} = (F \times F \times C_{\text{in}} + 1) \times C_{\text{out}}$$

$$\text{FC params} = (\text{input\_size} \times \text{output\_size}) + \text{output\_size}$$

(c) Explain how the combination of local receptive fields, weight sharing, and pooling gives CNNs **translation invariance**.

---

*End of Paper 1*
