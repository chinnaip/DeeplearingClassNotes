# Session 03 Visual Extraction Report

## Metadata
- **Session**: 03 - Deep Learning Live Session 3
- **Vimeo URL**: https://vimeo.com/1140267865
- **Duration**: 1:26:48
- **Instructor(s)**: Prof. Prasanna (theory), Sunil (demo/examples)
- **Topics**: Biological neuron recap, McCulloch-Pitts model, Perceptron model, Perceptron learning law, Bias role, Perceptron trick algorithm, Decision boundary, Linear separability, Multi-layer perceptron intro

## Visual Index

| # | Timestamp | Type | Description |
|---|-----------|------|-------------|
| 1 | 02:18 | Screen Share | Instructor begins screen sharing |
| 2 | 05:43 | Slide | Take Home Message - biological neuron concepts |
| 3 | 07:01 | Slide | Neuron activation, non-linear output function |
| 4 | 08:41 | Slide | Neural network layers, structure and organization |
| 5 | 09:33 | Slide | Activation dynamics vs synaptic dynamics |
| 6 | 11:58 | Slide | McCulloch-Pitts Model equations |
| 7 | 14:26 | Slide | Perceptron model - error function, weight update |
| 8 | 18:08 | Slide | Learning Laws - weight update general principle |
| 9 | 19:07 | Slide | Perceptron learning equation detailed |
| 10 | 23:37 | Slide | Perceptron working - inputs, weights, step function |
| 11 | 27:52 | Slide | Perceptron Working with Bias - diagram and step function |
| 12 | 30:00 | Code | Colab demo - PyTorch tensors, weighted sum |
| 13 | 32:37 | Code Output | Dot product result, step function output = 0 |
| 14 | 33:36 | Code Output | Bias added, weighted sum + bias = 0.12, output = 1 |
| 15 | 41:05 | Slide | Placement example - CGPA/IQ perceptron diagram |
| 16 | 47:41 | Slide | Perceptron Decision Boundary - calculation and plot |
| 17 | 53:21 | External Tool | Perceptron visualization website - animation |
| 18 | 58:49 | External Tool | Visualization converged - 29 iterations, 0 misclassifications |
| 19 | 1:00:04 | Slide | Perceptron trick algorithm steps |
| 20 | 1:04:00 | Code | Colab - data setup, CGPA/IQ tensor creation |
| 21 | 1:06:30 | Code | Colab - bias concatenation, weight initialization |
| 22 | 1:08:30 | Code | Colab - training loop, weight history |
| 23 | 1:10:30 | Code Output | Decision boundary plots across epochs |
| 24 | 1:18:38 | Slide | Perceptron Representation Problem - linearly inseparable |
| 25 | 1:22:30 | Slide | Multi-layer perceptron introduction |

## Visual Reconstructions

### V2: Take Home Message Slide (05:43)
**Type**: Lecture Slide

Key points from slide:
- Cell body acts like a summing device
- Synaptic junction repeatedly triggered grows in strength, others weaken
- Strength of synaptic connection gets modified continuously
- Synaptic plasticity plays dominant role in learning
- Each neuron receives input from several, sends to several
- Stacked in several layers

### V6: McCulloch-Pitts Model (11:58)
**Type**: Lecture Slide - Equations

**Reconstructed Content**:
- Activation value: `x = sum(w_i * a_i) - theta` (weighted sum of M inputs and bias)
- Bias (theta) allows shifting activation function by adding constant
- Output Value: `s = f(x)` is typically non-linear function (binary) of x
- Limitation: fixed weights and no learning
- Purpose: Capturing neuron function

### V7: Perceptron Model (14:26)
**Type**: Lecture Slide - Equations

**Key equations**:
- Error: `delta = B - S` (target minus actual output)
- Weight update: `delta_W = eta * delta * a_i`
- eta = learning rate (typically 0.001 to 0.01)
- Convergence: weights converge for linearly separable classes
- Supervised learning: requires target output for each input

### V8: Learning Laws (18:08)
**Type**: Lecture Slide

**Content**:
- Activation dynamics: activation state as function of time
- Synaptic dynamics: weights state as function of time
- Learning laws: implementation of synaptic dynamics
- General weight update: `w_i(t+1) = w_i(t) + delta_w_i(t)`
- Learning laws listed: Hebbian, Perceptron, Delta, LMS, Correlation, Instar, Outstar

### V11: Perceptron Working with Bias (27:52)
**Type**: Lecture Slide + Diagram

```
Perceptron with Bias:
  x1=0.2 --w1=-0.3-->
  x2=0.4 --w2=0.4 --> [SUM(w_i*x_i) + b] --> [Step] --> output
  x3=0.6 --w3=-0.8-->
                       b=0.5

Without bias: -0.38 --> Step --> 0
With bias: -0.38 + 0.5 = 0.12 --> Step --> 1
```

Step function graph: f(x)=0 for x<0, f(x)=1 for x>=0
Bias shifts decision boundary, adds flexibility to threshold

### V12-14: Colab Demo - Perceptron Basics (30:00-33:36)
**Type**: Code Demo (PyTorch)

```python
import torch
# Inputs and weights as tensors
x = torch.tensor([0.2, 0.4, 0.6])
weights = torch.tensor([-0.3, 0.4, -0.8])

# Weighted sum (dot product)
weighted_sum = torch.dot(weights, x)  # -0.37

# Step function
def step(x):
    return 1 if x > 0 else 0

step(weighted_sum)  # Output: 0

# With bias
bias = torch.tensor([0.5])
weighted_sum_bias = torch.dot(weights, x) + bias  # 0.12
step(weighted_sum_bias)  # Output: 1
```

### V16: Decision Boundary Calculation (47:41)
**Type**: Lecture Slide + Plot

**Decision boundary equation**: `w0*x0 + w1*x1 + w2*x2 = 0`
- Substituting w0=-1, w1=0.5, w2=-0.3
- Solving for x2: `x2 = (1 - 0.5*x1) / (-0.3)`
- Scatter plot: IQ/10 (x2) vs CGPA (x1), class 0 (blue), class 1 (green)

### V17-18: Perceptron Visualization (53:21-58:49)
**Type**: External Web Tool (vinizinho.net)

**Visualization details**:
- 100 data points, 2 classes (blue/red)
- Algorithm: Initialize weights to 0, randomly pick misclassified point, shift line
- Weight update: `w <- w + y_n * x_n`
- Converged after 29 iterations with 0/100 misclassifications
- Decision boundary successfully separates linearly separable data

### V19: Perceptron Trick Algorithm (1:00:04)
**Type**: Lecture Slide

**Algorithm steps**:
1. Initialize weights randomly, bias = 1
2. Compute weighted sum
3. Predict output using step function
4. Randomly choose a data point
5. If misclassified: update `w_i = w_i + eta * error * x_i`
6. Error = `y - y_hat` (can be +1 or -1)
7. Repeat until no misclassification or max epochs reached

### V20-23: Colab Training Demo (1:04:00-1:10:30)
**Type**: Code Demo (PyTorch)

```python
# Data: CGPA and IQ/10
X = torch.tensor([[6.1, 11.0], [7.2, 12.0], [8.2, 10.5]])
Y = torch.tensor([0, 1, 1])  # placement labels
# Add bias column (ones)
bias_col = torch.ones(X.shape[0], 1)
X_bias = torch.cat([bias_col, X], dim=1)  # 3x3 tensor

# Random weight initialization
w = 0.1 * torch.randn(3)

# Training loop
epochs = 100
for epoch in range(epochs):
    for i in range(len(X_bias)):
        y_hat = predict(X_bias[i], w)  # dot product + step
        error = Y[i] - y_hat
        if error != 0:
            w = w + 0.1 * error * X_bias[i]
```

Output: Decision boundary plots at epoch 1, 2, and final showing convergence

### V24-25: Multi-Layer Perceptron Intro (1:18:38-1:22:30)
**Type**: Lecture Slides

**Key concepts**:
- Single layer perceptron fails for non-linearly separable problems (XOR/XNOR)
- These are called "hard problems" in neural network terminology
- Two-layer perceptron can handle convex decision regions
- Three-layer needed for non-convex regions
- As non-linearity increases, number of layers increases

## Cross-Reference Matrix

| Topic | Slides | Code | External |
|-------|--------|------|----------|
| Biological neuron recap | V2 | - | - |
| MP Model | V6 | - | - |
| Perceptron equations | V7, V8, V9 | - | - |
| Perceptron working | V10, V11 | V12-14 | - |
| Bias role | V11 | V14 | - |
| Decision boundary | V16 | V20-23 | V17-18 |
| Perceptron trick | V19 | V20-23 | V17-18 |
| MLP introduction | V24-25 | - | - |

## Summary
Session 03 covers the transition from biological neuron concepts to computational models. Key progression: MP model (fixed weights) -> Perceptron (learnable weights via error-driven updates) -> limitations with non-linear data -> need for multi-layer networks. Includes PyTorch demos for weighted sum, bias effect, and perceptron trick training with CGPA/IQ placement classification example.
