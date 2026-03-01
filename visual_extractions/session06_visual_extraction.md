# Session 06: Backpropagation Hands-On with PyTorch

## Metadata
- **Session:** 06
- **Title:** Backpropagation Hands-On with PyTorch
- **Vimeo URL:** https://vimeo.com/1144087053
- **VTT File:** session06_transcript.vtt
- **Duration:** ~1:46:12
- **Instructors:** Mahadeva Prasanna (recap), Sunil Saumya (hands-on)
- **Topics:** Backpropagation recap, Forward/backward pass coding, PyTorch Autograd

---

## Visual Index

| # | Timestamp | Type | Description |
|---|-----------|------|-------------|
| 1 | 00:16:05 | Slide | Recap slides - previous session review |
| 2 | 00:17:00 | Slide | Two Layer Perceptron - Figure 4.12 three-layer feedforward neural network diagram |
| 3 | 00:18:06 | Slide | Hard Learning Challenge - MLP training problem |
| 4 | 00:19:03 | Slide | Feedforward network supervised learning - input-output patterns |
| 5 | 00:19:45 | Slide | Three-layer network with output layer K and hidden layer J - weight notation |
| 6 | 00:20:32 | Slide | Error backpropagation concept - hidden layer error estimation |
| 7 | 00:22:03 | Slide | Error propagation from output to hidden neurons |
| 8 | 00:23:03 | Slide | Backpropagation algorithm - error propagation mechanism |
| 9 | 00:24:10 | Slide | Sigmoidal activation function - step function replaced with differentiable function |
| 10 | 00:24:54 | Slide | Differential output function - sigmoid and its derivative curve |
| 11 | 00:25:00 | Slide | Hard Learning Problem slide - desired output only for output layer |
| 12 | 00:25:60 | Slide | Hard learning problem solved by differentiable nonlinear output function |
| 13 | 00:26:23 | Slide | Generalized Delta Rule - extending error correction to hidden layers |
| 14 | 00:26:32 | Slide | Weight change proportional to error - mathematical formulation |
| 15 | 00:27:29 | Slide | Discrete vs continuous perceptron learning law comparison |
| 16 | 00:27:53 | Slide | Continuous learning law with differentiable factor - weight update equation |
| 17 | 00:28:45 | Slide | Backpropagation derivation - chain rule for weight updates |
| 18 | 00:29:04 | Slide | Annotated Neural Network (2-2-1) for LPA Prediction - network diagram |
| 19 | 00:31:08 | Slide | Dataset table: IQ, CGPA, LPA with 4 rows (88/8, 60/9, 75/7, 127/5) |
| 20 | 00:32:00 | Slide | Forward propagation equations: Z1, H1, Z2, H2, Y_hat calculations |
| 21 | 00:33:30 | Slide | Forward pass result: Y_hat=10.36, Loss=54.17 (squared error) |
| 22 | 00:34:35 | Slide | Backpropagation update rule - gradient descent parameter update |
| 23 | 00:35:01 | Slide | Chain rule animations - partial derivatives flow diagram |
| 24 | 00:36:03 | Slide | Partial derivative chain rule for V1, V2, B3 gradients |
| 25 | 00:37:02 | Slide | Chain rule for H1 parameters: W11, W12, B1 with ReLu derivative |
| 26 | 00:38:07 | Slide | Chain rule for H2 parameters: W21, W22, B2 |
| 27 | 00:38:22 | Slide | Forward pass recap - substituted gradient values (97.152, 26.496, 14.72) |
| 28 | 00:39:05 | Slide | All nine gradients calculated for parameter update |
| 29 | 00:39:48 | Slide | Learning rate 10^-4 discussion - exploding gradient problem preview |
| 30 | 00:41:20 | Slide | Updated parameters after one iteration |
| 31 | 00:41:54 | Slide | New Y_hat=0.99, New Loss=4.005, comparison old vs new |
| 32 | 00:47:59 | Screen | Google search for convex function - loss curve visualization |
| 33 | 00:48:07 | Screen | Convex curve diagram - gradient descent left/right side explanation |
| 34 | 00:51:06 | Screen | Google Colab - Untitled6.ipynb opened for hands-on coding |
| 35 | 00:53:15 | Code | Colab: torch import and X tensor creation (2D: 4x2) |
| 36 | 00:55:00 | Code | Colab: Y tensor creation (4x1: 3, 4, 8, 11) |
| 37 | 00:57:00 | Code | Colab: Parameter initialization (W11=0.02, W12=0.50, W21=0.04, W22=-0.30, V1=1.2, V2=0.8, B1=B2=B3=1) |
| 38 | 00:59:48 | Code | Colab: Learning rate=1e-4, epochs=2, loss_history initialization |
| 39 | 01:01:00 | Code | Colab: ReLu function definition using torch.where |
| 40 | 01:01:30 | Code | Colab: ReLu derivative function definition |
| 41 | 01:03:00 | Code | Colab: Training loop - forward pass (Z1, H1, Z2, H2, Y_hat, loss) |
| 42 | 01:07:30 | Code | Colab: Backward pass - hand-coded gradient calculations (9 gradients) |
| 43 | 01:13:00 | Code | Colab: Chain rule for H1 gradients (DL_DH1, DH1_DZ1, DW11, DW12, DB1) |
| 44 | 01:17:00 | Code | Colab: Chain rule for H2 gradients (DW21, DW22, DB2) |
| 45 | 01:18:30 | Code | Colab: Parameter update rule (W -= lr * gradient) |
| 46 | 01:22:00 | Code | Colab: Debugging shape mismatch error in parameter update |
| 47 | 01:26:28 | Code | Colab: Loss history output [54.1696, 4.005] matching hand calculations |
| 48 | 01:27:30 | Code | Colab: PyTorch Autograd.ipynb - second notebook opened |
| 49 | 01:28:45 | Code | Colab: Autograd version - requires_grad=True for all parameters |
| 50 | 01:29:04 | Code | Colab: loss.backward() replacing all hand-coded gradient lines |
| 51 | 01:30:30 | Code | Colab: Autograd loss history [54.1695, 4.005] - matches manual version |
| 52 | 01:32:00 | Code | Colab: Forward pass code with z1, h1, z2, h2, y_hat, loss |
| 53 | 01:33:01 | Code | Colab: Autograd complete code - forward pass, backward(), manual parameter update |
| 54 | 01:36:00 | Code | Colab: Zero gradient initialization importance discussion |
| 55 | 01:38:00 | Code | Colab: Computational graph explanation - add_backward, relu_backward |
| 56 | 01:41:00 | Code | Colab: Z1 tensor showing grad_fn=AddBackward, H1 showing ReluBackward |

---

## Visual Reconstructions

### V2: Two Layer Perceptron (00:17:00)
- **Source:** Textbook Figure 4.12
- **Content:** Three-layer feedforward neural network with input layer (I nodes), hidden layer (J nodes), output layer (K nodes)
- **Labels:** Weights W_ij^h (input-to-hidden), W_jk^o (hidden-to-output), bias nodes
- **Caption:** "Figure 4.12 A three layer feedforward neural network" from Neural Networks, PHI, New Delhi, 1999
- **Verified:** Yes (screenshot at ~17:35)

### V18: Annotated Neural Network (2-2-1) for LPA Prediction (00:29:04)
- **Source:** Instructor slide
- **Content:** Network diagram with X1, X2 inputs; H1, H2 hidden nodes; Y_hat output
- **Labels:** W11, W12, W21, W22 (input-hidden weights), V1, V2 (hidden-output weights), B1, B3 (biases)
- **Equations:** Z1=W11*X1+W12*X2+B1, H1=ReLu(Z1), Y_hat=V1*H1+V2*H2+B3
- **Verified:** Yes (screenshot at ~32:40)

### V33: Convex Curve for Gradient Descent (00:48:07)
- **Source:** Google Images search during live session
- **Content:** Convex loss function curve with X-axis as weight, Y-axis as loss
- **Context:** Explaining why weights may increase or decrease depending on position in curve
- **Verified:** Yes (screenshot at ~47:51)

### V41: Training Loop - Forward Pass Code (01:03:00)
- **Source:** Google Colab (Untitled6.ipynb)
- **Content:** Forward pass code: Z1, H1 (ReLu), Z2, H2, Y_hat, squared error loss
- **Verified:** Yes (screenshot at ~1:02:56)

### V50: Autograd loss.backward() (01:29:04)
- **Source:** Google Colab (PyTorch Autograd.ipynb)
- **Content:** Single line loss.backward() replacing all hand-coded gradient calculations
- **Key Point:** Autograd calculates all 9 gradients automatically via computational graph
- **Verified:** Yes (screenshot at ~1:33:05)

### V53: Autograd Complete Code (01:33:01)
- **Source:** Google Colab (PyTorch Autograd.ipynb)
- **Content:** Full training code with forward pass, loss.backward(), manual parameter updates
- **Key Output:** Loss [54.1695, 4.005] matching manual version exactly
- **Verified:** Yes (screenshot at ~1:33:05)

---

### Cross-Reference Matrix

| Topic | VTT Cues | Visual Refs |
|-------|----------|-------------|
| Two Layer Perceptron Recap | 00:17:00 | V2 |
| Hard Learning Problem | 00:18:06-00:21:00 | V3, V5, V6, V11 |
| Error Backpropagation Concept | 00:21:00-00:24:00 | V7, V8 |
| Sigmoid/Differentiable Functions | 00:24:10-00:25:00 | V9, V10 |
| Generalized Delta Rule | 00:26:23-00:27:29 | V13, V14, V15 |
| Continuous Learning Law | 00:27:53-00:28:45 | V16, V17 |
| Network Architecture (2-2-1) | 00:29:04-00:31:08 | V18, V19 |
| Forward Propagation | 00:32:00-00:33:30 | V20, V21 |
| Backpropagation Chain Rule | 00:34:35-00:39:05 | V22-V28 |
| Learning Rate / Exploding Gradients | 00:39:48-00:41:20 | V29, V30 |
| Loss Comparison (54.17 vs 4.005) | 00:41:54 | V31 |
| Convex Curve / GD Direction | 00:47:59-00:48:07 | V32, V33 |
| PyTorch Manual Forward Pass | 00:51:06-01:03:00 | V34-V41 |
| PyTorch Manual Backward Pass | 01:07:30-01:18:30 | V42-V45 |
| Debugging / Loss Verification | 01:22:00-01:26:28 | V46, V47 |
| PyTorch Autograd Introduction | 01:27:30-01:33:01 | V48-V53 |
| Zero Gradients / Computational Graph | 01:36:00-01:41:00 | V54-V56 |

---

### Summary
- **56 visuals** flagged (31 slides, 2 screen captures, 23 code demos)
- **6 frames** verified against Vimeo video
- Session covers backpropagation recap from theory to hands-on PyTorch implementation
- Two-part structure: Prasanna recaps backprop theory (hard learning problem, generalized delta rule, chain rule), then Saumya demonstrates coding in Colab
- Key numerical example: 2-2-1 network for LPA prediction with Loss: 54.17 --> 4.005 after one iteration
- Two Colab notebooks demonstrated: manual backprop (hand-coded gradients) and PyTorch Autograd (loss.backward())
- Important concepts: requires_grad=True, zero gradient initialization, computational graph tracking (AddBackward, ReluBackward)
- Q&A covers: learning rate vs weight updates, gradient descent direction on convex curve, weight initialization importance, activation functions per layer
- Preview of next topics: vanishing/exploding gradient problems, weight initialization techniques, PyTorch modules for automated forward pass and parameter updates
