# Session 07 – Visual Extraction Report

## Metadata
| Field | Value |
|---|---|
| Session | 07 |
| Title | ANN: Advanced Concepts |
| Date | December 8, 2025 |
| Vimeo URL | https://vimeo.com/1144571367 |
| Duration | 1:51:32 |
| Instructor | Sunil Saumya & S R M Prasanna |
| Institution | Dept. of Data Science & AI, IIIT Dharwad |
| VTT File | session07_transcript.vtt |

## Visual Index

| # | Timestamp | Type | Description |
|---|---|---|---|
| 1 | 00:13:40 | Screen share | Instructor shares screen to begin presentation |
| 2 | 00:14:04 | Slide | Session plan overview – topics to cover in Session 07 |
| 3 | 00:18:28 | Title slide | "ANN: Advanced Concepts" – Sunil Saumya & S R M Prasanna, IIIT Dharwad, Dec 8, 2025 |
| 4 | 00:19:03 | Slide | Agenda slide – backprop recap, SGD/batch/mini-batch, vanishing gradient problem |
| 5 | 00:22:51 | Slide | Recap – network architecture: 2 inputs, 2 hidden neurons, 1 output |
| 6 | 00:23:02 | Diagram | Neural network diagram with feed forward propagation, ReLU activation, squared error loss |
| 7 | 00:23:47 | Slide | Back propagation recap – chain rule for gradient calculation across layers |
| 8 | 00:24:30 | Diagram | Network with gradient arrows showing partial derivatives dL/dW for each parameter |
| 9 | 00:25:25 | Slide | Parameter update rule: theta = theta - lr * gradient, 9 parameters updated |
| 10 | 00:26:06 | Table | Before/after update comparison – loss dropped from 54.17 to 4.006 |
| 11 | 00:27:03 | Slide | "Why Back Propagation Works" – intuition behind parameter updates |
| 12 | 00:27:50 | Equation | Loss as function of all trainable parameters: L = (y - y_hat)^2 |
| 13 | 00:28:00 | Equation | y_hat = v1*h1 + v2*h2 + b3, expanded nested function |
| 14 | 00:29:00 | Slide | Unfolding y_hat – showing all 9 parameters in nested computation |
| 15 | 00:30:06 | Equation | Full expansion: y_hat with ReLU(w11*x1+w12*x2+b1) substituted |
| 16 | 00:33:03 | Plot | 2D loss curve: Loss vs W11 parameter, showing gradient as slope |
| 17 | 00:34:14 | Slide | "Loss as a Function of All Trainable Parameters" with full nested equations |
| 18 | 00:36:06 | Slide | Concept of gradient – gradient > 0, gradient < 0, gradient = 0 cases |
| 19 | 00:38:18 | Plot | Convex loss curve with gradient descent direction arrows – both sides converge to minimum |
| 20 | 00:41:00 | Equation | Update rule sign analysis: positive gradient → decrease theta, negative → increase theta |
| 21 | 00:42:01 | Slide | "Opposite direction of gradient" – key takeaway of gradient descent |
| 22 | 00:44:09 | Plot | 3D surface plot: Loss vs theta_1 vs theta_2, showing global minimum |
| 23 | 00:46:23 | Slide | Learning rate impact – 3 cases: too small, moderate, too large |
| 24 | 00:47:00 | Plot | Small learning rate: slow convergence, tiny steps toward minimum |
| 25 | 00:47:42 | Plot | Large learning rate: overshooting, jumping across minimum |
| 26 | 00:48:06 | Plot | Moderate learning rate: balanced convergence toward minimum |
| 27 | 00:48:55 | Slide | Exploding gradient problem teaser – large LR causes divergence |
| 28 | 00:50:13 | Slide | Convergence definition – parameter updates become negligible |
| 29 | 00:50:55 | Plot | Loss vs epochs curve showing convergence plateau |
| 30 | 00:52:00 | Slide | Convergence summary: loss no longer decreasing, gradients near zero |
| 31 | 00:54:52 | Q&A | Discussion on convex vs concave functions, loss minimization vs accuracy maximization |
| 32 | 00:58:07 | Slide | "Recap - Feedforward NN" with full network diagram and forward pass calculations |
| 33 | 00:58:10 | Diagram | Neural network with weights, biases, forward pass numerical example (x1=80, x2=8) |
| 34 | 01:04:34 | Slide | Gradient descent types introduction: Batch, Stochastic, Mini-batch |
| 35 | 01:05:02 | Diagram | Batch GD: all 6 samples processed, cumulative loss, single parameter update |
| 36 | 01:07:37 | Diagram | SGD: each sample triggers parameter update individually |
| 37 | 01:08:45 | Diagram | Mini-batch: groups of 2 samples processed, update after each mini-batch |
| 38 | 01:09:06 | Table | Comparison table: Batch vs SGD vs Mini-batch – time, convergence, vectorization, GPU |
| 39 | 01:10:30 | Slide | Batch gradient descent details – stable updates, expensive for large datasets |
| 40 | 01:11:05 | Equation | Mean squared loss over all N samples for batch GD |
| 41 | 01:13:57 | Slide | "Stochastic Gradient Descent (SGD)" – fast updates, high variance, zigzag convergence plot |
| 42 | 01:15:00 | Plot | SGD zigzag convergence path toward global minimum |
| 43 | 01:16:30 | Slide | Mini-batch gradient descent – trade-off between speed and stability |
| 44 | 01:17:06 | Table | Summary: BGD processes N, SGD processes 1, mini-batch processes M samples |
| 45 | 01:18:03 | Code | Colab notebook: import statements and random dataset generation (500 samples) |
| 46 | 01:19:22 | Code | IQ (50-130) and CGPA (5-10) feature generation, X concatenation, Y = 0.1*IQ + 0.5*CGPA + noise |
| 47 | 01:20:02 | Code | Parameter initialization: W11, W12, W21, W22, b1, b2, b3, v1, v2 with requires_grad=True |
| 48 | 01:21:00 | Code | Forward pass function: Z1, H1, Z2, H2, Y_hat computation |
| 49 | 01:22:14 | Code | Update parameters function: theta -= lr * grad, zero_grad after update |
| 50 | 01:23:00 | Code | train_full_batch function: 200 epochs, MSE loss, backprop, loss history |
| 51 | 01:24:30 | Code | train_sgd function: per-sample update, 200 epochs x 500 samples |
| 52 | 01:25:00 | Code | train_minibatch function: batch_size=50, balanced approach |
| 53 | 01:27:00 | Code | Timing results: Full batch 0.20s, SGD 63s, Mini-batch 1.38s |
| 54 | 01:29:02 | Plot | Loss convergence comparison: SGD (orange, fast ~25 epochs), Batch (blue, ~65 epochs), Mini-batch (green, still decreasing at 200) |
| 55 | 01:30:06 | Slide | Mini-batch best practices: batch size in powers of 2 (8, 16, 32) for RAM optimization |
| 56 | 01:37:06 | Slide | Vanishing gradient problem introduction |
| 57 | 01:38:01 | Slide | Chain rule multiplication: small gradients (0.1) across layers shrink exponentially |
| 58 | 01:39:00 | Example | 10 layers with gradient 0.1 each: 0.1^10 = 1e-10, effectively zero |
| 59 | 01:39:50 | Diagram | Deep network with sigmoid activations showing gradient vanishing toward input layers |
| 60 | 01:41:30 | Slide | Solutions overview: reduce complexity, use ReLU, proper weight init, batch normalization, residual connections |

## Key Visual Reconstructions

### R1: Neural Network Architecture (00:23:02)
- 2 input nodes (x1, x2), 2 hidden neurons (h1, h2) with ReLU, 1 output (y_hat)
- Weights: W11, W12, W21, W22 (input-hidden), V1, V2 (hidden-output)
- Biases: b1, b2 (hidden), b3 (output)
- Forward: Z = W*X + b, H = ReLU(Z), y_hat = V1*H1 + V2*H2 + b3
- Loss: L = (y - y_hat)^2

### R2: Gradient Descent on Convex Curve (00:38:18)
- U-shaped convex loss curve with parameter theta on x-axis, loss on y-axis
- Gradient > 0 (right side): slope upward, decrease theta to move toward minimum
- Gradient < 0 (left side): slope downward, increase theta to move toward minimum
- Gradient = 0 (bottom): minimum reached, training converged
- Update rule: theta_new = theta - lr * dL/dtheta works for both sides

### R3: Learning Rate Effects (00:46:23)
- Small LR (e.g. 1e-6): very slow convergence, tiny steps, many epochs needed
- Large LR (e.g. 1.0): overshooting, jumping across minimum, possible divergence
- Moderate LR (e.g. 0.01-0.1): balanced convergence, recommended as default
- LR is a hyperparameter, tunable during training
- Example: gradient=97, LR=1 would change param from 1.2 to -95.9 (catastrophic)

### R4: Batch vs SGD vs Mini-batch Comparison (01:09:06)
- Batch GD: process all N samples, compute cumulative loss, update once per epoch
- SGD: process 1 sample, update after each; N updates per epoch
- Mini-batch: process M samples (e.g. 50), update after each batch; N/M updates per epoch
- Time: Batch fastest (0.20s), Mini-batch moderate (1.38s), SGD slowest (63s)
- Convergence: SGD fastest (~25 epochs), Batch moderate (~65), Mini-batch slowest (>200)
- Recommended: Mini-batch with batch_size as power of 2 (8, 16, 32)

### R5: Vanishing Gradient Problem (01:37:06)
- During backprop, gradients multiply through chain rule across layers
- If each gradient < 1 (e.g. sigmoid outputs 0.1), product shrinks exponentially
- 10 layers x 0.1 gradient = 0.1^10 = 1e-10 (effectively zero)
- Zero gradient means theta_new = theta (no parameter update)
- Early layers stop learning; network fails to train deeper parts
- Sigmoid activation is primary cause (squashes to 0-1 range)

## Cross-Reference Matrix

| Topic | Slides | Diagrams | Equations | Code | Plots |
|---|---|---|---|---|---|
| Backprop recap | 5,7,11 | 6,8 | 9,10 | - | - |
| Loss as function of params | 12,14,17 | - | 13,15 | - | 16 |
| Gradient concept | 18,21 | - | 20 | - | 19 |
| 3D optimization | - | - | - | - | 22 |
| Learning rate | 23,27 | - | - | - | 24,25,26 |
| Convergence | 28,30 | - | - | - | 29 |
| GD types theory | 34,39,41,43,44,55 | 35,36,37 | 40 | - | 42 |
| GD types code demo | - | - | - | 45-53 | 54 |
| Vanishing gradient | 56,57,60 | 59 | 58 | - | - |
| Feedforward recap | 32 | 33 | - | - | - |

## Summary
- **Session focus**: Advanced ANN concepts – backprop intuition, gradient descent variants, vanishing gradients
- **Total visuals extracted**: 60
- **Key topics**: Why backprop works (loss depends on all params via nested functions), gradient sign determines update direction, learning rate controls step size, 3 types of gradient descent (Batch/SGD/Mini-batch) with PyTorch demo, vanishing gradient problem in deep networks with sigmoid
- **Code demonstrated**: Colab notebook comparing Batch/SGD/Mini-batch GD on synthetic IQ-CGPA-LPA dataset (500 samples, 200 epochs)
- **Quiz announced**: Next Monday at 8:00 PM, MCQ format, 15-20 questions, 30 minutes, covering all topics through Session 07 including PyTorch
- **Next session preview**: Vanishing gradient solutions (ReLU, weight init, batch norm, residual connections), PyTorch NN module, optimizer module
