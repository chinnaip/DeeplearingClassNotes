# Session 05 Visual Extraction Report
## Neural Network: Feedforward and Backpropagation

### Metadata
- **Source VTT**: `session05_transcript.vtt`
- **Vimeo URL**: https://vimeo.com/1142296251
- **Duration**: ~1h 37m (00:14:16 - 01:51:36)
- **Instructor(s)**: Prof. Mahadeva Prasanna (theory), Sunil Saumya (examples/demo)
- **Topics**: Perceptron recap, MLP architecture, hard learning problem, backpropagation algorithm, sigmoid/tanh activation, generalized delta rule, chain rule, numerical example (IQ/CGPA), forward/backward pass, weight updates, gradient clipping, feature scaling

---

### Visual Index

| # | Timestamp | Type | Description | Verified |
|---|-----------|------|-------------|----------|
| 1 | 00:16:12 | Slide | Screen share setup / title slide area | - |
| 2 | 00:17:02 | Slide | Recap: perceptron with step function as output | - |
| 3 | 00:18:09 | Slide | MLP layers: single-layer vs multi-layer perceptron | Yes |
| 4 | 00:18:29 | Slide | Title: "Neural Network: Feedforward and Backpropagation" | Yes |
| 5 | 00:20:01 | Slide | Introduction to backpropagation genesis | - |
| 6 | 00:21:17 | Diagram | 3-layer feedforward network from textbook (Fig 4.12) | Yes |
| 7 | 00:22:03 | Slide | Hidden layer target output problem explained | - |
| 8 | 00:24:00 | Slide | Hard learning problem definition | - |
| 9 | 00:27:28 | Diagram | Step function discontinuity illustration | - |
| 10 | 00:29:03 | Equation | Logistic sigmoid: f(x) = 1/(1+exp(-2Bx)), 0 to 1 | - |
| 11 | 00:30:10 | Graph | Sigmoid derivative f'(x) - non-zero differential region | - |
| 12 | 00:31:14 | Slide | Two Layer Perceptron with Fig 4.12 network diagram | Yes |
| 13 | 00:31:41 | Graph | Tanh function and its derivative, -1 to +1 range | - |
| 14 | 00:33:37 | Slide | Hard learning problem summary - desired output unknown for hidden | - |
| 15 | 00:34:05 | Slide | Solution: differentiable nonlinear output function | - |
| 16 | 00:35:00 | Slide | Backpropagation framework / generalized delta rule | - |
| 17 | 00:37:00 | Diagram | 3-layer notation: I (input), J (hidden), K (output) | - |
| 18 | 00:39:00 | Equation | Discrete perceptron learning law recall | - |
| 19 | 00:40:25 | Equation | Continuous perceptron / delta learning law with f'(x) | - |
| 20 | 00:42:15 | Slide | Delta Learning Law or Continuous Perceptron Law equations | Yes |
| 21 | 00:47:57 | Diagram | 4-layer feedforward network (input, 2 hidden, output) | - |
| 22 | 00:54:45 | Table | IQ/CGPA dataset: 4 rows, predict LPA | - |
| 23 | 00:56:12 | Diagram | Network architecture for example: 3-input, 2-hidden, 1-output | - |
| 24 | 00:58:14 | Diagram | Annotated Neural Network (2-2-1) for LPA Prediction | Yes |
| 25 | 00:59:45 | Slide | Forward pass: weighted sum + ReLU calculations | - |
| 26 | 01:00:40 | Equation | Loss = (y - y_hat)^2 = (3 - 10.36)^2 = 54.17 | - |
| 27 | 01:03:02 | Slide | Backprop notation: partial derivatives for all 9 params | - |
| 28 | 01:08:52 | Diagram | Dependency graph: L -> y_hat -> {v1,v2,h1,h2,b3} | - |
| 29 | 01:14:02 | Slide | Backprop: Compute dL/dv1, dL/dv2, dL/db3 with chain rule | Yes |
| 30 | 01:21:55 | Graph | ReLU derivative: 1 for positive, 0 for negative | - |
| 31 | 01:25:00 | Slide | Chain rule for hidden layer: L->y_hat->h1->z1->w11 | - |
| 32 | 01:29:55 | Slide | Backprop: Compute dL/dw12 with full chain rule path | Yes |
| 33 | 01:31:05 | Table | All gradient values substituted numerically | - |
| 34 | 01:33:15 | Table | Weight update results: old vs new parameter values | - |
| 35 | 01:34:35 | Slide | 2nd iteration: loss reduced from 54.17 to 4.006 | - |
| 36 | 01:37:00 | Slide | Feature scaling / gradient clipping discussion | - |

---

### Visual Reconstructions

#### V6/V12: Three-Layer Feedforward Neural Network (Fig 4.12)
```
  a_i    s_i    w_k^h    s_j^h    w_N    s_k^o
  [1]--->[i]===>[bias]==>[ j ]===>[ k ]--->
         ...     ...      ...     ...
  [I]    [I]    [J]      [J]     [K]
         Input   Hidden   Hidden  Output
         Layer   Layer    Layer   Layer
```
- Source: B. Yegnanarayan, New Delhi, 1999
- Shows fully connected 3-layer network with I input, J hidden, K output neurons

#### V24: Annotated Neural Network (2-2-1) for LPA Prediction
```
  [b1]---(bias)--->[h1]---v1--->
  [x1]--w11-->[h1]              [y_hat]
  [x1]--w12-->[h2]---v2--->
  [x2]--w21-->[h1]
  [x2]--w22-->[h2]
  [b2]---(bias)--->[h2]
  [b3]---(bias)---------->[y_hat]

  h1 = ReLU(w11*x1 + w12*x2 + b1)
  h2 = ReLU(w21*x1 + w22*x2 + b2)
  y_hat = v1*h1 + v2*h2 + b3
```

#### V29: Backpropagation Chain Rule - Output Layer
```
  Loss L = (y - y_hat)^2
  dL/dy_hat = 2(y_hat - y)

  Dependency: L --> y_hat --> {v1, v2, h1, h2, b3}

  dL/dv1 = dL/dy_hat * dy_hat/dv1 = 2(y_hat - y) * h1
  dL/dv2 = dL/dy_hat * dy_hat/dv2 = 2(y_hat - y) * h2
  dL/db3 = dL/dy_hat * dy_hat/db3 = 2(y_hat - y) * 1
```

#### V32: Backpropagation Chain Rule - Hidden Layer
```
  Path: L --> y_hat --> h1 --> z1 --> {w11, w12, b1}

  dL/dw12 = dL/dy_hat * dy_hat/dh1 * dh1/dz1 * dz1/dw12
          = 2(y_hat-y) * v1 * ReLU'(z1) * x2

  ReLU'(z) = 1 if z > 0, 0 if z < 0
```

#### V34: Weight Update Results (1st iteration)
```
  Learning rate: eta = 1e-4
  Parameter  | Old    | New
  -----------|--------|--------
  v1         | 1.2    | 1.19
  v2         | 0.8    | 0.79
  b3         | 1.0    | 0.99
  w11        | 0.02   | -0.12
  w12        | 0.5    | 0.48
  b1         | 1.0    | 0.99
  w21        | 0.04   | -0.05
  w22        | -0.3   | -0.30
  b2         | 1.0    | 0.99

  Loss: 54.17 --> 4.006 (after 1 iteration)
```

---

### Cross-Reference Matrix

| Topic | VTT Cues | Visual IDs | Key Concept |
|-------|----------|------------|-------------|
| Perceptron Recap | 00:16:53-00:17:15 | V2, V3 | Step function, linearly separable |
| MLP Architecture | 00:17:17-00:20:00 | V3, V6 | Convex/non-convex, 3-4 layer networks |
| Hard Learning Problem | 00:21:17-00:26:00 | V7, V8, V14 | No target output at hidden layer |
| Sigmoid/Tanh Activation | 00:27:28-00:32:00 | V9-V13 | Differentiable output functions |
| Generalized Delta Rule | 00:33:37-00:36:00 | V14-V16 | Error backpropagation via derivatives |
| Weight Notation (I,J,K) | 00:37:00-00:39:00 | V17, V18 | WKJ, WHJ notation |
| Delta Learning Law | 00:39:00-00:42:15 | V18-V20 | Discrete vs continuous perceptron |
| Feedforward Example | 00:47:57-00:58:14 | V21-V24 | IQ/CGPA 2-2-1 network |
| Forward Pass Calc | 00:58:14-01:01:00 | V25, V26 | Z, ReLU, y_hat=10.36, loss=54.17 |
| Backprop Output Layer | 01:03:02-01:16:00 | V27-V29 | Chain rule for v1, v2, b3 |
| Backprop Hidden Layer | 01:16:00-01:30:00 | V30-V32 | Chain rule through ReLU for w11,w12 |
| Weight Updates | 01:30:00-01:35:00 | V33-V35 | 9 params updated, loss 54->4 |
| Feature Scaling | 01:35:00-01:40:00 | V36 | Large gradients, gradient clipping |

---

### Summary
- **36 visuals** flagged from VTT analysis
- **8 frames** verified against Vimeo video
- Session covers backpropagation theory from first principles through a complete numerical example
- Key progression: Hard problem -> Hard learning problem -> Sigmoid activation -> Chain rule -> Weight updates
- Practical insights: feature scaling importance, learning rate selection, gradient clipping
- Demo in PyTorch deferred to next session
