# Session 04 - Visual Extraction Report
## DL LS 4 (Deep Learning Live Session 4)

### Metadata
- **Source**: https://vimeo.com/1141589945
- **Transcript**: session04_transcript.vtt
- **Duration**: ~1:20:55
- **Instructors**: Prof. SRM Prasanna, Dr. Sunil Saumya
- **Topics**: Perceptron review, decision boundary, linearly separable vs inseparable classes, XOR problem, multi-layer perceptron (MLP), activation functions, TensorFlow Playground visualization, PyTorch XOR demo

---

## Visual Index

| # | Timestamp | Type | Description |
|---|-----------|------|-------------|
| 1 | 00:21:17 | Slide | Perceptron recap - building block of neural network |
| 2 | 00:22:44 | Slide | Perceptron architecture: weighted sum + bias + activation |
| 3 | 00:23:55 | Slide | Decision boundary equation: w0x0 + w1x1 + w2x2 = 0 |
| 4 | 00:24:45 | Slide | Decision boundary plot - solving for x2 |
| 5 | 00:25:30 | Slide | Perceptron Decision Boundary - calculation and plot with equation and graph |
| 6 | 00:25:55 | Slide | Perceptron trick algorithm - random weight initialization and iterative update |
| 7 | 00:27:02 | Slide | Activation function discussion - step function default, sigmoid for probability |
| 8 | 00:29:15 | Slide | Linearly separable vs inseparable classes introduction |
| 9 | 00:30:00 | Slide | OR gate truth table - linearly separable example |
| 10 | 00:31:11 | Slide | XOR gate truth table - linearly inseparable example |
| 11 | 00:33:00 | Slide | Multi-layer perceptron concept - adding hidden layers for nonlinear boundaries |
| 12 | 00:34:00 | Slide | Binary linearly separable classes - 2-class classification with perceptron |
| 13 | 00:40:48 | Slide | Binary Linearly Separable Classes by Perceptron - Figure 4.3 decision boundaries at m=0,10,20,30,40 |
| 14 | 00:43:00 | Slide | Multi-class linearly separable - 3 classes with 3 perceptrons in output layer |
| 15 | 00:48:00 | Slide | Binary linearly inseparable classes - decision boundary wandering |
| 16 | 00:50:00 | Slide | AND gate as linearly separable, XOR as linearly inseparable |
| 17 | 00:52:00 | Slide | Two-layer vs three-layer vs four-layer network comparison figure |
| 18 | 00:56:13 | Slide | Linearly Separable vs Linearly Inseparable - Perceptrons: Figure 4.10 with 2/3/4-layer networks |
| 19 | 00:59:00 | Slide | Hard problems definition - not representable by single layer perceptron |
| 20 | 01:02:00 | Slide | Summary: single layer to MLP evolution |
| 21 | 01:05:00 | Slide | Hyperplane A vs B - no maximum margin in perceptron |
| 22 | 01:07:00 | Slide | Hinge-style loss function for perceptron |
| 23 | 01:10:00 | Slide | Sigmoid activation function - output between 0 and 1 |
| 24 | 01:11:00 | Slide | Tanh activation function - output between -1 and +1 |
| 25 | 01:11:45 | Slide | ReLU activation function - max(z,0) |
| 26 | 01:13:01 | Slide | Problem with Perceptron - AND/OR/XOR truth tables with perceptron diagrams |
| 27 | 01:14:00 | Demo | TensorFlow Playground - XOR data with single perceptron fails |
| 28 | 01:17:00 | Demo | TensorFlow Playground - adding hidden layer neurons solves XOR |
| 29 | 01:22:00 | Code | PyTorch perceptron code - AND/OR/XOR data creation and plotting |
| 30 | 01:24:00 | Code | Train perceptron function - predict, error, weight update loop |
| 31 | 01:26:00 | Slide | Decision boundary plots - AND (separable), OR (separable), XOR (fails) |
| 32 | 01:26:49 | Slide | The Solution: Multilayer Perceptron Network - XOR with hidden layer, ReLU, transformed input space |
| 33 | 01:34:00 | Slide | MLP architectures for different tasks - binary, multi-class, regression |
| 34 | 01:36:00 | Slide | Network flexibility - neurons, layers, activation functions |

---

## Key Visual Reconstructions

### V1: Perceptron Architecture (00:22:44)
```
Inputs        Perceptron       Output
x1 --w1-->\
            [Sum + Bias] --> f(z) --> y
x2 --w2-->/
     +1 --w0--> (bias)

z = sum(wi*xi) + b
y = f(z)  where f = step function (default)
```

### V2: Decision Boundary Equation (00:25:30)
```
Decision boundary equation:
  w0*x0 + w1*x1 + w2*x2 = 0

Substitute w's:
  -1 + 0.5*x1 - 0.3*x2 = 0

Solve for x2:
  x2 = (1 - 0.5*x1) / (-0.3)

Plot: IQ/10 (x2) vs x1 with decision line
  - class 0 and class 1 points separated
```

### V3: Perceptron Training Iterations (00:40:48)
```
Figure 4.3: Binary Linearly Separable Classes

  x2=a2 ^    m=0  m=10  m=40
         |   /     /    |
    O O  | /    /     |
    O    |/   /      |  <- decision boundaries
   ------/--/--------|--> x1=a1
    X X /  /    m=30
    X  / /     m=20
       //

At m=40: line separates X class from O class
Convergence achieved (no unique solution - infinite lines possible)
```

### V4: Network Layer Capabilities (00:56:13)
```
Figure 4.10: Linearly Separable vs Inseparable - Perceptrons

| Structure       | Decision Region  | XOR | Convex | Non-convex |
|-----------------|------------------|-----|--------|------------|
| 2-layer (SLP)   | Half plane       | No  | No     | No         |
| 3-layer (1 HL)  | Convex regions   | Yes | Yes    | No         |
| 4-layer (2 HL)  | Arbitrary shapes | Yes | Yes    | Yes        |

- 2-layer: linear hyperplanes only
- 3-layer: intersection of linear -> convex surfaces
- 4-layer: intersection of convex -> any non-convex surface
```

### V5: AND/OR/XOR with Perceptron (01:13:01)
```
AND Truth Table:        Logical AND:
x1 x2 | y              x1 --1-->\
0  0  | 0              x2 --1--> (o) --> y
0  1  | 0              +1 ---1->/
1  0  | 0
1  1  | 1              Linearly separable: YES

OR Truth Table:         Logical OR:
x1 x2 | y              x1 --1-->\
0  0  | 0              x2 --1--> (o) --> y
0  1  | 1              +1 --0->/
1  0  | 1
1  1  | 1              Linearly separable: YES

XOR Truth Table:
x1 x2 | y
0  0  | 0              Linearly separable: NO
0  1  | 1              -> Single perceptron FAILS
1  0  | 1
1  1  | 0
```

### V6: MLP Solution for XOR (01:26:49)
```
Network: Input -> Hidden(ReLU) -> Output(Step)

  x1 ---(-1)---> [h1] ---(-2)-->\
       \(1)/          \          [y1] --> output
  x2 ---(-1)---> [h2] ---(+1)-->/
       bias=+1        bias=+1

Transformed space:
  x=[0,0]: h=ReLU[0,-1]=[0,0],  y=0
  x=[0,1]: h=ReLU[1,0]=[1,0],   y=1
  x=[1,0]: h=ReLU[1,0]=[1,0],   y=1
  x=[1,1]: h=ReLU[2,1]=[2,1],   y=0

In h-space: points become linearly separable!
```

### V7: Activation Functions (01:10:00 - 01:11:45)
```
1. Step Function (default):    f(z) = 1 if z>0, else 0
2. Sigmoid:  f(z) = 1/(1+e^(-z))     output: [0, 1]
3. Tanh:     f(z) = (e^z - e^(-z))/(e^z + e^(-z))  output: [-1, +1]
4. ReLU:     f(z) = max(z, 0)         output: [0, inf)
```

### V8: MLP Architecture Types (01:34:00)
```
1. Binary logistic regression: N inputs -> 1 output (sigmoid)
2. Multinomial logistic: N inputs -> K outputs (softmax)
3. 2-layer binary: N inputs -> H hidden -> 1 output
4. 2-layer multi-class: N inputs -> H hidden -> K outputs (softmax)
5. Regression: N inputs -> H hidden -> 1 output (linear activation)
```

---

## Cross-Reference Matrix

| Visual | Concept | Instructor | Method |
|--------|---------|------------|--------|
| V1-V2 | Perceptron basics | Dr. Sunil | Slide + equation |
| V3 | Training convergence | Prof. Prasanna | Figure 4.3 |
| V4 | Layer capabilities | Prof. Prasanna | Figure 4.10 |
| V5 | Logic gates | Dr. Sunil | Truth tables + diagrams |
| V6 | XOR MLP solution | Dr. Sunil | Code demo + slide |
| V7 | Activation functions | Dr. Sunil | Slide |
| V8 | MLP architectures | Dr. Sunil | Slide |

---

## Summary
Session 4 transitions from single perceptron to multi-layer perceptron (MLP). Key progression: perceptron review -> decision boundary -> linearly separable classes work with SLP -> XOR/inseparable classes fail -> adding hidden layers creates nonlinear boundaries -> MLP with 2+ perceptron layers handles convex and non-convex regions. Demonstrated via TensorFlow Playground visualization and PyTorch XOR code. Covered activation functions (step, sigmoid, tanh, ReLU) and MLP flexibility for various task types.
