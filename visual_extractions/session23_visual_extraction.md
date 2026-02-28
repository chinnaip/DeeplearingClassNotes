# Session 23: Visual Extraction Notes
## Video: https://vimeo.com/1163336970

---

## Topics Covered
1. Bidirectional Sequence Models (Bidirectional RNN/LSTM/GRU)
2. Deep RNN Architectures
3. Sequence-to-Sequence (Seq2Seq) Model Recap
4. Seq2Seq Code Walkthrough (Machine Translation: English to Hindi)

---

## 1. Bidirectional Sequence Models

### Slide: Bi-Directional Sequence Models (~20:03)
**Key Points from Slide:**
- Sequence modeling involves deriving knowledge from the sequence for prediction or classification
- Conventional RNN, LSTM and GRU are designed to look only at the sequence from the past (left to right)
- Many applications exist where the future part of the sequence needs to be looked into, or both directions
- Mahabharata example used: "Aswathhama hathaha (and then he murmured) iti Narova kunjaraha" - Drona heard only the first part (left-to-right) and concluded his son was dead; had he heard the full sentence (bidirectional), he would have known it referred to an elephant

### Slide: Bi-Directional RNN Architecture (~16:30)
**Visual Description:**
- Two rows of RNN cells shown:
  - **Bottom row (Forward/Left-to-Right):** Cells labeled A, connected S_0 -> S_1 -> ... -> S_i
  - **Top row (Backward/Right-to-Left):** Cells labeled A', connected S'_i -> ... -> S'_1
- Inputs X_0, X_1, X_2, ..., X_i feed into both rows from below
- Outputs y_0, y_1, y_2, ..., y_i are produced by combining both forward and backward hidden states
- At each position, the output uses knowledge from both left-to-right and right-to-left directions

### Slide: Bidirectional RNNs Equations (~23:50)
**Equations:**
- **Forward Direction:** a(t) = g(W_a[a(t-1), x(t)] + b_a)
- **Backward Direction:** b(t) = g(W_b[b(t+1), x(t)] + b_a)
- **Output:** y_hat(t) = g(W_y[a(t), b(t)] + b_y)

**Explanation:**
- a(t) is the forward hidden state computed using previous state a(t-1) and current input x(t)
- b(t) is the backward hidden state computed using future state b(t+1) and current input x(t)
- The output combines both forward and backward states through weight matrix W_y
- Unidirectional forward = only a(t); Unidirectional backward = only b(t); Bidirectional = combination

### Practical Examples Discussed:
- **Named Entity Recognition (NER):** "Ashoka is a great warrior" vs "Ashoka is a good hotel" - the word after "Ashoka" determines if it refers to a person or place
- **Word Sense Disambiguation:** "I went to the bank to deposit money" - "bank" is ambiguous until future context ("deposit money") clarifies it means financial bank, not river bank
- **Where bidirectional is NOT suitable:** Language modeling / next-word prediction tasks where future context is not available (e.g., autoregressive generation). Speech processing where signal evolves strictly left-to-right.
- **BERT vs LLMs:** BERT uses bidirectional encoder (good for fill-in-the-blank / masked language modeling); LLMs use unidirectional decoder (good for next-word prediction)

### Vanishing Gradient in Bidirectional RNNs:
- The vanishing gradient problem still persists in bidirectional RNNs
- It depends on sequence length, not on the direction of processing
- LSTM and GRU help mitigate this in both directions

---

## 2. Deep RNN Architectures

### Slide: Deep RNN (~28:39)
**Visual Description:**
- Multi-layer architecture with 3 stacked layers: a[1], a[2], a[3]
- Inputs x<1>, x<2>, x<3>, x<4> feed into the first layer
- Each layer's output feeds as input to the layer above
- Outputs y<1>, y<2>, y<3>, y<4> come from the topmost layer
- Handwritten equation shown: a^(2)<3> = g(W_a^(2)[a^(1)<3>, a^(2)<2>] + b_a^(2))

**Key Points:**
- Each bidirectional RNN layer can be treated as one layer of a deep architecture
- Stack multiple layers to get deeper representations
- Layer i cell state depends on: output from layer (i-1) at same timestep AND cell state from layer i at previous timestep
- Shallow = 1-2 layers; Deep = many layers
- Same code for unidirectional can be extended: left-to-right -> add right-to-left -> combine for bidirectional -> stack for deep

### Q&A on Deep Architecture:
- Q: Does the output of the entire first layer act as input to the second layer?
- A: No, at each timestep, each node's output acts as input to the corresponding node in the next layer (not the whole sequence output)

---

## 3. Sequence-to-Sequence (Seq2Seq) Model Recap

### Slide: Seq2Seq Encoder-Decoder Architecture (~41:42)
**Key Idea:**
- Encoder reads entire input sequence
- Compresses it into a context vector
- Decoder generates output sequence step-by-step
- Used in: Machine Translation, Summarization, Question Answering

**Visual Description:**
- Encoder: Chain of LSTM cells processing English input ("I", "am", "Going")
- Arrow from encoder to Context Vector (fixed-size representation)
- Decoder: Chain of LSTM cells generating Hindi output, starting with <Start> token, producing words one by one
- Hindi output sequence: Start -> me -> ja -> raha -> hu -> END

### Training with Teacher Forcing:
- At training time, feed the correct target word at each decoder timestep (not the model's own prediction)
- This helps the model converge faster
- Teacher forcing is optional - can apply always, or with some probability (e.g., 0.5)
- At each timestep, create self-supervised learning pairs from the target corpus
- Loss is calculated at each decoder timestep and summed for backpropagation

### Inference (Autoregressive Decoding):
- At test time, no parallel corpus available
- Feed the model's own prediction as input to the next timestep
- Works in an autoregressive / next-word prediction manner

### Slide: Decoder Training Loss: Categorical Cross-Entropy (~48:14)
**Visual Description:**
- Shows full encoder-decoder architecture with LSTM cells
- Encoder processes English input
- Decoder generates Hindi words with softmax output at each step
- Categorical Cross-Entropy Loss computed: -sum(y_true * log(y_pred)) for i=1 to N
- One-hot encoded target vectors shown for each Hindi word
- Loss accumulated across all decoder timesteps

### Slide: Seq2Seq Limitation with Long-Range Dependencies (~53:10)
**Encoder-side Problem:**
- Long input sentence must be summarized into one vector
- Early words and late words compete for representation
- Important details may be forgotten

**Decoder-side Problem:**
- Same context vector is used at every decoding step
- Decoder cannot focus on relevant source words
- Leads to incorrect or generic translations

**Key Insight:** To translate "me" (I), the decoder only needs to know "I" from the encoder, not the entire sentence. But the context vector carries everything, causing information bottleneck.

### Enhancements Discussed:
1. **Embedding Layer:** Transform sparse one-hot vectors to dense low-dimensional representations before feeding to LSTM
2. **Deep/Stacked LSTM:** Multiple LSTM layers to capture more complex patterns
3. **Attention Mechanism:** (Preview for next class) - Allows decoder to focus on relevant encoder positions at each timestep

---

## 4. Seq2Seq Code Walkthrough (MTSeq2Seq Model.ipynb)

### Dataset: English-Hindi Machine Translation
- Source: Kaggle English-to-Hindi MT dataset
- ~177,000 sentence pairs; sampled 10,000 for demo
- Two columns: English sentence, Hindi sentence

### Pre-processing Pipeline:
1. **Data Cleaning:** Drop rows with missing values, reset index
2. **Sampling:** Random sample of 10,000 rows
3. **Type Conversion:** Convert to string then list format
4. **Tokenization:** Lowercase, strip, split on whitespace
5. **Vocabulary Building:** Separate vocabularies for English (22K words) and Hindi (24.5K words)
   - Special tokens: PAD (0), UNK (1), SOS (2), EOS (3)
6. **Encoding:** Convert tokens to index sequences
   - Hindi sequences get SOS prepended and EOS appended
7. **Padding:** Pad/truncate all sequences to fixed max length (20 for demo)
   - Actual max: English=170, Hindi=280 (too large for demo)
8. **Tensor Conversion:** Convert to PyTorch tensors, shape (10000, 20)

### Model Architecture:

**Encoder:**
- nn.Embedding(english_vocab_size, embed_dim)
- nn.LSTM(embed_dim, hidden_dim, batch_first=True)
- Returns: final hidden state (h_n) and cell state (c_n) as context vector

**Decoder:**
- nn.Embedding(hindi_vocab_size, embed_dim)
- nn.LSTM(embed_dim, hidden_dim, batch_first=True)
- nn.Linear(hidden_dim, hindi_vocab_size) -- fully connected output layer
- Returns: predictions at each timestep + updated hidden/cell states

**Seq2Seq Wrapper:**
- Combines encoder and decoder
- Forward pass: encode input -> get context vector -> loop through decoder timesteps
- Teacher forcing applied with configurable ratio (0.5)
- Gradient clipping used for exploding gradient prevention

### Training Configuration:
- Embed dimension: 128
- Hidden dimension: 128
- Batch size: 8 (small to avoid OOM errors)
- Epochs: 50 (demo); recommended 500-1000 for good results
- Loss: CrossEntropyLoss
- Optimizer: Adam
- Training time: ~35-40 min for 50 epochs with demo settings
- Initial loss: ~9300, reduced to ~700 after 50 epochs

### OOM (Out of Memory) Issue Explained:
- At each decoder timestep, storing output tensors for batch_size x vocab_size
- With batch_size=32 and large vocab, memory fills up quickly
- Solution: Reduce batch size (8), reduce sequence length (20), reduce hidden dim

### BLEU Score Evaluation:
- BLEU score ranges 0 to 1 (higher is better)
- Measures similarity between predicted and reference translations
- Score of 0.2-0.3 considered good for seq2seq without attention
- Demo achieved ~0.578 but translations were still incorrect
- BLEU limitations: does not account for valid alternative translations
- Other metrics: BERTScore for semantic similarity

### Inference Results (Demo):
- "How are you" -> incorrect translation (model undertrained)
- Need 500-1000 epochs for reasonable translations
- Full dataset (177K) requires better GPU resources

---

## Key Takeaways
1. Bidirectional RNNs capture context from both past and future, essential for tasks like NER and fill-in-the-blank
2. Deep RNNs stack multiple layers for richer representations
3. Seq2Seq encoder-decoder architecture handles variable-length input/output sequences
4. Teacher forcing accelerates training convergence
5. Context vector bottleneck is the main limitation -> motivates attention mechanism (next session)
6. Practical considerations: batch size, sequence length, and hidden dimensions affect memory and training time

---

## Next Session Preview
- Attention mechanism to address the context vector bottleneck
- Same seq2seq code with attention layer added
- Transformer architecture introduction in the following course (LLM course)
