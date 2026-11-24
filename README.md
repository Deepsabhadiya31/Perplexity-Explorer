# Perplexity Explorer (Surprisal)

An interactive, browser-based educational web application that demystifies how causal language models predict tokens and compute sequence perplexity. Built with **Transformers.js** and **DistilGPT-2**, running 100% locally in your browser with zero backend requirements or API keys.

---

## 🚀 Overview

Language models do not perceive sentences as holistic thoughts—they process text sequentially as tokens, estimating probability distributions for each subsequent step based on preceding context. **Perplexity** measures how "surprised" a model is by a sequence of text.

**Perplexity Explorer** provides an intuitive, hands-on playground to:
1. Deconstruct sentences into subword tokens.
2. Inspect exact model logits and numerically stable softmax probabilities.
3. Trace step-by-step mathematical calculations from reciprocal probabilities to final perplexity.
4. Explore top-8 next-token predictions and iteratively build sentences.
5. Manually simulate probability distributions in an offline playground.

---

## ✨ Features

### 1. 🧪 Sentence Lab (Real Browser-Based Inference)
- **Local Model Execution**: Runs `Xenova/distilgpt2` (q8 quantization) directly in the browser via ONNX Runtime Web and WebAssembly.
- **Interactive Experiments**:
  - Sample sentence buttons (*"I love Python"*, *"Python love I"*, *"The capital of France is Paris"*).
  - **Surprise Me**: Generates diverse linguistic examples.
  - **Shuffle & Reverse Words**: Demonstrates how word order dramatically shifts perplexity even when vocabulary remains identical.
- **History Comparison**: Visual bar comparison across multiple sentence runs to track relative surprise.

### 2. 📊 Token Probability Analysis
- **Token Breakdown Table**: Displays token positions, decoded subwords (with visible space markers `␣`), individual probabilities $P_i$, and reciprocal values $1/P_i$.
- **Step-by-Step Mathematical Transparency**:
  - **Step 1**: Multiplies reciprocal probabilities $\prod_{i=1}^N \frac{1}{P_i}$.
  - **Step 2**: Evaluates total sequence length $N$ (excluding the initial context marker).
  - **Step 3**: Computes geometric mean perplexity using numerically stable logarithms:
    $$\text{Perplexity} = \exp\left(\frac{1}{N} \sum_{i=1}^N -\ln(P_i)\right)$$
- **Interpretive Gauge**: Color-coded scale from *Very Predictable* ($\text{PPL} < 5$) to *Very Surprising* ($\text{PPL} > 200$).
- **Surprise Distribution**: Relative surprise bar chart identifying the biggest surprise token and equivalent discrete choices (dice visualization).

### 3. 🔮 Next-Token Prediction
- Computes the probability distribution across the entire model vocabulary (50,257 tokens) at the final sequence position.
- Highlights the top 8 candidates with real percentage bars.
- **Click-to-Append**: Click any suggested token to automatically append it to your sentence and re-evaluate.

### 4. 🎛️ Probability Playground (Offline Simulator)
- Works independently without requiring model download.
- Add, adjust, or remove custom token rows with live $(0, 1]$ validation.
- **Built-in Presets**:
  - **I love Python**: $P = [0.5, 0.8, 0.2] \rightarrow \text{PPL} \approx 2.321$
  - **All 1/8**: 8 uniform tokens with $P = 0.125 \rightarrow \text{PPL} = 8.000$ (representing 8 equally likely choices)
  - **One Surprising Word**: 3 expected tokens ($P = 0.9$) followed by 1 rare token ($P = 0.01$).
- **Interactive Challenge**: Real-time goal indicator triggering when sequence perplexity drops below $1.5$.

---

## 🛠️ Tech Stack & Architecture

- **Frontend**: Pure HTML5, Modern CSS3 (CSS Grid & Flexbox, responsive fluid scaling), Vanilla JavaScript (ES Modules).
- **Inference Engine**: [Transformers.js v3](https://github.com/huggingface/transformers.js) via jsDelivr CDN.
- **Model**: `Xenova/distilgpt2` (8-bit quantized ONNX).
- **Zero Build Setup**: Single self-contained application file (`index.html`), no node modules, no bundler, no framework overhead.

---

## 🏁 Quick Start

Because the application uses ES Modules and WebAssembly, it must be served over HTTP/HTTPS rather than opened directly via `file://`.

### Running Locally

```bash
# Clone the repository
git clone https://github.com/Deepsabhadiya31/Perplexity-Explorer.git
cd Perplexity-Explorer

# Start a lightweight local server (Python 3)
python3 -m http.server 8080

# Or with Node.js npx
# npx serve .
```

Open your browser and navigate to:
```
http://localhost:8080/
```

> **Note**: On the first calculation, the model weights (~60 MB) are downloaded and cached locally by the browser. Subsequent evaluations run completely offline and instantaneously.

---

## 📐 Mathematical Formulation

Given a sequence of tokens $W = (w_1, w_2, \dots, w_N)$, a causal language model assigns probability to the sequence via the chain rule:

$$P(W) = \prod_{i=1}^N P(w_i \mid w_1, \dots, w_{i-1})$$

The perplexity of the sequence is the inverse geometric mean of the token probabilities:

$$\text{PPL}(W) = P(W)^{-\frac{1}{N}} = \sqrt[N]{\prod_{i=1}^N \frac{1}{P(w_i \mid w_{<i})}}$$

To prevent arithmetic underflow with longer sequences, calculations are performed in log-space:

$$\text{PPL}(W) = \exp\left( -\frac{1}{N} \sum_{i=1}^N \ln P(w_i \mid w_{<i}) \right)$$


