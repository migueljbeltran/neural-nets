# ECS 170 Artificial Intelligence

## Spring 2026 Course Project: Stage 4 Report

## Team Information

**Team Name:** python pookies

**Student 1:** Miguel Beltran  
**Student 1 ID:** 922433771  
**Student 1 Email:** mmbeltran@ucdavis.edu

**Student 2:** Himani Manjunath  
**Student 2 ID:** 923180010  
**Student 2 Email:** hmanjunath@ucdavis.edu

**Student 3:** Nithya Sunku  
**Student 3 ID:** 923186959  
**Student 3 Email:** nsunku@ucdavis.edu

## Section 1: Task Description

This stage uses recurrent neural networks for text classification and text generation. For classification, the model predicts whether an IMDb movie review is positive or negative. For generation, the model learns from short joke text and generates new text starting from three seed words.

## Section 2: Model Description

The models are implemented in PyTorch. Text is cleaned by converting to lowercase and keeping simple word tokens. Each word is converted to an integer id from a vocabulary built on the training data.

For classification, the model uses an embedding layer, one recurrent layer, dropout, and a final linear layer for binary sentiment prediction. The classifier tracks each review's real length and reads the recurrent output at the final real word, so padding tokens do not decide the label. For generation, the model uses an embedding layer, one recurrent layer, and a final linear layer that predicts the next word.

The same code is used for three recurrent units:

| Variant | Recurrent Unit |
|---|---|
| RNN | Vanilla recurrent unit |
| LSTM | Long short-term memory unit |
| GRU | Gated recurrent unit |

## Section 3: Experiment Settings

### 3.1 Dataset Description

| Dataset | Task | Train Size | Test Size |
|---|---|---:|---:|
| IMDb reviews | Binary sentiment classification | 25,000 | 25,000 |
| Joke text | Text generation | 1,622 jokes | N/A |

IMDb has balanced positive and negative labels. The joke dataset is a CSV file with one joke per row.

### 3.2 Detailed Experimental Setups

| Task | Max Vocab | Max Length | Embedding Dim | Hidden Dim | Epochs | Batch Size |
|---|---:|---:|---:|---:|---:|---:|
| Classification | 10,000 | 200 | 64 | 128 | 4 | 128 |
| Generation | 3,000 | 5 | 64 | 64 | 10 | 128 |

Adam optimization and cross-entropy loss are used for all models. The code automatically uses CUDA or MPS if available, otherwise it uses CPU.

### 3.3 Evaluation Metrics

Classification is evaluated with accuracy, precision, recall, and F1 score. Generation is evaluated qualitatively by comparing whether generated text has similar short joke structure and vocabulary to the training jokes.

### 3.4 Source Code

Important files:

```text
script/stage_4_script/script_rnn.py
src/stage_4_code/dataset_loader.py
src/stage_4_code/method_rnn_classifier.py
src/stage_4_code/method_rnn_generator.py
```

### 3.5 Training Convergence Plot

Learning curve PNG files and CSV history files are saved in:

```text
result/stage_4_result/
```

### 3.6 Classification Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| RNN | 0.6267 | 0.6509 | 0.5466 | 0.5942 |
| LSTM | 0.7584 | 0.7784 | 0.7225 | 0.7494 |
| GRU | 0.8241 | 0.8593 | 0.7750 | 0.8150 |

### 3.7 Text Generation Results

The seed words were:

```text
what did the
```

| Model | Generated Text |
|---|---|
| RNN | what did the winter candy are on a few for a little lover it was his chickens by was a pair |
| LSTM | what did the other candy the other day says up to the road to the his chickens the tree you get |
| GRU | what did the other slide the bee use to the other day but it was a chickens joke because you in |

The generated text does not exactly match one training joke. The results are not fully grammatical, but they use short joke-like phrases and common training words. The shorter output length makes the examples easier to read.

## Section 4: Conclusion

The RNN, LSTM, and GRU models use the same preprocessing and training settings, so the comparison mainly shows the effect of the recurrent unit. GRU performed best on the classification task with 0.8241 accuracy and 0.8150 F1. The generated text is still rough because the generator is intentionally small and trained for only 10 epochs, but the decreasing training loss shows that it learned common word patterns from the joke dataset.
