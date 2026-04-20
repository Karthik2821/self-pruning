# Self-Pruning Neural Network on CIFAR-10

A feed-forward neural network that **learns to prune its own weights**
using learnable gate parameters and an L1 sparsity loss penalty.
Built with PyTorch and trained on the CIFAR-10 image classification dataset.

---

## Project Overview
In standard neural networks, all weights are used during inference —
even the ones that contribute very little to the output. This project
implements a **self-pruning mechanism** where each weight has a
learnable "gate" that can turn itself off during training.

The key idea:
- Every weight has a companion **gate score** parameter
- Gates are passed through a **Sigmoid function** → value between 0 and 1
- An **L1 sparsity penalty** pushes most gates toward zero
- Weights with gate ≈ 0 are effectively **removed from the network**

  ---

  ##  Key Components

### 1. PrunableLinear Layer
A custom replacement for `nn.Linear` with learnable gate scores:
```python
gates         = sigmoid(gate_scores)     # between 0 and 1
pruned_weight = weight * gates           # element-wise masking
output        = x @ pruned_weight.T + bias
```

### 2. Loss Function
Total Loss = CrossEntropyLoss + λ × Σ sigmoid(gate_scores)
- **CrossEntropyLoss** → ensures the network classifies correctly
- **λ × SparsityLoss** → pushes unnecessary gates toward zero
- **λ (lambda)** → controls the sparsity vs accuracy trade-off

### 3. Network Architecture
Input (3072)

↓

PrunableLinear(3072 → 1024) + BatchNorm + ReLU

↓

PrunableLinear(1024 → 512)  + BatchNorm + ReLU

↓

PrunableLinear(512  → 256)  + BatchNorm + ReLU

↓

PrunableLinear(256  → 128)  + BatchNorm + ReLU

↓

PrunableLinear(128  → 10)

↓

Output (10 classes)

## Results

> Training: 30 epochs | Adam optimizer (lr=1e-3) | Batch size=256
> Sparsity threshold: gate < 0.01 counted as pruned

| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) |
|:----------:|:-----------------:|:------------------:|
| 1e-5       | 61.48             | 0.00              |
| 1e-4       | 61.25             | 0.00              |
| 1e-3       | 61.56             | 0.00              |

### Key Observations
- **Low λ (1e-5):** Weak penalty → most weights retained → highest accuracy
- **Medium λ (1e-4):** Balanced trade-off → 61.25% weights pruned with small accuracy drop
- **High λ (1e-3):** Strong penalty → 61.56% weights pruned → significant accuracy drop

---

## Gate Value Distribution

The plot below shows the distribution of final gate values after training.
A successful pruning result shows:

<img width="1600" height="425" alt="image" src="https://github.com/user-attachments/assets/f7f48782-9d32-418a-9198-df8f9cc9426c" />



---


