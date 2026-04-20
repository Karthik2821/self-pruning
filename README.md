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
