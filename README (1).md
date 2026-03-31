# GSoC 2026 – QML-HEP Evaluation Tasks

This repository contains my implementations for the **Quantum Machine Learning for High Energy Physics (QML-HEP)** evaluation tasks for **Google Summer of Code 2026**.

---

## Task Overview

| Task | Title | Status |
|------|-------|--------|
| **Task I** | Quantum Computing Basics (Cirq / PennyLane) | ✅ Complete |
| **Task II** | Classical GNN – Quark/Gluon Jet Classification | ✅ Complete |
| **Task III** | Open Task – Commentary on QML | ✅ Complete |
| **Task IV** | Quantum GAN (QGAN) with Cirq + TFQ | ✅ Complete |
| **Task V** | Quantum Graph Neural Network (QGNN) | ✅ Complete |
| **Task VI** | Quantum Representation Learning (Contrastive Loss) | ✅ Complete |
| **Task VII** | Equivariant Quantum Neural Network (Z₂×Z₂) | ✅ Complete |
| **Task VIII** | Vision Transformer + Quantum ViT (MNIST) | ✅ Complete |
| **Task IX** | Kolmogorov-Arnold Network (KAN) + Quantum KAN | ✅ Complete |
| **Task X** | Diffusion Network for Jet Images (DeepFalcon Task 2) | ✅ Complete |
| **Task XI** | Hybrid MLP + Parameterized Quantum Circuit (PQC) | ✅ Complete |
| **Task XII** | Actor-Critic RL for Quantum State Embedding | ✅ Complete |

---

## Task Descriptions

**Task I – Quantum Computing Basics**  
Implements two quantum circuits using PennyLane: a 5-qubit circuit with Hadamard, CNOT, SWAP, and RX gates; and a SWAP test circuit to measure fidelity between two 2-qubit states.

**Task II – Classical GNN**  
Classifies quark vs. gluon jets from the [ParticleNet dataset](https://zenodo.org/record/3164691) using two GNN architectures (GCN and GAT). Point-cloud data is projected to a graph via kNN (sklearn). Includes feature normalization and architecture comparison.

**Task III – Open Task**  
Written commentary on quantum machine learning concepts, algorithms, and potential research directions. Original writing; no code.

**Task IV – Quantum GAN (QGAN)**  
Trains a QGAN using Cirq and TensorFlow Quantum to separate signal from background events in HEP data. Includes discriminator-based classification and training curves.  
*Note: Requires `numpy==1.26.4` and `tensorflow-quantum==0.7.6`.*

**Task V – Quantum GNN (QGNN)**  
Extends Task II by describing and implementing a quantum graph neural network circuit. Demonstrates how the graph structure of jet data can be encoded into a parameterized quantum circuit.

**Task VI – Quantum Representation Learning**  
Implements contrastive representation learning on MNIST using a PQC. Two images are embedded as quantum states and a SWAP test measures their fidelity. Same-class pairs maximize fidelity; different-class pairs minimize it. Uses PCA preprocessing.

**Task VII – Equivariant QNN**  
Generates a Z₂×Z₂-symmetric classification dataset and trains two QNNs — a standard QNN and a structurally equivariant QNN — to compare how symmetry-aware circuit design improves generalization.

**Task VIII – Vision Transformer + QViT**  
Trains a classical ViT on MNIST (10 epochs, CosineAnnealingLR) reaching ~97% accuracy. Proposes and details a Quantum Vision Transformer architecture using parameterized circuits and SWAP-test-based attention.

**Task IX – KAN + Quantum KAN**  
Applies a classical Kolmogorov-Arnold Network (B-spline basis) to MNIST using the `efficient-kan` library. Discusses a quantum extension using parameterized rotations as learnable activation functions.

**Task X – Diffusion Network (DeepFalcon Task 2)**  
Trains a DDPM with a lightweight U-Net (~1.5M params) on 32×32 jet images (T=300 timesteps). Uses log1p normalization to handle sparse physics data. Discusses extensions toward quantum diffusion models.

**Task XI – Hybrid MLP + PQC**  
An MLP maps normally distributed inputs to PQC rotation angles, minimizing MSE loss against target angles. The trained MLP is then used to parameterize a PennyLane quantum circuit for state preparation demonstration.

**Task XII – RL Quantum Embedding**  
Uses an Actor-Critic (PPO-style) agent to learn an embedding policy that prepares quantum states via a PQC. Reward is based on MSE between target and prepared state parameters.

---

## Datasets

| Task | Dataset | Source |
|------|---------|--------|
| II, V | ParticleNet – Quark/Gluon Jets | [Zenodo 3164691](https://zenodo.org/record/3164691) |
| IV | HEP Signal/Background (Delphes, 200 events) | [Google Drive](https://drive.google.com/file/d/1r_MZB_crfpij6r3SxPDeU_3JD6t6AxAj/view) |
| VI, VIII, IX | MNIST | Via `torchvision` / `tensorflow.keras.datasets` |
| X | Jet Images (32×32) | [Google Drive](https://drive.google.com/file/d/1WO2K-SfU2dntGU4Bb3IYBp9Rh7rtTYEr) |
| XI, XII | Synthetic (Normal Distribution) | Generated in-notebook |

---

## Setup

All notebooks are self-contained and designed to run on **Google Colab (free tier)**. Each notebook installs its own dependencies in the first cell.

```bash
git clone https://github.com/arnavsinghal09/GSOC-QMLHEP
```

Open any `.ipynb` in Colab and run all cells from top to bottom.

### Key dependency notes

- **Task IV**: Pin `numpy==1.26.4` and `tensorflow-quantum==0.7.6` for TFQ + Keras 3 compatibility
- **Task IX**: Install `efficient-kan` from GitHub (`git+https://github.com/Blealtan/efficient-kan.git`); restart runtime after install
- **Task XII**: Uninstall `autoray` before installing PennyLane to avoid JAX version conflicts

---

## Frameworks Used

- [PennyLane](https://pennylane.ai/) — Primary quantum framework (Tasks VI, VII, XI, XII)
- [Cirq](https://quantumai.google/cirq) + [TensorFlow Quantum](https://www.tensorflow.org/quantum) — Tasks IV
- [PyTorch](https://pytorch.org/) + [PyTorch Geometric](https://pyg.org/) — Tasks II, V, VIII, IX, XI, XII
- [Qiskit](https://qiskit.org/) — Task I (reference)

---

## References

- [QML-HEP GSoC 2026 Task Description](https://docs.google.com/document/d/1imoMEyC0r5IESonwgA7BThEQWDfdrOsoyfMfyJgyXmU)
- [Equivariant QNNs – arXiv:2205.06217](https://arxiv.org/abs/2205.06217)
- [ParticleNet – Zenodo 3164691](https://zenodo.org/record/3164691)
- [DeepFalcon Task Description](https://docs.google.com/document/d/15XrY0vLMWQgVEvNSBcy4knGpoaw6X7s6hSo_SpX-nw8)
