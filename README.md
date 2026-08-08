# Continuous-Time Generative Flows: From Flow Matching to 1-Step Improved Mean Flows

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A PyTorch implementation exploring continuous-time generative image modeling on MNIST and CIFAR-10. The repo is designed to be able to run on 1 moderately powerful GPU (T4) that you can access for free on Kaggle or Google Colab. The code progresses from multi-step **Flow Matching** to single-step **Mean Flows** and **Improved Mean Flows (iMF)**.

This project's main motivation was an exercise into learning the methods that have come after diffusion and translating research paper theory into code. 

---

## Motivation

Diffusion models and Continuous Normalizing Flows (CNFs) produce high-fidelity samples, but traditionally require integrating along an Ordinary Differential Equation (ODE) over multiple steps sequentially. 

While Optimal Transport (OT) straightens sampling paths, standard Flow Matching still learns *instantaneous* velocity vectors $v(x_t, t)$. Taking a single massive Euler step along instantaneous vectors at intermediate timesteps leads to path off-shooting and blurry samples.

The goal of this project is to build, evaluate, and benchmark **fast-forward continuous flows** that learn the integrated **average velocity** across time windows $[r, t]$. By doing so, we can jump directly from pure noise ($t=0$) to data ($t=1$) in a single function evaluation (**1-NFE**) without needing offline distillation, teacher models, or multi-stage training.

---

## Summary of Methods

### 1. Flow Matching (FM)
* **Paper:** [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) (Lipman et al., ICLR 2023)
* **Concept:** Regresses a neural network onto the vector field $v_t(x)$ of conditional probability paths. By pairing standard Gaussian noise $x_0$ with target data $x_1$ via linear optimal transport interpolation ($x_t = (1-t)x_0 + t x_1$), FM minimizes trajectory curvature.
* **Sampling:** Multi-step (Euler method). Solve the ODE using instantaneous velocity to approximate the curved vector field. Usually requires at least 16 steps for good results on MNIST from experiments.

### 2. Mean Flows (MF)
* **Paper:** [Mean Flows for One-step Generative Modeling](https://arxiv.org/abs/2505.13447) (Gao et al., 2025)
* **Concept:** Shifts the modeling target from instantaneous velocity $v$ to average velocity $u(z_r, r, t)$ over arbitrary time windows $[r, t]$. Uses the Fundamental Theorem of Calculus to establish the **Mean Flow Identity**:
  $$u(z_r, r, t) = v(z_r, r) - (t - r) \frac{d}{dr} u(z_r, r, t)$$
  This differential relation enables training 1-step look-ahead trajectories directly from scratch.
* **Sampling:** 1-NFE ( $x_1 = x_0 + u(x_0, 0, 1)$ ).

### 3. Improved Mean Flows (iMF)
* **Paper:** [Improved Mean Flows: On the Challenges of Fastforward Generative Models](https://arxiv.org/abs/2512.02012) (Geng et al., 2025)
* **Concept:** Resolves parameter-dependence issues in the original Mean Flow loss by recasting it into a well-posed regression problem. iMF extracts the boundary velocity $v = u_\theta(z_r, r, r)$ directly from the model, passes it as a spatial directional tangent into a Jacobian-Vector Product (JVP) via forward-mode AD (`torch.func.jvp`), and compares the resulting compound prediction $V_\theta$ against ground-truth velocity $(x_1 - x_0)$.
* **Sampling:** 1-NFE ( $x_1 = x_0 + u(x_0, 0, 1)$ ).

### Notes

Flow matching can be seen as a special case of Mean Flows where $p(r=t) = 1$. When $r = t$ we are predicting the instantaneous velocity at time $t$. This way we can train a model that takes in parameters (z, r, t) for all 3 loss functions.

q---

## 🚀 Quickstart

### Environment Setup
```bash
git clone [https://github.com/curtisjchen/flow-matching.git](https://github.com/curtisjchen/flow-matching.git)
cd flow-matching
uv sync
```

### Multi-GPU Training (`torchrun`)
To launch distributed training across 2 GPUs (e.g., dual NVIDIA T4s on Kaggle):

```bash
torchrun --nproc_per_node=2 train.py --config_path configs/dit_mnist_fm_xl_compile.yaml
```

## Models Used
### 1. UNet
A UNet was used as the baseline for the project. I didn't save the results so unfortunately I can't put it here, but a UNet is basically a neural network architecture than uses residual connections and downsampling and upsampling blocks in order to generate an image. The model takes in random Gaussian noise in the shape (batch_size, 1, 28, 28). time and cfg-weight are injected into the model using a sinusoidal time and weight embedding and then adding it to the latent vector at each step of the network.

### 2. Diffusion Transformer (DiT)
Diffusion transformers replaced UNets for diffusion tasks. They are similar in that they take in random noise and output an image where the input and output are of the same shape. All 3 types of loss can use the same architecture, which no change in parameter size, so it is convenient to perform ablations. 

Here are the parameters and hyperparameters used:

| Model | DiT-Base | 
| :---: | :---: |
| Parameters | 4.2M |
| Hidden Size | 196 |
| Num Heads | 4 |
| Num Layers | 6 |
| Patch Size | 2 |
| Epochs Trained | 300 |
| LR | 3e-4 |
| Min LR | 1e-4 |
| LR Scheduler | Cosine Annealing |
| Warmup Steps | 5 |

---

## 📊 Experimental Results

### 1. MNIST
#### Trained conditionally but evaluated unconditionally
| Method | $p(r=t)$ | Model | 1-NFE FID-50K | 32-NFE FID-50K | 
| :--- | :---: | :---:| :---: | :---: | 
| **Flow Matching** | 1.0 | DiT-Base | 361.2 | 10.2 | 
| **Mean Flow** | 0.5 | DiT-Base | 59.2 | x |
| **Improved Mean Flow** | 0.5 | DiT-Base | 39.0 | x |

#### Visual Samples
*(Insert sample grids for 1-NFE vs 50-NFE here)*
```text
[ Placeholders for MNIST Generated Sample Grids ]
```

#### Classifier free guidance performance 
To evaluate each model's performance on generating conditioned samples, we can use a pretrained classifier that has high accuracy on MNIST, and pass our generated samples through that classifier. We use the classifier as an oracle.

---

### 2. CIFAR-10 (Unconditional Baseline)

| Method | $p(r=t)$ | Model | 1-NFE FID-50K | 32-NFE FID-50K | 
| :--- | :---: | :---:| :---: | :---: | 
| **Flow Matching** | 1.0 | DiT-Base | x | x | 
| **Mean Flow** | 0.5 | DiT-Base | x | x |
| **Improved Mean Flow** | 0.5 | DiT-Base | x | x | 

#### Visual Samples
*(Insert CIFAR-10 unconditional sample grids here)*
```text
[ Placeholders for CIFAR-10 Generated Sample Grids ]
```

---

## References

```bibtex
@inproceedings{lipman2023flow,
  title={Flow Matching for Generative Modeling},
  author={Lipman, Yaron and Chen, Ricky T. Q. and Ben-Hamu, Heli and Nickel, Maximilian and Le, Matt},
  booktitle={ICLR},
  year={2023}
}

@article{gao2025meanflows,
  title={Mean Flows for One-step Generative Modeling},
  author={Gao, et al.},
  journal={arXiv preprint arXiv:2505.13447},
  year={2025}
}

@article{geng2025improvedmeanflows,
  title={Improved Mean Flows: On the Challenges of Fastforward Generative Models},
  author={Geng, Zhengyang and Lu, Yiyang and Wu, Zongze and Shechtman, Eli and Kolter, J. Zico and He, Kaiming},
  journal={arXiv preprint arXiv:2512.02012},
  year={2025}
}
```