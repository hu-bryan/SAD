# Generative Data Distillation with Reward-Attuned Flow Maps

## Introduction

This project studies a generative approach to **dataset distillation**: compressing a large real dataset into a very small synthetic dataset that still preserves downstream training utility.

The main idea is to generate distilled images using a **flow-based generative model** and guide sampling with a **representativeness reward**. Instead of optimizing pixels directly through expensive bilevel optimization, the method uses a pretrained **flow map** to generate candidate images and nudges the generation process toward samples that are more representative of the underlying class distribution.

The project was developed and evaluated primarily on **CIFAR-10**, with downstream validation using **ResNet-18**.

---

## Hypothesis

The core question of this project is:

> Can a flow-based generative model with look-ahead improve representativeness-guided sampling for data distillation?

This question came from a limitation I saw in diffusion-style guidance. In diffusion-based generation, if you want to guide the process at intermediate steps, you typically score the **current noisy intermediate state**. But those states are not yet close to the final sample, so the score can be unstable or hard to interpret.

Flow maps offer a different possibility. Because the model carries a **look-ahead estimate** of where the sample is expected to end up at time $t = 1$, guidance can be based on the model’s current best estimate of the **final image**, rather than the current noisy state.

My hypothesis was that this would make representativeness-guided sampling more meaningful and more stable for data distillation.

---

## Pipeline and Methodology

## Overview

The pipeline can be summarized as:

```text
Real dataset (CIFAR-10)
  ↓
Pretrained flow map
  ↓
Look-ahead estimate of final sample during generation
  ↓
Representativeness reward computed from U-Net hidden states
  ↓
Reward-guided Euler–Maruyama sampling
  ↓
Synthetic distilled dataset
  ↓
Train ResNet-18 on synthetic data
  ↓
Evaluate on real test set
```

---

## 1. Training the Flow Map

The project uses a flow map as the generative model. It is trained from on CIFAR-10 images.

A key property of the flow map is that it supports **look-ahead**: during generation, the model can estimate what the sample is expected to look like at the end of the trajectory. That estimate is then used inside the reward-guided sampling procedure.

This is the main modeling decision of the project. The flow model is not used just as a generic image generator. It is used specifically because its look-ahead structure makes reward guidance more natural than in diffusion-style intermediate-state scoring.

---

## 2. Representativeness-Guided Sampling

The generation process is guided by a representativeness reward.

At each stage of sampling, I use the flow map’s look-ahead estimate of the final image and score how representative that sample is relative to real examples from the same class.

The reward is based on distances between hidden representations, averaged across training images from a class. The purpose of this reward is to push generation toward regions of the data distribution that better capture class structure.

Conceptually, the guided process modifies the drift term so that the model follows both:

- the default generative dynamics, and
- a reward gradient encouraging representativeness

This produces a synthetic sample that is not just plausible, but hopefully more useful for downstream classification.

---

## 3. Hidden-State Scoring in the U-Net Backbone

The representativeness score was computed using hidden representations taken from the U-Net backbone of the flow model itself.

More specifically, the score used an averaged mixture of hidden states from selected encoder-side blocks:

- `32x32 block3`
- `16x16 block3`
- `8x8 block3` (bottleneck)

The idea was to compare generated and real images in the model’s own learned hidden space, rather than in raw pixel space.

This was important for two reasons:

1. pixel-space similarity is often too brittle or too low-level for dataset distillation
2. hidden states inside the U-Net give a more meaningful notion of semantic similarity for the reward

I avoided decoder-side blocks because they tended to introduce more pixel-level abstraction into the score and could make images overly abstract.

---

## 4. Sampling Method

Synthetic images were generated using **Euler–Maruyama** sampling.

In the experiments, I used:

- **2500 Euler–Maruyama steps**
- a representativeness strength parameter$\gamma$
- a pretrained flow map trained beforehand

Empirically, fewer steps produced noisier images and worse quality. Around 2500 steps gave much better convergence.

For the reward strength, I found that:

- ** $\gamma = 0.2$ ** worked best
- larger values, especially ** $\gamma \ge 0.5$ **, made images too abstract

So there was a clear tradeoff: stronger reward increased representativeness pressure, but too much reward damaged image quality.

---

## 5. Distilled Dataset Construction and Evaluation

Once synthetic images were generated, they were collected into a distilled dataset and used to train a downstream classifier.

The main downstream evaluation model was ResNet-18.

Evaluation was done by training on the synthetic dataset and testing on the real test data.

This is a key point in the project:

> The real measure of success is not just whether the generated images look good, but whether they preserve enough information to support downstream learning.

---

## Experiment Tracking and Monitoring with Weights & Biases

The codebase includes Weights & Biases (W&B) for experiment tracking and monitoring.

W&B was used to document and compare runs by logging:

- experiment progress
- training and optimization losses
- downstream classifier accuracy
- best accuracy achieved so far
- standard deviation across evaluations
- synthetic image snapshots
- latent code histograms
- synthetic pixel histograms
- run configurations and hyperparameters

This made it much easier to compare runs and analyze how changes in reward strength, sampling behavior, or optimization settings affected the final distilled dataset.

In particular, W&B is useful here because this project has several moving parts that interact:

- generative sampling behavior
- reward-guidance strength
- downstream evaluation performance
- qualitative image quality

Tracking all of those together in one place makes the project much more reproducible and easier to debug.

---

## Conclusion

This project explored whether flow-based models with look-ahead can improve representativeness-guided sampling for dataset distillation.

The main contribution is not just using a flow model as a generator, but using a flow map specifically for its look-ahead property. That lets guidance depend on the model’s estimate of the final image, rather than on a noisy intermediate state.

The representativeness reward was computed using hidden states from selected blocks of the model’s U-Net backbone, allowing guidance to happen in a learned representation space rather than raw pixel space.

In its current form, the project should be viewed as a proof of concept, but it demonstrates a promising idea:

- flow maps provide a natural mechanism for look-ahead guidance
- representativeness-guided generation can be adapted to dataset distillation
- hidden-state scoring inside the generator offers a meaningful way to define the reward

A stronger next version of the project would include broader ablations, larger-scale experiments, and stronger comparisons against other generative distillation methods.

---

## Tech Stack

- Python
- PyTorch
- JAX
- Weights & Biases
- Flow-based generative modeling
- Euler–Maruyama sampling
- ResNet-18 evaluation
- CIFAR-10
