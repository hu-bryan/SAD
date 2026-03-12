# Generative Data Distillation with Reward-Guided Flow Models

## Introduction

Training modern machine learning models often requires very large datasets, which can be expensive to store, train on, and distribute. **Data distillation** aims to compress a large dataset into a much smaller synthetic dataset that preserves most of the original dataset’s utility for downstream learning tasks.

This project explores a **generative approach to data distillation**, where synthetic samples are generated using a generative model rather than optimized directly through expensive bilevel optimization. The goal is to produce a small set of synthetic images that, when used for training, achieve competitive performance on downstream classification tasks.

Instead of relying on traditional pixel-level optimization approaches, this project investigates **reward-guided generation**, where a generative model produces candidate samples that are nudged toward being representative of the original dataset.

The core research question explored is:

> Can a generative model produce high-quality distilled datasets by guiding generation with a representativeness reward?

---

# Hypothesis

Early work in data distillation relied on **computationally expensive bilevel optimization**, where synthetic samples were directly optimized to train a downstream model.

Later work introduced **generative approaches**, particularly using GANs and diffusion models. Diffusion-based methods guide generation using gradients computed from intermediate noisy states of the image.

However, scoring intermediate noisy states can produce unstable or misleading signals, since those states may not resemble the final generated sample.

This project investigates a different generative paradigm based on **flow-based models**.

Unlike diffusion models, the flow model used in this project carries an estimate of the **expected final sample** during the generation process. This enables a **look-ahead reward mechanism**, where generation is guided using an estimate of the final image rather than the current noisy intermediate state.

The hypothesis explored in this project is therefore:

> Flow-based generative models with look-ahead reward guidance provide a more stable and interpretable signal for data distillation compared to diffusion-based guidance.

---

# Pipeline and Methodology

The system is structured as an **end-to-end generative data distillation pipeline**.

## Pipeline Overview

```
Dataset
  ↓
Feature Extraction
  ↓
Generative Model Sampling (Flow Model)
  ↓
Reward-Guided Generation
  ↓
Synthetic Dataset
  ↓
Classifier Training
  ↓
Evaluation on Real Test Set
```

The pipeline consists of several key stages.

---

## 1. Feature Extraction

Images from the original dataset are embedded using intermediate features from a pretrained convolutional network.

These feature representations are used to measure **representativeness** of generated samples relative to the real dataset.

---

## 2. Generative Model

A **flow-based generative model** is used to generate synthetic samples.

The generation process proceeds through a sequence of steps transforming noise into a structured sample. Unlike diffusion models, the flow model provides an estimate of the **expected final sample** during generation.

This enables reward guidance to operate on a **look-ahead prediction** rather than the current intermediate state.

---

## 3. Reward-Guided Sampling

During generation, samples are scored using a **representativeness reward function**.

The reward measures how well the generated sample covers regions of the dataset distribution in feature space.

Generation is nudged using:

```
Modified Drift = Model Drift + λ * Reward Gradient
```

Where:

- `λ` is a scaling hyperparameter controlling reward strength
- the reward gradient pushes samples toward representative regions of the dataset

This creates a **guided generative process** that biases the model toward producing useful distilled samples.

---

## 4. Synthetic Dataset Construction

The generative model produces a small synthetic dataset consisting of only a few samples per class.

These synthetic samples serve as the **distilled dataset**.

---

## 5. Downstream Evaluation

The quality of the distilled dataset is evaluated using downstream classification performance.

A classifier is trained using the synthetic dataset and evaluated on the real dataset's test split.

Performance metrics include:

- classification accuracy
- stability across architectures
- qualitative inspection of generated samples

The core evaluation principle is:

> Synthetic data quality is determined by downstream task performance.

---

# Experiment Tracking with Weights & Biases

Experiments are tracked using **Weights & Biases (W&B)** to ensure reproducibility and systematic experimentation.

Each run logs:

- model configuration
- reward scaling parameter (`λ`)
- sampling configuration
- classifier training metrics
- final evaluation accuracy
- runtime and system metrics

### Hyperparameter Sweeps

To identify effective reward scaling values, hyperparameter sweeps are performed over:

```
λ ∈ {0.01, 0.1, 0.5, 1, 5, 10}
```

This allows automated comparison of reward strength and its effect on downstream performance.

### Logged Metrics

Each experiment records:

- training loss
- reward magnitude
- classifier accuracy
- runtime
- GPU utilization

This structured tracking makes experiments **reproducible and comparable**.

---

# Results

The reward-guided generative approach successfully produced synthetic datasets capable of training classifiers with meaningful performance on the original dataset.

Key observations include:

- reward guidance significantly influences sample representativeness
- excessively large reward scaling leads to over-abstract samples
- moderate reward strength improves downstream classifier performance
- flow-based models enable more stable reward guidance than diffusion-style intermediate scoring

While the project is exploratory in nature, results demonstrate that **generative approaches to data distillation are viable and promising**.

---

# Conclusion

This project explored **reward-guided generative data distillation using flow-based models**.

The key contribution is the use of **look-ahead reward guidance**, which evaluates the expected final sample rather than noisy intermediate states during generation.

This approach offers several advantages:

- more interpretable reward signals
- improved stability during guided sampling
- flexible generative pipelines for dataset compression

The project demonstrates the feasibility of combining **generative modeling with data distillation**, and suggests that flow-based models provide a promising alternative to diffusion-based guidance.

Future work could explore:

- larger datasets (e.g. ImageNet)
- stronger reward functions
- alternative generative architectures
- improved evaluation across multiple downstream models

---

# Technologies Used

- Python  
- PyTorch  
- JAX  
- Weights & Biases  
- Flow-based generative models  
- Convolutional neural networks
