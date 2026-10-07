# PRODIGY TASK 04 — Image-to-Image Translation using Conditional GAN

<p align="center">
  <b>Image-to-Image Translation using pix2pix Conditional GAN</b>
</p>

<p align="center">
  A deep learning project that learns to translate architectural label images into realistic facade images using a Conditional Generative Adversarial Network.
</p>

---

## 📌 Project Overview

This project implements an **Image-to-Image Translation model using a Conditional Generative Adversarial Network (cGAN)**.

The project uses the **pix2pix architecture**, which learns a mapping between paired input and target images.

For this implementation, the **Facades dataset** is used. The model takes an architectural label/map image as input and generates a corresponding realistic building facade image.

The model consists of two main neural networks:

- **U-Net Generator** — Generates realistic target images from input images.
- **PatchGAN Discriminator** — Determines whether an image pair is real or generated.

---

## 🎯 Objective

The main objective of this project is to build a Conditional GAN capable of learning an image-to-image mapping from paired training data and generating realistic target images from previously unseen input images.

---

## ✨ Key Features

- 🖼️ Paired image-to-image translation
- 🤖 Conditional GAN-based architecture
- 🧠 U-Net based Generator
- 🔍 PatchGAN based Discriminator
- 🔗 Skip connections for preserving image details
- 🎨 Realistic facade image generation
- 📐 Image preprocessing and normalization
- 🔄 Random cropping and horizontal flipping for data augmentation
- 📊 Generator and Discriminator loss tracking
- 📈 Training loss visualization
- 💾 Trained Generator model saving
- ⚡ GPU-accelerated training using Google Colab

---

## 🛠️ Tech Stack

### Programming Language
- Python

### Deep Learning
- TensorFlow
- Keras
- Conditional Generative Adversarial Networks (cGAN)
- pix2pix
- U-Net
- PatchGAN

### Data Processing
- NumPy
- TensorFlow Data API

### Visualization
- Matplotlib

### Development Environment
- Google Colab
- Jupyter Notebook
- VS Code
- Git & GitHub

### Hardware Acceleration
- NVIDIA GPU / CUDA
- Google Colab GPU

---

## 📂 Dataset

### Facades Dataset

The project uses the **Facades dataset**, which contains paired architectural images.

Each dataset image contains two sections:

```text
┌─────────────────────┬─────────────────────┐
│                     │                     │
│   Input / Label     │   Target / Facade   │
│       Image         │       Image         │
│                     │                     │
└─────────────────────┴─────────────────────┘
```
