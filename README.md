# CloudModelTrainer

## Introduction

**CloudModelTrainer** is a Python tool designed to mount remote storage, load datasets, fine-tune models, and train AI directly from a mounted server. It allows you to train large AI models with datasets stored remotely and is optimized to work with GPUs using ROCm. The tool provides an easy setup for both local and remote training environments.

With this tool, you can:
- Mount remote directories via SSHFS.
- Install necessary dependencies (like SSHFS) if not already present.
- Automatically detect and utilize available AMD GPUs (with ROCm).
- Fine-tune machine learning models using custom datasets stored on a remote server.
- Automatically run the training job at system boot.
- Expose the trained model through an API for generating code.

---

## Features

- Mounts a remote server directory for AI model training.
- Installs necessary dependencies (e.g., SSHFS) if not already present.
- Automatically detects and uses AMD GPUs for training (with ROCm support).
- Loads models and datasets from Hugging Face and fine-tunes them.
- Option to configure training with a customizable dataset and model.
- Supports system boot integration for running training jobs automatically.
- Provides a simple API server to interact with the fine-tuned model.

---

## Installation

Follow the steps below to set up **CloudModelTrainer**:

### 1. Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/your-username/CloudModelTrainer.git
cd CloudModelTrainer
