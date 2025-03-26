#!/usr/bin/env python3

import os
import subprocess
import sys
import torch
import logging
import shutil
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from huggingface_hub import login


# --- Configuration ---
MOUNTED_DIR = "/mnt/remote"  # Default mount point, can be customized
TEMP_DIR = "/tmp/ai_training"
MODEL_DIR = os.path.join(MOUNTED_DIR, "models")
DATASET_DIR = os.path.join(MOUNTED_DIR, "datasets")
TRAINING_DIR = os.path.join(MOUNTED_DIR, "training")

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_gpu():
    """Check if an AMD GPU with ROCm is available."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"CUDA is available with {device_count} device(s): {device_name}")
        
        if "Radeon" in device_name or "AMD" in device_name:
            logger.info(f"Using AMD GPU: {device_name}")
            return torch.device("cuda")
        else:
            logger.warning("CUDA available, but AMD GPU not detected. Defaulting to CPU.")
    else:
        logger.warning("No compatible GPU found. Using CPU.")
    
    return torch.device("cpu")

def install_sshfs():
    """Install SSHFS if it's not installed."""
    try:
        subprocess.check_call(["sshfs", "-V"])
        logger.info("SSHFS is already installed.")
    except FileNotFoundError:
        logger.info("SSHFS not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sshfs"])
    except subprocess.CalledProcessError:
        logger.error("Failed to check SSHFS installation. Try installing manually.")
        sys.exit(1)

def mount_remote_folder(remote_address, remote_folder, local_folder):
    """Mount remote folder using SSHFS."""
    if not os.path.exists(local_folder):
        os.makedirs(local_folder)
    
    mount_cmd = f"sshfs {remote_address}:{remote_folder} {local_folder}"
    
    try:
        subprocess.check_call(mount_cmd, shell=True)
        logger.info(f"Mounted {remote_address}:{remote_folder} to {local_folder}.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to mount remote folder: {e}")
        sys.exit(1)

def add_mount_to_fstab(remote_address, remote_folder, local_folder):
    """Add the SSHFS mount to /etc/fstab for automatic mounting at boot."""
    fstab_entry = f"{remote_address}:{remote_folder} {local_folder} fuse.sshfs defaults,_netdev 0 0\n"
    with open("/etc/fstab", "a") as fstab:
        fstab.write(fstab_entry)
    logger.info(f"Added SSHFS mount to /etc/fstab for {remote_address}:{remote_folder}.")

def load_model_and_tokenizer():
    """Load model and tokenizer using mounted storage."""
    MODEL_NAME = "codellama/CodeLlama-13b-Instruct-hf"
    HUGGINGFACE_TOKEN = "YOUR_HF_TOKEN"
    
    logger.info(f"Loading model and tokenizer {MODEL_NAME}...")
    model_path = os.path.join(MODEL_DIR, MODEL_NAME.replace('/', '_'))
    os.makedirs(model_path, exist_ok=True)
    
    device = check_gpu()
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=model_path,
            token=HUGGINGFACE_TOKEN
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=model_path,
            token=HUGGINGFACE_TOKEN
        )
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def load_dataset_from_hub():
    """Load dataset using mounted storage and filter for specific programming languages."""
    DATASET_NAME = "codeparrot/github-code"
    
    logger.info(f"Loading dataset {DATASET_NAME}...")
    dataset_path = os.path.join(DATASET_DIR, DATASET_NAME.replace('/', '_'))
    os.makedirs(dataset_path, exist_ok=True)
    
    try:
        dataset = load_dataset(
            DATASET_NAME,
            split="train",
            trust_remote_code=True,
            cache_dir=dataset_path,
            token="YOUR_HF_TOKEN"
        )
        
        # Filter dataset for relevant programming languages
        languages_of_interest = {"Java", "Python", "HTML", "CSS", "JavaScript"}
        dataset = dataset.filter(lambda example: example.get("language") in languages_of_interest)
        
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def train_model(model, tokenizer, dataset):
    """Train the model and save to mounted storage."""
    from transformers import TrainingArguments, Trainer
    logger.info("Setting up training...")
    
    device = check_gpu()
    model.to(device)
    
    training_args = TrainingArguments(
        output_dir=TRAINING_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        save_steps=1000,
        evaluation_strategy="steps",
        save_total_limit=3,
        learning_rate=2e-5,
        num_train_epochs=3,
        fp16=True,
        logging_dir=os.path.join(TRAINING_DIR, "logs"),
        overwrite_output_dir=True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    final_model_path = os.path.join(MOUNTED_DIR, "fine_tuned_model")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    logger.info("Training complete and model saved to mounted drive.")
    return model, tokenizer

def cleanup():
    """Clean up temporary files on mounted storage."""
    try:
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        logger.info(f"Cleaned up temporary directory: {TEMP_DIR}")
    except Exception as e:
        logger.error(f"Error cleaning up: {e}")

def main():
    """Main function to run the script."""
    # Install SSHFS if not installed
    install_sshfs()

    # Prompt user for remote mount details
    remote_address = input("Enter the remote address (e.g., user@hostname): ")
    remote_folder = input("Enter the remote folder path (e.g., /path/to/folder): ")
    local_folder = input("Enter the local folder path (e.g., /mnt/remote): ")
    
    # Ask if they want to mount at boot
    mount_at_boot = input("Do you want to mount this folder at boot? (yes/no): ").lower()
    
    if mount_at_boot == "yes":
        add_mount_to_fstab(remote_address, remote_folder, local_folder)
    
    # Mount the remote folder
    mount_remote_folder(remote_address, remote_folder, local_folder)
    
    # Load model and dataset
    model, tokenizer = load_model_and_tokenizer()
    dataset = load_dataset_from_hub()
    
    # Train the model
    train_model(model, tokenizer, dataset)
    
    # Clean up
    cleanup()

if __name__ == "__main__":
    main()
