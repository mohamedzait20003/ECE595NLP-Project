import os
import sys
import yaml
import torch
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.main.model import MainModel
from src.main.training import TrainingCallback
from src.main.utils import CustomDataset, Collator

# Global Constants
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Helper Functions
def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Evaluation Function
def evaluate(model, val_loader, device, max_batches: int = 50) -> float:
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            count += 1

    model.train()
    return total_loss / count if count > 0  else float('inf')

# Main Training Loop
def train(config_path: str):
    # Load config and set seed
    config = load_config(config_path)
    set_seed(config["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    print("Loading Model...")
    model = MainModel(
        whispher_model=config["model"]["whisper_model"],
        bart_model=config["model"]["bart_model"],
        freeze_audio=config["model"]["freeze_audio"],
        freeze_text=config["model"]["freeze_text"],
        fused_dim=config["model"]["fused_dim"],
        num_heads=config["model"]["fusion_heads"],
        num_layers=config["model"]["fusion_layers"]
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,}")

    # Prepare data
    train_manifest = str(PROJECT_ROOT / config["data"]["train_manifest"])
    val_manifest = str(PROJECT_ROOT / config["data"]["val_manifest"])

    train_dataset = CustomDataset(
        manifest_path=train_manifest,
        whisper_model=config["model"]["whisper_model"],
        bart_model=config["model"]["bart_model"],
        max_audio_len=config["data"]["max_audio_len"],
        max_text_len=config["data"]["max_text_len"],
        max_target_len=config["data"]["max_target_len"]
    )

    val_dataset = CustomDataset(
        manifest_path=val_manifest,
        whisper_model=config["model"]["whisper_model"],
        bart_model=config["model"]["bart_model"],
        max_audio_len=config["data"]["max_audio_len"],
        max_text_len=config["data"]["max_text_len"],
        max_target_len=config["data"]["max_target_len"]
    )

    collator = Collator(bart_model=config["model"]["bart_model"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        collate_fn=collator,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        collate_fn=collator,
        pin_memory=True
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")

    # Optimizer and Scheduler
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["training"]["warmup_steps"],
        num_training_steps=config["training"]["total_steps"]
    )

    scaler = torch.cuda.amp.GradScaler(enabled=config["training"]["fp16"])

    # Callbacks
    checkpoint_dir = str(PROJECT_ROOT / config["training"]["checkpoint_dir"])
    callback = TrainingCallback(
        checkpoint_dir=checkpoint_dir,
        log_every=config["training"]["log_every"],
        save_every=config["training"]["save_every"]
    )

    # Training Loop
    print("\nStarting Stage 1 pre-training...")
    step = 0
    accum_steps = config["training"]["gradient_accumulation_steps"]
    total_steps = config["training"]["total_steps"]
    max_grad_norm = config["training"]["max_grad_norm"]

    model.train()
    optimizer.zero_grad()

    pbar = tqdm(total=total_steps, desc="Pre-training", unit="step")

    while step < total_steps:
        for batch in train_loader:
            if step >= total_steps:
                break

            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.cuda.amp.autocast(enabled=config["training"]["fp16"]):
                output = model(**batch)
                loss = output.loss / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            step += 1
            train_loss = loss.item() * accum_steps
            pbar.update(1)
            pbar.set_postfix(loss=f"{train_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            callback.on_step(step, train_loss, scheduler.get_last_lr()[0])

            if step % config["training"]["eval_every"] == 0:
                val_loss = evaluate(model, val_loader, device)
                callback.on_eval(step, val_loss, model)
                model.train()

            if step % config["training"]["save_every"] == 0:
                callback.on_save(step, model)

    pbar.close()

    callback.on_train_end()
    print(f"\nStage 1 complete. Best val loss: {callback.best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")