import os
import json
import torch
from pathlib import Path
from datetime import datetime

class TrainingCallback:
    def __init__(self, checkpoint_dir: str, log_every: int = 50, save_every: int = 1000):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_every = log_every
        self.save_every = save_every

        self.best_val_loss = float('inf')
        self.history = []

        self.log_path = self.checkpoint_dir / "training_log.json"

    def on_step(self, step: int, loss: float, lr: float):
        if step % self.log_every == 0:
            entry = {
                "step": step,
                "train_loss": round(loss, 4),
                "lr": lr,
                "time": datetime.now().strftime("%H:%M:%S")
            }

            self.history.append(entry)
            print(f"  Step {step:>6} | loss: {loss:.4f} | lr: {lr:.2e}")

    def on_eval(self, step: int, val_loss: float, model: torch.nn.Module) -> bool:
        print(f"\n  [Eval] Step {step} | val_loss: {val_loss:.4f}", end="")

        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
            self._save(model, step, tag="best")
            print(f"  ← best so far, saved checkpoint")
        else:
            print()

        return is_best

    def on_save(self, step: int, model: torch.nn.Module):
        self._save(model, step, tag=f"step_{step}")

    def on_train_end(self):
        with open(self.log_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"\nTraining log saved to {self.log_path}")

    def _save(self, model: torch.nn.Module, step: int, tag: str):
        path = self.checkpoint_dir / f"checkpoint_{tag}.pt"
        torch.save({
            "step": step,
            "model_state_dict": model.state_dict(),
            "val_loss": self.best_val_loss
        }, path) 
            