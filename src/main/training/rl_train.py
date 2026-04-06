import copy
import json
import yaml
import torch
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BartTokenizer, get_cosine_schedule_with_warmup

from src.main.model.main_model import MainModel
from src.main.utils.dataset import CustomDataset
from src.main.utils.collator import Collator
from src.main.training.reward import CombinedReward


class ValueHead(torch.nn.Module):
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden[:, 0, :]).squeeze(-1)  # [B]


class RunningRewardNormalizer:
    """Tracks running mean/std of rewards for stable advantage estimation."""
    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0

    def update(self, rewards: torch.Tensor):
        batch_mean = rewards.mean().item()
        batch_var = rewards.var().item() if rewards.numel() > 1 else 0.0
        batch_count = rewards.numel()

        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / max(total, 1)
        self.var = (
            (self.count * self.var + batch_count * batch_var
             + delta ** 2 * self.count * batch_count / max(total, 1))
            / max(total, 1)
        )
        self.count = total

    def normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        std = max(self.var ** 0.5, 1e-8)
        return (rewards - self.mean) / std


def sequence_logprob(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-sequence mean log-prob over non-padding tokens."""
    log_probs = F.log_softmax(logits, dim=-1)
    mask = (labels != -100).float()
    safe_labels = labels.clamp(min=0)  # -100 is invalid gather index; clamp then mask
    token_lp = log_probs.gather(2, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_lp * mask).sum(-1) / mask.sum(-1).clamp(min=1)


def ppo_loss(
    logprobs_new: torch.Tensor,
    logprobs_old: torch.Tensor,
    values: torch.Tensor,
    rewards: torch.Tensor,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
):
    advantages = rewards - values.detach()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    ratio = torch.exp(logprobs_new - logprobs_old)
    pg_loss = -torch.min(
        ratio * advantages,
        torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages,
    ).mean()

    vf_loss = F.mse_loss(values, rewards)
    return pg_loss + vf_coef * vf_loss, pg_loss.item(), vf_loss.item()


def rl_train(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ppo_epochs = cfg["rl"].get("ppo_epochs", 4)
    early_stop_patience = cfg["rl"].get("early_stop_patience", 10)

    # ── Active model (policy) ──
    model = MainModel(
        whispher_model=cfg["model"]["whisper_model"],
        bart_model=cfg["model"]["bart_model"],
        freeze_audio=cfg["model"]["freeze_audio"],
        freeze_text=cfg["model"]["freeze_text"],
        fused_dim=cfg["model"]["fused_dim"],
        num_heads=cfg["model"]["fusion_heads"],
        num_layers=cfg["model"]["fusion_layers"],
    ).to(device)

    ckpt = torch.load(cfg["rl"]["stage1_checkpoint"], map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded Stage 1 checkpoint (step {ckpt['step']}, val_loss {ckpt['val_loss']:.4f})")

    # ── Frozen reference model (KL penalty) ──
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    print("Created frozen reference model for KL divergence.")

    kl_coef = cfg["rl"].get("kl_coef", 0.3)

    value_head = ValueHead(hidden_size=cfg["model"]["fused_dim"]).to(device)

    # ── Tokenizer & reward ──
    tokenizer = BartTokenizer.from_pretrained(cfg["model"]["bart_model"])
    reward_fn = CombinedReward(
        retrieval_weight=cfg["rl"]["reward_weights"]["retrieval"],
        nli_weight=cfg["rl"]["reward_weights"]["nli"],
        hallucination_weight=cfg["rl"]["reward_weights"]["hallucination"],
        device=str(device),
    )

    reward_normalizer = RunningRewardNormalizer()

    # ── Data ──
    train_dataset = CustomDataset(
        manifest_path=cfg["data"]["train_manifest"],
        whisper_model=cfg["model"]["whisper_model"],
        bart_model=cfg["model"]["bart_model"],
        max_audio_len=cfg["data"]["max_audio_len"],
        max_text_len=cfg["data"]["max_text_len"],
        max_target_len=cfg["data"]["max_target_len"],
    )
    loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=Collator(),
        num_workers=cfg["data"]["num_workers"],
    )

    # ── Optimizer ──
    params = list(model.parameters()) + list(value_head.parameters())
    optimizer = torch.optim.AdamW(
        [p for p in params if p.requires_grad],
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg["training"]["warmup_steps"],
        num_training_steps=cfg["training"]["total_steps"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["training"]["fp16"])

    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_reward = -float("inf")
    no_improve_count = 0
    history = []
    step = 0

    pbar = tqdm(total=cfg["training"]["total_steps"], desc="RL fine-tuning")

    while step < cfg["training"]["total_steps"]:
        for batch in loader:
            if step >= cfg["training"]["total_steps"]:
                break

            batch = {k: v.to(device) for k, v in batch.items()}

            # ── Phase 1: Collect rollout (no gradients) ──────────────────
            model.eval()
            with torch.no_grad():
                generated_ids = model.generate(
                    audio_features=batch["audio_features"],
                    text_input_ids=batch["text_input_ids"],
                    text_attention_mask=batch["text_attention_mask"],
                    max_length=cfg["data"]["max_target_len"],
                    do_sample=True,
                    temperature=0.9,
                )

            generated_strs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            reference_strs = tokenizer.batch_decode(
                batch["labels"].clamp(min=0), skip_special_tokens=True
            )
            context_strs = tokenizer.batch_decode(
                batch["text_input_ids"], skip_special_tokens=True
            )

            rewards = reward_fn(generated_strs, reference_strs, context_strs).to(device)

            # Normalize rewards with running statistics
            reward_normalizer.update(rewards)
            rewards_norm = reward_normalizer.normalize(rewards)

            # Compute old logprobs and ref logprobs (frozen, computed once per rollout)
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=cfg["training"]["fp16"]):
                    old_output = model(**batch)
                    logprobs_old = sequence_logprob(old_output.logits, batch["labels"])

                    ref_output = ref_model(**batch)
                    logprobs_ref = sequence_logprob(ref_output.logits, batch["labels"])

                    kl_div = logprobs_old - logprobs_ref

            rewards_penalized = rewards_norm - kl_coef * kl_div.abs()

            # ── Phase 2: Multiple PPO epochs on this rollout ─────────────
            model.train()
            for _ in range(ppo_epochs):
                optimizer.zero_grad()

                with torch.amp.autocast("cuda", enabled=cfg["training"]["fp16"]):
                    output = model(**batch)
                    logprobs_new = sequence_logprob(output.logits, batch["labels"])

                # Value head outside autocast to avoid fp16 mismatch
                values = value_head(output.encoder_hidden_states.detach().float())

                loss, pg, vf = ppo_loss(
                    logprobs_new, logprobs_old, values, rewards_penalized,
                    clip_eps=cfg["rl"]["clip_eps"],
                    vf_coef=cfg["rl"]["vf_coef"],
                )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in params if p.requires_grad],
                    cfg["training"]["max_grad_norm"],
                )
                scaler.step(optimizer)
                scaler.update()

            scheduler.step()

            step += 1
            mean_reward = rewards.mean().item()
            mean_kl = kl_div.mean().item()
            pbar.set_postfix(
                reward=f"{mean_reward:.4f}",
                kl=f"{mean_kl:.4f}",
                pg=f"{pg:.4f}",
                vf=f"{vf:.4f}",
            )
            pbar.update(1)

            if step % cfg["training"]["log_every"] == 0:
                entry = {
                    "step": step,
                    "reward": mean_reward,
                    "kl_div": mean_kl,
                    "pg_loss": pg,
                    "vf_loss": vf,
                }
                history.append(entry)
                print(
                    f"  Step {step:>6} | reward: {mean_reward:.4f} | "
                    f"kl: {mean_kl:.4f} | pg: {pg:.4f} | vf: {vf:.4f}"
                )

            if step % cfg["training"]["save_every"] == 0:
                if mean_reward > best_reward:
                    best_reward = mean_reward
                    no_improve_count = 0
                    torch.save({
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "value_head_state_dict": value_head.state_dict(),
                        "best_reward": best_reward,
                    }, ckpt_dir / "checkpoint_best_rl.pt")
                else:
                    no_improve_count += 1

                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "value_head_state_dict": value_head.state_dict(),
                }, ckpt_dir / f"checkpoint_step{step}.pt")

                # Early stopping
                if no_improve_count >= early_stop_patience:
                    print(f"  Early stopping at step {step} (no improvement for "
                          f"{early_stop_patience} save intervals)")
                    step = cfg["training"]["total_steps"]  # break outer loop
                    break

    pbar.close()

    with open(ckpt_dir / "rl_log.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"RL training complete. Best reward: {best_reward:.4f}")
    print(f"Checkpoints saved to: {ckpt_dir}")
