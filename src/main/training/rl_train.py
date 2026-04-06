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


def sequence_logprob(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-sequence mean log-prob over non-padding tokens."""
    log_probs = F.log_softmax(logits, dim=-1)
    token_lp = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
    mask = (labels != -100).float()
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


def expand_batch(batch: dict, k: int) -> dict:
    """Repeat each example K times so we can generate K rollouts per input."""
    return {
        key: val.repeat_interleave(k, dim=0)
        for key, val in batch.items()
    }


def rl_train(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Number of rollouts per input example (reduces PPO variance)
    n_rollouts = cfg["rl"].get("n_rollouts", 4)
    print(f"Rollouts per sample: {n_rollouts}")

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

    kl_coef = cfg["rl"].get("kl_coef", 0.1)

    value_head = ValueHead(hidden_size=cfg["model"]["fused_dim"]).to(device)

    # ── Tokenizer & reward ──
    tokenizer = BartTokenizer.from_pretrained(cfg["model"]["bart_model"])
    reward_fn = CombinedReward(
        retrieval_weight=cfg["rl"]["reward_weights"]["retrieval"],
        nli_weight=cfg["rl"]["reward_weights"]["nli"],
        hallucination_weight=cfg["rl"]["reward_weights"]["hallucination"],
        device=str(device),
    )

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
    history = []
    step = 0

    pbar = tqdm(total=cfg["training"]["total_steps"], desc="RL fine-tuning")

    while step < cfg["training"]["total_steps"]:
        for batch in loader:
            if step >= cfg["training"]["total_steps"]:
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            B = batch["audio_features"].shape[0]

            # ── Collect K rollouts per input ──────────────────────────────
            # Expand batch: [B, ...] → [B*K, ...]
            expanded = expand_batch(batch, n_rollouts)

            model.eval()
            with torch.no_grad():
                generated_ids = model.generate(
                    audio_features=expanded["audio_features"],
                    text_input_ids=expanded["text_input_ids"],
                    text_attention_mask=expanded["text_attention_mask"],
                    max_length=cfg["data"]["max_target_len"],
                    num_beams=1,          # sampling, not beam search
                    do_sample=True,       # stochastic rollouts for diversity
                    temperature=0.9,
                )

            generated_strs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            reference_strs = tokenizer.batch_decode(
                expanded["labels"].clamp(min=0), skip_special_tokens=True
            )
            context_strs = tokenizer.batch_decode(
                expanded["text_input_ids"], skip_special_tokens=True
            )

            # rewards: [B*K]
            rewards_flat = reward_fn(generated_strs, reference_strs, context_strs).to(device)

            # ── Normalize rewards within each group of K rollouts ──────────
            rewards_grouped = rewards_flat.view(B, n_rollouts)
            baseline = rewards_grouped.mean(dim=1, keepdim=True)
            rewards_normed = (rewards_grouped - baseline).view(-1)

            # ── Prepare generated labels for forward pass ─────────────────
            gen_labels = generated_ids.clone()
            # Mask pad tokens with -100 for loss computation
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            gen_labels[gen_labels == pad_id] = -100

            # Build forward batch using original audio/text but generated labels
            rl_batch = {
                "audio_features": expanded["audio_features"],
                "text_input_ids": expanded["text_input_ids"],
                "text_attention_mask": expanded["text_attention_mask"],
                "labels": gen_labels,
            }

            # ── Forward pass on expanded batch ────────────────────────────
            model.train()
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=cfg["training"]["fp16"]):
                output = model(**rl_batch)
                logprobs_new = sequence_logprob(output.logits, gen_labels)
                values = value_head(output.encoder_hidden_states.detach())

                with torch.no_grad():
                    ref_output = ref_model(**rl_batch)
                    logprobs_ref = sequence_logprob(ref_output.logits, gen_labels)

                kl_div = logprobs_new - logprobs_ref
                rewards_penalized = rewards_normed - kl_coef * kl_div.detach()

            logprobs_old = logprobs_new.detach()

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
            mean_reward = rewards_flat.mean().item()
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
                    torch.save({
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "value_head_state_dict": value_head.state_dict(),
                        "best_reward": best_reward,
                    }, ckpt_dir / "checkpoint_best_rl.pt")

                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "value_head_state_dict": value_head.state_dict(),
                }, ckpt_dir / f"checkpoint_step{step}.pt")

    pbar.close()

    with open(ckpt_dir / "rl_log.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"RL training complete. Best reward: {best_reward:.4f}")
    print(f"Checkpoints saved to: {ckpt_dir}")
