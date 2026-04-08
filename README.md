# CiteMind — Multimodal Audio-Language Model for Academic Citation Recommendation

**ECE 595: Natural Language Processing — Course Project**

CiteMind is a multimodal model that takes a **spoken academic claim** (audio) and a **paper's title + abstract** (text) and generates the most likely citation in author-year format (e.g., `Wu et al., 2023`). It combines Whisper's audio encoder with BART's sequence-to-sequence decoder via a cross-modal fusion layer, trained in two stages: masked generative pre-training followed by PPO-based reinforcement learning.

---

## Results

| Metric | Text Only | Stage 1 | Stage 2 (RL) |
|---|---|---|---|
| BLEU | 0.479 | 0.498 | **0.505** |
| ROUGE-L | 0.465 | 0.518 | **0.523** |
| Format Accuracy | 0.170 | 0.890 | **0.980** |
| Hallucination Rate | 0.830 | 0.110 | **0.020** |
| Author Accuracy | 0.005 | 0.075 | **0.085** |
| Year Accuracy | 0.475 | 0.350 | **0.365** |
| Recall@5 | 0.000 | 0.010 | **0.010** |

**Key findings:**
- Audio is essential — format accuracy collapses from 89% → 17% without it
- RL reduces hallucination rate from 11% → 2%
- RL improves format accuracy from 89% → 98%

---

## Architecture

```
Audio (WAV) ──► Whisper Encoder (frozen) ──────────────┐
                                                        ├──► CrossModalFusion ──► BART Decoder ──► Citation
Text (title + abstract) ──► BART Encoder ──────────────┘
```

| Component | Model | Params | Frozen |
|---|---|---|---|
| Audio Encoder | `openai/whisper-small` | 122M | Yes |
| Text Encoder + Decoder | `facebook/bart-base` | 139M | No |
| NLI (reward) | `cross-encoder/nli-deberta-v3-small` | 86M | Yes |
| Retrieval (reward) | `sentence-transformers/all-MiniLM-L6-v2` | 22M | Yes |

---

## Project Structure

```
Code/
├── requirements.txt
├── src/
│   ├── config/
│   │   ├── pretrain_config.yaml       # Stage 1 hyperparameters
│   │   └── rl_config.yaml             # Stage 2 hyperparameters
│   ├── main/
│   │   ├── model/
│   │   │   ├── main_model.py          # Full model (audio + text + fusion + decoder)
│   │   │   ├── audio_encoder.py       # Whisper encoder wrapper
│   │   │   ├── text_encoder.py        # BART encoder wrapper
│   │   │   ├── fusion.py              # CrossModalFusion (concat + transformer)
│   │   │   └── citation_head.py       # LM head for citation generation
│   │   ├── training/
│   │   │   ├── pretrain.py            # Stage 1: masked generative pre-training
│   │   │   ├── rl_train.py            # Stage 2: PPO-based RL fine-tuning
│   │   │   └── reward.py              # CombinedReward + ExactMatchReward
│   │   ├── evaluation/
│   │   │   ├── metrics.py             # BLEU, ROUGE-L, hallucination_rate, MRR@5, Recall@5
│   │   │   └── evaluate.py            # evaluate_checkpoint, compare_checkpoints
│   │   ├── inference/
│   │   │   └── pipeline.py            # CitationPipeline: audio → ranked citations
│   │   └── utils/
│   │       ├── dataset.py             # CustomDataset (audio + text + labels)
│   │       ├── collator.py            # Batch collation + padding
│   │       ├── data_processor.py      # Citation extraction (author-year + numeric)
│   │       └── tts_synthesizer.py     # pyttsx3-based audio synthesis
│   ├── scripts/
│   │   ├── download_data.py           # Semantic Scholar API data collection
│   │   ├── run_synthesis.py           # TTS synthesis launcher
│   │   ├── run_pretrain.py            # Stage 1 training launcher
│   │   ├── run_rl.py                  # Stage 2 training launcher
│   │   └── evaluate_dataset.py        # Dataset validation before training
│   ├── notebooks/
│   │   ├── 03_training_stage1.ipynb   # Colab: Stage 1 pre-training (A100)
│   │   ├── 04_training_stage2.ipynb   # Colab: Stage 2 RL fine-tuning (A100)
│   │   └── 05_evaluation.ipynb        # Colab: evaluation + ablation study
│   └── data/
│       ├── raw/                       # papers.json, citation_contexts.json
│       ├── processed/                 # train.json, val.json, test.json
│       └── audio/                     # .wav files + manifests
```

---

## Setup

```bash
git clone https://github.com/mohamedzait20003/ECE595NLP-Project
cd ECE595NLP-Project
pip install -r requirements.txt
```

Set your Semantic Scholar API key in `.env`:
```
S2_API_KEY=your_key_here
```

---

## Training Pipeline

### Step 1 — Download data
```bash
python src/scripts/download_data.py \
    --query "natural language processing" \
    --max_papers 1000 \
    --max_workers 3
```

### Step 2 — Preprocess citations
```bash
python src/scripts/run_synthesis.py
```

### Step 3 — Validate dataset
```bash
python src/scripts/evaluate_dataset.py --stage all
```

### Step 4 — Stage 1: Pre-training (run in Colab)
Open `src/notebooks/03_training_stage1.ipynb` on Google Colab with A100 GPU.

Or locally:
```bash
python src/scripts/run_pretrain.py --config src/config/pretrain_config.yaml
```

### Step 5 — Stage 2: RL Fine-tuning (run in Colab)
Open `src/notebooks/04_training_stage2.ipynb` on Google Colab with A100 GPU.

Or locally:
```bash
python src/scripts/run_rl.py --config src/config/rl_config.yaml
```

### Step 6 — Evaluate
Open `src/notebooks/05_evaluation.ipynb` on Google Colab.

---

## Inference

```python
from src.main.inference.pipeline import CitationPipeline

pipeline = CitationPipeline("path/to/checkpoint_best_rl.pt")

candidates = pipeline.predict(
    audio_path="claim.wav",
    source_title="Attention Is All You Need",
    source_abstract="We propose a new simple network architecture...",
    num_candidates=5,
)

for c in candidates:
    print(f"{c['citation']}  (confidence: {c['confidence']:.0%})")
```

Or from a masked sentence (synthesizes audio automatically):

```python
candidates = pipeline.predict_from_text(
    masked_sentence="The transformer architecture [MASK] has become the dominant paradigm.",
    source_title="BERT: Pre-training of Deep Bidirectional Transformers",
    source_abstract="We introduce BERT...",
)
```

---

## Training Configuration

**Stage 1** (`src/config/pretrain_config.yaml`):

| Parameter | Value |
|---|---|
| `batch_size` | 32 |
| `gradient_accumulation_steps` | 2 |
| `effective_batch` | 64 |
| `learning_rate` | 5e-5 |
| `total_steps` | 5,000 |
| `warmup_steps` | 300 |

**Stage 2** (`src/config/rl_config.yaml`):

| Parameter | Value |
|---|---|
| `batch_size` | 8 |
| `learning_rate` | 5e-7 |
| `total_steps` | 1,500 |
| `kl_coef` | 3.0 |
| `ppo_epochs` | 1 |
| `kl_target` | 0.1 |

---

## Dataset

- **Source**: Semantic Scholar API (NLP + Deep Learning papers)
- **Train / Val / Test**: 63,972 / 7,996 / 7,997 samples
- **Citation types**: 77.4% numeric (`[9]`) / 22.6% author-year (`Wu et al., 2023`)
- **Audio**: synthesized via pyttsx3 at 16 kHz, masked sentence with `[MASK]` → "citation needed"
- **Valid samples**: 99.7% pass all training requirements

---

## Reward Function (Stage 2)

```
R = 0.3 × R_retrieval  +  0.3 × R_nli  −  0.2 × R_hallucination  +  0.5 × R_exact_match
```

| Component | Signal |
|---|---|
| `R_retrieval` | Semantic similarity (MiniLM) between generated and reference |
| `R_nli` | NLI entailment score (DeBERTa-v3) of cited abstract vs claim |
| `R_hallucination` | Penalizes invalid citation format |
| `R_exact_match` | +0.5 exact author, +0.25 first-letter match; +0.5 exact year, +0.25 off-by-one |
