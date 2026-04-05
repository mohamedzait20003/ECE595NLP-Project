import torch
import torch.nn.functional as F
from transformers import pipeline
from sentence_transformers import SentenceTransformer

class RetrievalReward:
    """Cosine similarity between generated and reference citation embeddings."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, generated: list[str], references: list[str]) -> torch.Tensor:
        gen_emb = self.model.encode(generated, convert_to_tensor=True, normalize_embeddings=True)
        ref_emb = self.model.encode(references, convert_to_tensor=True, normalize_embeddings=True)
        scores = (gen_emb * ref_emb).sum(dim=-1)  # [B]
        return scores.cpu()


class NLIReward:
    """Entailment probability: context entails generated citation."""
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small", device: int = 0):
        self.pipe = pipeline("text-classification", model=model_name, device=device)

    def __call__(self, contexts: list[str], generated: list[str]) -> torch.Tensor:
        pairs = [f"{ctx} [SEP] {gen}" for ctx, gen in zip(contexts, generated)]
        results = self.pipe(pairs, top_k=None, truncation=True, max_length=512)
        scores = []

        for result in results:
            label_score = {r["label"].lower(): r["score"] for r in result}
            scores.append(label_score.get("entailment", 0.0))
        return torch.tensor(scores)


class HallucinationPenalty:
    """Semantic hallucination penalty using sentence embeddings.

    Computes 1 - cosine_similarity(generated, context) so that outputs
    semantically far from the context receive a high penalty.
    """
    def __init__(self, encoder: SentenceTransformer):
        self.encoder = encoder

    @torch.no_grad()
    def __call__(self, contexts: list[str], generated: list[str]) -> torch.Tensor:
        ctx_emb = self.encoder.encode(contexts, convert_to_tensor=True, normalize_embeddings=True)
        gen_emb = self.encoder.encode(generated, convert_to_tensor=True, normalize_embeddings=True)
        sim = (ctx_emb * gen_emb).sum(dim=-1)
        penalty = 1.0 - sim
        return penalty.cpu()


class CombinedReward:
    def __init__(self, retrieval_weight=0.4, nli_weight=0.4, hallucination_weight=0.2, device="cpu"):
        dev_idx = 0 if device == "cuda" else -1
        self.retrieval = RetrievalReward(device=device)
        self.nli = NLIReward(device=dev_idx)

        self.hallucination = HallucinationPenalty(encoder=self.retrieval.model)

        self.w_ret = retrieval_weight
        self.w_nli = nli_weight
        self.w_hall = hallucination_weight

    def __call__(
        self,
        generated: list[str],
        references: list[str],
        contexts: list[str],
    ) -> torch.Tensor:
        r_ret  = self.retrieval(generated, references)
        r_nli  = self.nli(contexts, generated)
        r_hall = self.hallucination(contexts, generated)

        reward = (
            self.w_ret  * r_ret
            + self.w_nli  * r_nli
            - self.w_hall * r_hall
        )
        return reward
