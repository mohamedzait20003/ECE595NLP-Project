import re
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


class ExactMatchReward:
    """Rewards exact matches on author names and year."""

    # Common citation patterns: "Author et al., 2024", "Author (2024)", "Author and Other, 2024"
    _author_re = re.compile(r'^([A-Z][a-z]+(?:\s+(?:et\s+al|and\s+[A-Z][a-z]+))?)')
    _year_re = re.compile(r'((?:19|20)\d{2})')

    def __call__(self, generated: list[str], references: list[str]) -> torch.Tensor:
        scores = []
        for gen, ref in zip(generated, references):
            score = 0.0

            # Author match
            gen_author = self._extract_author(gen)
            ref_author = self._extract_author(ref)
            if gen_author and ref_author:
                if gen_author.lower() == ref_author.lower():
                    score += 0.5  # exact author match
                elif gen_author.lower() in ref_author.lower() or ref_author.lower() in gen_author.lower():
                    score += 0.25  # partial author match

            # Year match
            gen_year = self._extract_year(gen)
            ref_year = self._extract_year(ref)
            if gen_year and ref_year:
                if gen_year == ref_year:
                    score += 0.5  # exact year
                elif abs(int(gen_year) - int(ref_year)) == 1:
                    score += 0.25  # off by one year

            scores.append(score)
        return torch.tensor(scores)

    def _extract_author(self, text: str) -> str:
        text = text.strip()
        m = self._author_re.search(text)
        return m.group(1) if m else ""

    def _extract_year(self, text: str) -> str:
        m = self._year_re.search(text)
        return m.group(1) if m else ""


class CombinedReward:
    def __init__(self, retrieval_weight=0.4, nli_weight=0.4, hallucination_weight=0.2,
                 exact_match_weight=0.5, device="cpu"):
        dev_idx = 0 if device == "cuda" else -1
        self.retrieval = RetrievalReward(device=device)
        self.nli = NLIReward(device=dev_idx)

        self.hallucination = HallucinationPenalty(encoder=self.retrieval.model)
        self.exact_match = ExactMatchReward()

        self.w_ret = retrieval_weight
        self.w_nli = nli_weight
        self.w_hall = hallucination_weight
        self.w_exact = exact_match_weight

    def __call__(
        self,
        generated: list[str],
        references: list[str],
        contexts: list[str],
    ) -> torch.Tensor:
        r_ret  = self.retrieval(generated, references)
        r_nli  = self.nli(contexts, generated)
        r_hall = self.hallucination(contexts, generated)
        r_exact = self.exact_match(generated, references)

        reward = (
            self.w_ret  * r_ret
            + self.w_nli  * r_nli
            - self.w_hall * r_hall
            + self.w_exact * r_exact
        )
        return reward
