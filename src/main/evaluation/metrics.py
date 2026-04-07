import re
import math
from collections import Counter


def _ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def bleu_score(generated: str, reference: str, max_n: int = 4) -> float:
    """Corpus-free sentence-level BLEU (smoothed)."""
    gen_tokens = generated.lower().split()
    ref_tokens = reference.lower().split()

    if not gen_tokens or not ref_tokens:
        return 0.0

    scores = []
    for n in range(1, max_n + 1):
        gen_ng = _ngrams(gen_tokens, n)
        ref_ng = _ngrams(ref_tokens, n)
        if not gen_ng:
            scores.append(0.0)
            continue
        ref_counts = Counter(ref_ng)
        matches = sum(min(count, ref_counts[ng]) for ng, count in Counter(gen_ng).items())
        # Add-1 smoothing
        scores.append((matches + 1) / (len(gen_ng) + 1))

    # Geometric mean
    log_avg = sum(math.log(s) for s in scores if s > 0) / max(len(scores), 1)

    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(gen_tokens), 1)))

    return bp * math.exp(log_avg)


def rouge_l(generated: str, reference: str) -> float:
    """ROUGE-L F1 based on longest common subsequence."""
    gen_tokens = generated.lower().split()
    ref_tokens = reference.lower().split()

    if not gen_tokens or not ref_tokens:
        return 0.0

    # LCS length via DP
    m, n = len(gen_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if gen_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]

    precision = lcs / m
    recall = lcs / n
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def exact_match(generated: str, reference: str) -> float:
    """1.0 if generated matches reference after normalization, else 0.0."""
    def normalize(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r'[^\w\s]', '', s)
        s = re.sub(r'\s+', ' ', s)
        return s
    return 1.0 if normalize(generated) == normalize(reference) else 0.0


def _extract_author(text: str) -> str:
    """Extract the first author name from a citation string."""
    text = text.strip()
    m = re.match(r'^([A-Z][a-z]+)', text)
    return m.group(1).lower() if m else ""


def _extract_year(text: str) -> str:
    """Extract a 4-digit year from a citation string."""
    m = re.search(r'((?:19|20)\d{2})', text)
    return m.group(1) if m else ""


def author_accuracy(generated: str, reference: str) -> float:
    """1.0 if first author last name matches, 0.5 for first-letter match, else 0.0."""
    gen_author = _extract_author(generated)
    ref_author = _extract_author(reference)

    if not gen_author or not ref_author:
        return 0.0
    if gen_author == ref_author:
        return 1.0
    if gen_author[0] == ref_author[0]:
        return 0.5
    return 0.0


def year_accuracy(generated: str, reference: str) -> float:
    """1.0 for exact year match, 0.5 for off-by-one, else 0.0."""
    gen_year = _extract_year(generated)
    ref_year = _extract_year(reference)

    if not gen_year or not ref_year:
        return 0.0
    if gen_year == ref_year:
        return 1.0
    if abs(int(gen_year) - int(ref_year)) == 1:
        return 0.5
    return 0.0


def format_accuracy(generated: str) -> float:
    """1.0 if the output matches a valid citation pattern."""
    patterns = [
        r'^[A-Z][a-z]+\s+et\s+al\.\s*,\s*\d{4}$',            # Wu et al., 2023
        r'^[A-Z][a-z]+\s+et\s+al\.\s*\(\d{4}\)$',             # Wu et al. (2023)
        r'^[A-Z][a-z]+\s+and\s+[A-Z][a-z]+\s*,\s*\d{4}$',     # Wu and Li, 2023
        r'^[A-Z][a-z]+\s+and\s+[A-Z][a-z]+\s*\(\d{4}\)$',     # Wu and Li (2023)
        r'^[A-Z][a-z]+\s*\(\d{4}\)$',                          # Li (2023)
        r'^[A-Z][a-z]+\s*,\s*\d{4}$',                          # Li, 2023
        r'^[A-Z]\s+et\s+al\.\s*,\s*\d{4}$',                    # K et al., 2024
        r'^[A-Z]\s+et\s+al\.\s*\(\d{4}\)$',                    # L et al. (2022)
    ]
    text = generated.strip()
    return 1.0 if any(re.match(p, text) for p in patterns) else 0.0


_VALID_YEAR_RANGE = (1990, 2026)


def hallucination_rate(generated: list[str]) -> float:
    """
    Fraction of outputs that are hallucinated.
    A citation is hallucinated if it has invalid format OR year outside [1990, 2026].
    """
    count = 0
    for g in generated:
        if format_accuracy(g) == 0.0:
            count += 1
            continue
        year_str = _extract_year(g)
        if not year_str:
            count += 1
            continue
        year = int(year_str)
        if not (_VALID_YEAR_RANGE[0] <= year <= _VALID_YEAR_RANGE[1]):
            count += 1
    return count / max(len(generated), 1)


def _normalize_citation(s: str) -> str:
    """Lowercase + strip punctuation for loose matching."""
    s = s.lower().strip()
    s = re.sub(r'[^\w\s]', '', s)
    return re.sub(r'\s+', ' ', s)


def mrr_at_k(
    candidates_list: list[list[str]], references: list[str], k: int = 5
) -> float:
    """
    Mean Reciprocal Rank @ K.
    candidates_list[i] is a ranked list of candidates for sample i.
    """
    rr_sum = 0.0
    for candidates, ref in zip(candidates_list, references):
        ref_norm = _normalize_citation(ref)
        for rank, cand in enumerate(candidates[:k], start=1):
            if _normalize_citation(cand) == ref_norm:
                rr_sum += 1.0 / rank
                break
    return rr_sum / max(len(references), 1)


def recall_at_k(
    candidates_list: list[list[str]], references: list[str], k: int = 5
) -> float:
    """
    Recall @ K.
    Fraction of samples where at least one of the top-K candidates matches the reference.
    """
    hits = 0
    for candidates, ref in zip(candidates_list, references):
        ref_norm = _normalize_citation(ref)
        if any(_normalize_citation(c) == ref_norm for c in candidates[:k]):
            hits += 1
    return hits / max(len(references), 1)


class CitationMetrics:
    """Compute all metrics for a list of generated vs reference citations."""

    METRIC_NAMES = [
        "bleu", "rouge_l", "exact_match",
        "author_accuracy", "year_accuracy", "format_accuracy",
        "hallucination_rate",
    ]

    BEAM_METRIC_NAMES = ["mrr_at_5", "recall_at_5"]

    def __call__(
        self, generated: list[str], references: list[str]
    ) -> dict[str, float]:
        n = len(generated)
        assert n == len(references), "Length mismatch"

        totals = {m: 0.0 for m in self.METRIC_NAMES if m != "hallucination_rate"}
        per_sample = []

        for gen, ref in zip(generated, references):
            sample = {
                "bleu": bleu_score(gen, ref),
                "rouge_l": rouge_l(gen, ref),
                "exact_match": exact_match(gen, ref),
                "author_accuracy": author_accuracy(gen, ref),
                "year_accuracy": year_accuracy(gen, ref),
                "format_accuracy": format_accuracy(gen),
            }
            per_sample.append(sample)
            for k, v in sample.items():
                totals[k] += v

        averages = {k: v / n for k, v in totals.items()}
        averages["hallucination_rate"] = hallucination_rate(generated)
        return {"averages": averages, "per_sample": per_sample}

    def compute_beam_metrics(
        self,
        candidates_list: list[list[str]],
        references: list[str],
        k: int = 5,
    ) -> dict[str, float]:
        """Compute MRR@K and Recall@K from multi-candidate output."""
        return {
            f"mrr_at_{k}": mrr_at_k(candidates_list, references, k),
            f"recall_at_{k}": recall_at_k(candidates_list, references, k),
        }
