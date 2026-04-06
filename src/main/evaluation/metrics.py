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


class CitationMetrics:
    """Compute all metrics for a list of generated vs reference citations."""

    METRIC_NAMES = [
        "bleu", "rouge_l", "exact_match",
        "author_accuracy", "year_accuracy", "format_accuracy",
    ]

    def __call__(
        self, generated: list[str], references: list[str]
    ) -> dict[str, float]:
        n = len(generated)
        assert n == len(references), "Length mismatch"

        totals = {m: 0.0 for m in self.METRIC_NAMES}
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
        return {"averages": averages, "per_sample": per_sample}
