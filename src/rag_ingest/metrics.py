"""Text accuracy metrics."""


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance, O(len(a) * len(b)) time, O(len(b)) memory."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(hypothesis: str, reference: str) -> float:
    """Character error rate: edits needed per reference character (lower is better)."""
    return edit_distance(hypothesis, reference) / max(len(reference), 1)
