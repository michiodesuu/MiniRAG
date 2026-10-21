"""Text accuracy metrics."""
from collections import Counter

NGRAM = 3


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


def _ngrams(text: str, n: int) -> Counter:
    return Counter(text[i:i + n] for i in range(len(text) - n + 1))


def ngram_overlap(hypothesis: str, reference: str, n: int = NGRAM) -> tuple[float, float]:
    """(recall, precision) of character n-grams, ignoring order. Recall: how much
    of the reference was read (text left out lowers it); precision: how much of
    the output is real (invented or repeated text lowers it). Unlike CER, text
    read in another order (e.g. a sidebar after the main panel) is not an error."""
    ref, hyp = _ngrams(reference, n), _ngrams(hypothesis, n)
    common = sum((ref & hyp).values())
    return common / max(sum(ref.values()), 1), common / max(sum(hyp.values()), 1)
