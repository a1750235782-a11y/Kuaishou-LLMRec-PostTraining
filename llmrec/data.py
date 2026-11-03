"""Synthetic-data utilities with explicit input validation."""

import re

DOMAINS = {"prod", "video", "ad", "living"}
SID_RE = re.compile(r"(prod|video|ad|living):(\d+):(\d+):(\d+)")


def parse_sid(sid):
    """Demo-only SID syntax; not the competition tokenizer format."""
    match = SID_RE.fullmatch(sid)
    if not match:
        raise ValueError(f"Invalid demo SID: {sid!r}")
    domain, *codes = match.groups()
    if any(int(code) >= 8192 for code in codes):
        raise ValueError("Demo code must be in [0, 8192)")
    return domain, tuple(int(code) for code in codes)


def clean_caption(text):
    """Only remove exact demo marketing phrases; never infer attributes."""
    for phrase in ("限时抢购", "全网最低", "包邮到家"):
        text = text.replace(phrase, "")
    return " ".join(text.split())


def supervision_mode(domain):
    if domain not in DOMAINS:
        raise ValueError(f"Unknown domain: {domain}")
    return "think" if domain == "prod" else "nothink"


def select_trajectories(rows, target_sid, max_keep=2):
    """Select cached outputs for ONE request. Exact match is not reasoning proof."""
    parse_sid(target_sid)
    if max_keep < 1:
        raise ValueError("max_keep must be positive")
    kept, seen = [], set()
    for row in rows:
        reasoning = " ".join(row["reasoning"].split())
        if row["sid"] != target_sid or not reasoning or reasoning in seen:
            continue
        kept.append({"reasoning": reasoning, "sid": row["sid"]})
        seen.add(reasoning)
        if len(kept) == max_keep:
            break
    return kept


def build_dpo_pairs(chosen, catalog, known_positives=(), history=()):
    """One request; select at most one negative from each SID-prefix bucket.

    Same-domain catalog order makes this deterministic. Unobserved items are
    only assumed negatives; this does not resolve false-negative bias.
    """
    domain, codes = parse_sid(chosen)
    excluded = {chosen, *known_positives, *history}
    buckets = {}
    for candidate in dict.fromkeys(catalog):
        other_domain, other_codes = parse_sid(candidate)
        if other_domain != domain or candidate in excluded:
            continue
        if other_codes[0] != codes[0]:
            bucket = "different_first"
        elif other_codes[1] != codes[1]:
            bucket = "same_first_different_second"
        else:
            continue
        buckets.setdefault(bucket, candidate)
    return [{"chosen": chosen, "rejected": sid, "bucket": bucket}
            for bucket, sid in buckets.items()]
