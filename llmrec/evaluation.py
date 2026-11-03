"""Transparent SID-level diagnostics. Not the official multi-task scorer."""

from .data import parse_sid


def evaluate(rows, catalog, k=32):
    """Take first k raw outputs per mode BEFORE filtering; no replacement draws.

    Validity requires catalog membership AND the request's target domain.
    Every request, including all-invalid output, remains in the denominator.
    """
    if not rows or k < 1:
        raise ValueError("Need nonempty rows and positive k")
    catalog = set(catalog)
    for sid in catalog:
        parse_sid(sid)
    hits = valid_count = raw_count = 0
    overlaps, sizes = [], []
    for row in rows:
        targets = set(row["targets"])
        if not targets or not targets <= catalog:
            raise ValueError("Targets must be nonempty catalog members")
        if any(parse_sid(s)[0] != row["domain"] for s in targets):
            raise ValueError("Target domain mismatch")
        modes = []
        for mode in ("think", "nothink"):
            raw = row[mode][:k]
            valid = [sid for sid in raw if sid in catalog and parse_sid(sid)[0] == row["domain"]]
            raw_count += len(raw)
            valid_count += len(valid)
            modes.append(set(valid))
        think, nothink = modes
        merged = think | nothink
        hits += bool(merged & targets)
        sizes.append(len(merged))
        overlaps.append(len(think & nothink) / len(merged) if merged else 0.0)
    return {
        "n_requests": len(rows),
        "k_per_mode": k,
        "sid_hit_rate": hits / len(rows),
        "valid_sid_rate": valid_count / raw_count if raw_count else 0.0,
        "mean_unique_candidates": sum(sizes) / len(rows),
        "mean_mode_jaccard": sum(overlaps) / len(rows),
    }
