def compute_mean_metrics(per_query_metrics: list[dict]) -> dict:
    """
    Computes mean precision, recall, and MRR across a set of per-query metric dicts.
    Expects each dict to have 'precision', 'recall', 'mrr' keys (e.g. the
    'per_query' list already produced by evaluate_batch).

    NOTE: evaluate_batch's own top-level precision/recall/mrr fields are raw
    SUMS, not means -- don't use those directly for a resume number. This
    function recomputes the mean from the per-query list instead.
    """
    n = len(per_query_metrics)
    if n == 0:
        return {"mean_precision": 0.0, "mean_recall": 0.0, "mean_mrr": 0.0, "n_queries": 0}

    mean_precision = sum(m["precision"] for m in per_query_metrics) / n
    mean_recall = sum(m["recall"] for m in per_query_metrics) / n
    mean_mrr = sum(m["mrr"] for m in per_query_metrics) / n

    return {
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "mean_mrr": mean_mrr,
        "n_queries": n,
    }


def compare_systems(bm25_per_query, dense_per_query, hybrid_per_query) -> dict:
    """
    Compares mean MRR/precision/recall across BM25, dense, and hybrid systems.
    All three inputs MUST come from evaluate_batch(...)["per_query"] run on the
    SAME query range and SAME k -- this function does not verify that itself,
    so confirm it before calling.

    Also computes per-query MRR deltas (hybrid - dense, hybrid - bm25) so you
    can see whether any improvement is consistent across queries or driven by
    a small number of outliers -- important to check before trusting a
    resume-worthy aggregate number, especially with a small query set.
    """
    bm25_ids = [m["query_id"] for m in bm25_per_query]
    dense_ids = [m["query_id"] for m in dense_per_query]
    hybrid_ids = [m["query_id"] for m in hybrid_per_query]

    if not (bm25_ids == dense_ids == hybrid_ids):
        raise ValueError(
            "Query ID sets/order don't match across systems -- these results "
            "aren't from the same query range/order and can't be validly compared. "
            f"bm25: {bm25_ids}\ndense: {dense_ids}\nhybrid: {hybrid_ids}"
        )

    bm25_means = compute_mean_metrics(bm25_per_query)
    dense_means = compute_mean_metrics(dense_per_query)
    hybrid_means = compute_mean_metrics(hybrid_per_query)

    per_query_deltas = []
    for b, d, h in zip(bm25_per_query, dense_per_query, hybrid_per_query):
        per_query_deltas.append({
            "query_id": h["query_id"],
            "bm25_mrr": b["mrr"],
            "dense_mrr": d["mrr"],
            "hybrid_mrr": h["mrr"],
            "hybrid_minus_dense": h["mrr"] - d["mrr"],
            "hybrid_minus_bm25": h["mrr"] - b["mrr"],
        })

    return {
        "bm25_means": bm25_means,
        "dense_means": dense_means,
        "hybrid_means": hybrid_means,
        "per_query_deltas": per_query_deltas,
    }


def print_comparison(comparison: dict):
    """Pretty-prints the comparison for quick inspection."""
    print(f"{'System':<10} {'Mean MRR':<12} {'Mean Precision':<16} {'Mean Recall':<12} {'N queries'}")
    for name, means in [
        ("BM25", comparison["bm25_means"]),
        ("Dense", comparison["dense_means"]),
        ("Hybrid", comparison["hybrid_means"]),
    ]:
        print(f"{name:<10} {means['mean_mrr']:<12.4f} {means['mean_precision']:<16.4f} "
              f"{means['mean_recall']:<12.4f} {means['n_queries']}")

    print("\nPer-query MRR deltas (hybrid vs dense, hybrid vs bm25):")
    for row in comparison["per_query_deltas"]:
        print(f"  {row['query_id']}: bm25={row['bm25_mrr']:.2f} dense={row['dense_mrr']:.2f} "
              f"hybrid={row['hybrid_mrr']:.2f}  "
              f"(Δ vs dense={row['hybrid_minus_dense']:+.2f}, Δ vs bm25={row['hybrid_minus_bm25']:+.2f})")

    n_wins_vs_dense = sum(1 for r in comparison["per_query_deltas"] if r["hybrid_minus_dense"] > 0)
    n_ties_vs_dense = sum(1 for r in comparison["per_query_deltas"] if r["hybrid_minus_dense"] == 0)
    n_losses_vs_dense = sum(1 for r in comparison["per_query_deltas"] if r["hybrid_minus_dense"] < 0)
    print(f"\nHybrid vs Dense: {n_wins_vs_dense} wins, {n_ties_vs_dense} ties, {n_losses_vs_dense} losses "
          f"(out of {len(comparison['per_query_deltas'])} queries)")