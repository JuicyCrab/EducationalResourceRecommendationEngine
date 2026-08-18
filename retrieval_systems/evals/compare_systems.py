"""
This module provides functions to compare the performance of different retrieval systems.

It computes mean precision, recall, and MRR across a set of per-query metric dicts 
for BM25, dense, and hybrid systems. It also computes per-query MRR deltas to 
analyze the consistency of improvements across queries.
"""


def compute_mean_metrics(per_query_metrics: list[dict]) -> dict:
    """
    Compute mean precision, recall, and MRR across metric dicts.
    """
    n = len(per_query_metrics)
    if n == 0:
        return {
            "mean_precision": 0.0,
            "mean_recall": 0.0,
            "mean_mrr": 0.0,
            "n_queries": 0,
        }

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
    Compare mean metrics across BM25, dense, and hybrid retrieval systems.
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
    """Print the retrieval system comparison metrics."""
    header = (
        f"{'System':<10} {'Mean MRR':<12} {'Mean Precision':<16} "
        f"{'Mean Recall':<12} {'N queries'}"
    )
    print(header)

    for name, means in [
        ("BM25", comparison["bm25_means"]),
        ("Dense", comparison["dense_means"]),
        ("Hybrid", comparison["hybrid_means"]),
    ]:
        print(
            f"{name:<10} {means['mean_mrr']:<12.4f} "
            f"{means['mean_precision']:<16.4f} "
            f"{means['mean_recall']:<12.4f} {means['n_queries']}"
        )

    print("\nPer-query MRR deltas (hybrid vs dense, hybrid vs bm25):")
    for row in comparison["per_query_deltas"]:
        # FIXED: Wrapped long print strings to maximize code readability
        print(
            f"  {row['query_id']}: bm25={row['bm25_mrr']:.2f} "
            f"dense={row['dense_mrr']:.2f} hybrid={row['hybrid_mrr']:.2f}  "
            f"(Δ vs dense={row['hybrid_minus_dense']:+.2f}, "
            f"Δ vs bm25={row['hybrid_minus_bm25']:+.2f})"
        )

    n_wins = sum(1 for r in comparison["per_query_deltas"] if r["hybrid_minus_dense"] > 0)
    n_ties = sum(1 for r in comparison["per_query_deltas"] if r["hybrid_minus_dense"] == 0)
    n_losses = sum(1 for r in comparison["per_query_deltas"] if r["hybrid_minus_dense"] < 0)
    
    print(
        f"\nHybrid vs Dense: {n_wins} wins, {n_ties} ties, {n_losses} losses "
        f"(out of {len(comparison['per_query_deltas'])} queries)"
    )
