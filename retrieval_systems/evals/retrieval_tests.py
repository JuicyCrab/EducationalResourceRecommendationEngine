"""
This module tests and evaluates the retrieval systems (BM25, dense, and hybrid).

It includes functions to query batches of documents, evaluate retrieval performance, 
and compare the results across different systems.
"""

from dense.dense_retriever import DenseRetriever
from evals.compare_systems import compare_systems, print_comparison
from evals.eval_utils import EvalUtils
from evals.retriever_evals import RetrieverEvals
from hybrid.reciprocal_rank_fusion import ReciprocalRankFusion
from sparse.BM25_retriever import BM25Retriever


def query_batch(retrieval_system, start_idx: int, end_idx: int, k: int = 5) -> list[dict]:
    """Query a batch of documents using a single retrieval system."""
    queries = EvalUtils.get_resource_queries(start_idx, end_idx)
    if not isinstance(retrieval_system, (BM25Retriever, DenseRetriever)):
        raise ValueError(
            "Invalid retrieval system. Must be an instance of "
            "BM25Retriever or DenseRetriever."
        )

    batch_results = []
    for query in queries:
        relevant_ids = set(query["query_relevant_resource_ids"])
        retrieved = retrieval_system.search(query["query_text"], top_k=k)
        batch_results.append(
            {
                "query_id": query["query_id"],
                "query_text": query["query_text"],
                "relevant_ids": relevant_ids,
                "retrieved": retrieved,
            }
        )

    return batch_results


def query_batch_rrf(
    bm25_retrieval_system,
    dense_retrieval_system,
    start_idx: int,
    end_idx: int,
    k: int = 5,
    rrf_k: int = 60,
) -> list[dict]:
    """Query a batch of documents combining systems using RRF."""
    queries = EvalUtils.get_resource_queries(start_idx, end_idx)
    if not isinstance(bm25_retrieval_system, BM25Retriever):
        raise ValueError("Invalid BM25 system. Must be BM25Retriever instance.")
    if not isinstance(dense_retrieval_system, DenseRetriever):
        raise ValueError("Invalid dense system. Must be DenseRetriever instance.")

    batch_results = []
    for query in queries:
        relevant_ids = set(query["query_relevant_resource_ids"])
        bm25_retrieved = bm25_retrieval_system.search(query["query_text"], top_k=k)
        dense_retrieved = dense_retrieval_system.search(query["query_text"], top_k=k)
        
       
        hybrid_retrieved = ReciprocalRankFusion.search(
            bm25_retrieved, dense_retrieved, k=rrf_k
        )

        batch_results.append(
            {
                "query_id": query["query_id"],
                "query_text": query["query_text"],
                "relevant_ids": relevant_ids,
                "retrieved": hybrid_retrieved,
            }
        )

    return batch_results


def evaluate_batch(batch_results: list[dict], k: int = 5) -> dict:
    """Evaluate batch query retrieval data across precision, recall, and MRR."""
    metrics = []
    for item in batch_results:
        precision = RetrieverEvals.precision(item["retrieved"], item["relevant_ids"], k=k)
        recall = RetrieverEvals.recall(item["retrieved"], item["relevant_ids"], k=k)
        mrr = RetrieverEvals.mean_reciprocal_rank(item["retrieved"], item["relevant_ids"], k=k)
        metrics.append(
            {
                "query_id": item["query_id"],
                "precision": precision,
                "recall": recall,
                "mrr": mrr,
            }
        )

    n_metrics = len(metrics)
    if n_metrics == 0:
        return {"per_query": [], "precision": 0.0, "recall": 0.0, "mrr": 0.0}


    mean_precision = sum(m["precision"] for m in metrics) / n_metrics
    mean_recall = sum(m["recall"] for m in metrics) / n_metrics
    mean_mrr = sum(m["mrr"] for m in metrics) / n_metrics

    return {
        "per_query": metrics,
        "precision": mean_precision,
        "recall": mean_recall,
        "mrr": mean_mrr,
    }


def bm25_tests():
    """Execute evaluation validation pipeline tests for BM25 system."""
    bm25 = BM25Retriever()
    resource_texts = EvalUtils.get_resource_texts()
    for text in resource_texts:
        bm25.add_document(text["idx"], text["extracted_text"])

    test_bm25_batch = query_batch(bm25, 10, 15, k=5)
    test_bm25_metrics = evaluate_batch(test_bm25_batch, k=5)

    print("Per-query metrics:")
    for metric in test_bm25_metrics["per_query"]:
        print(metric)


def dense_tests():
    """Execute evaluation validation pipeline tests for Dense system."""
    resource_texts = EvalUtils.get_resource_texts()

    dense = DenseRetriever()
    dense.embed_documents(resource_texts)
    dense.save_embeddings()

    test_dense_batch = query_batch(dense, 10, 15, k=5)
    test_dense_metrics = evaluate_batch(test_dense_batch, k=5)

    print("Per-query metrics:")
    for metric in test_dense_metrics["per_query"]:
        print(metric)


def hybrid_tests():
    """Execute comparative pipeline metrics across all engine variations."""
    resource_text = EvalUtils.get_resource_texts()
    bm25 = BM25Retriever()
    for text in resource_text:
        bm25.add_document(text["idx"], text["extracted_text"])

    train_bm25_batch = query_batch(bm25, 10, 15, k=5)
    train_bm25_metrics = evaluate_batch(train_bm25_batch, k=5)

    dense = DenseRetriever()
    dense.embed_documents(resource_text)
    train_dense_batch = query_batch(dense, 10, 15, k=5)
    train_dense_metrics = evaluate_batch(train_dense_batch, k=5)

    train_hybrid_batch = query_batch_rrf(bm25, dense, 10, 15, k=5, rrf_k=60)
    train_hybrid_metrics = evaluate_batch(train_hybrid_batch, k=5)

    print("Per-query bm25 metrics:")
    for metric in train_bm25_metrics["per_query"]:
        print(metric)

    print("\nPer-query dense metrics:")
    for metric in train_dense_metrics["per_query"]:
        print(metric)

    print("\nPer-query hybrid metrics:")
    for metric in train_hybrid_metrics["per_query"]:
        print(metric)

    comparison = compare_systems(
        train_bm25_metrics["per_query"],
        train_dense_metrics["per_query"],
        train_hybrid_metrics["per_query"],
    )
    print_comparison(comparison)


if __name__ == "__main__":
    print("Running BM25 tests...")
    bm25_tests()
    print("\nRunning Dense tests...")
    dense_tests()
    print("\nRunning Hybrid tests...")
    hybrid_tests()
