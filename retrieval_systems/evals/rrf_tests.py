from hybrid.reciprocal_rank_fusion import rrf


def test_no_overlap():
    """
    Category: No overlap at all.
    Every doc_id in bm25 is disjoint from every doc_id in dense.
    Probes: does RRF handle disjoint sets without crashing, and does the
    final sort order make sense when nothing gets summed?
    """
    bm25_results = [("DocA", 0.0), ("DocB", 0.0)]
    dense_results = [("DocX", 0.0), ("DocY", 0.0)]
    k = 60

    result = rrf(bm25_results, dense_results, k)
    print("test_no_overlap:", result)
    return result


def test_complete_overlap():
    """
    Category: Complete overlap, different order in each list.
    Both lists contain the exact same doc_ids.
    Probes: rank indexing correctness (off-by-one bugs surface clearly here
    since every doc gets two contributions and totals are sensitive to
    whether rank starts at 0 or 1).
    """
    bm25_results = [("DocA", 0.0), ("DocB", 0.0), ("DocC", 0.0)]
    dense_results = [("DocC", 0.0), ("DocA", 0.0), ("DocB", 0.0)]
    k = 60

    result = rrf(bm25_results, dense_results, k)
    print("test_complete_overlap:", result)
    return result


def test_empty_bm25():
    """
    Category: One retriever returns nothing.
    Probes: does the function run without error on an empty list, and does
    the output reduce to dense's results re-scored by RRF's formula
    (not dense's original raw scores)?
    """
    bm25_results = []
    dense_results = [("DocA", 0.0), ("DocB", 0.0)]
    k = 60

    result = rrf(bm25_results, dense_results, k)
    print("test_empty_bm25:", result)
    return result


def test_empty_dense():
    """
    Category: The other retriever returns nothing (mirror of above --
    worth testing both directions in case the two loops were implemented
    asymmetrically by accident).
    """
    bm25_results = [("DocA", 0.0), ("DocB", 0.0)]
    dense_results = []
    k = 60

    result = rrf(bm25_results, dense_results, k)
    print("test_empty_dense:", result)
    return result


def test_both_empty():
    """
    Category: Degenerate case, both retrievers return nothing.
    Probes: does the function crash, or does it correctly return an empty list?
    """
    bm25_results = []
    dense_results = []
    k = 60

    result = rrf(bm25_results, dense_results, k)
    print("test_both_empty:", result)
    return result


def test_single_element():
    """
    Category: Simplest possible non-trivial case.
    Good for isolating the core formula if a more complex test fails --
    if this one is wrong, the bug is in the formula itself, not in
    list-handling edge cases.
    """
    bm25_results = [("DocA", 0.0)]
    dense_results = [("DocA", 0.0)]
    k = 60

    result = rrf(bm25_results, dense_results, k)
    print("test_single_element:", result)
    return result


def test_rank1_vs_rank2_tradeoff():
    """
    Category: 'Obviously relevant' doc doesn't necessarily win.
    DocA ranks #1 in bm25 but is entirely absent from dense.
    DocB ranks #2 in BOTH bm25 and dense.
    Probes intuition: does RRF favor a single strong top-1 signal, or
    consistent moderate agreement across both retrievers? Work this out by
    hand BEFORE running -- your intuition might be wrong, and that's the
    point of this test (understanding RRF's actual behavior, not just
    checking for bugs).
    """
    bm25_results = [("DocA", 0.0), ("DocB", 0.0)]
    dense_results = [("DocC", 0.0), ("DocB", 0.0)]
    k = 60

    result = rrf(bm25_results, dense_results, k)
    print("test_rank1_vs_rank2_tradeoff:", result)
    return result


def test_duplicate_within_same_list():
    """
    Category: A doc_id appears twice within the SAME retriever's result list
    (shouldn't happen if search() is correct, but worth checking your RRF
    function doesn't do something weird if it does -- defensive testing).
    """
    bm25_results = [("DocA", 0.0), ("DocA", 0.0), ("DocB", 0.0)]
    dense_results = [("DocA", 0.0)]
    k = 60

    result = rrf(bm25_results, dense_results, k)
    print("test_duplicate_within_same_list:", result)
    return result


if __name__ == "__main__":
    test_no_overlap()
    test_complete_overlap()
    test_empty_bm25()
    test_empty_dense()
    test_both_empty()
    test_single_element()
    test_rank1_vs_rank2_tradeoff()
    test_duplicate_within_same_list()