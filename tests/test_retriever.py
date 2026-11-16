from rag.retriever import rank_chunks


def test_rank_chunks_boosts_crop_rotation_terms():
    chunks = [
        {
            "content": "Crop cover protects soil.",
            "source": "cover.pdf",
            "page": 1,
            "embedding": [1.0, 0.0],
        },
        {
            "content": "Crop rotational practices improve soil fertility.",
            "source": "rotational.pdf",
            "page": 2,
            "embedding": [1.0, 0.0],
        },
        {
            "content": "Crop rotation improves soil health.",
            "source": "rotation.pdf",
            "page": 3,
            "embedding": [1.0, 0.0],
        },
    ]

    results = rank_chunks("crop rotation", [1.0, 0.0], chunks, top_k=3)

    assert [result["source"] for result in results] == [
        "rotation.pdf",
        "rotational.pdf",
        "cover.pdf",
    ]
    assert all(
        abs(result["topic_boost"] - expected) < 1e-9
        for result, expected in zip(results, [0.15, 0.15, 0.0])
    )


def test_rank_chunks_does_not_apply_rotation_boost_to_other_queries():
    chunks = [
        {
            "content": "Crop rotation improves soil health.",
            "source": "rotation.pdf",
            "page": 3,
            "embedding": [1.0, 0.0],
        }
    ]

    results = rank_chunks("soil health", [1.0, 0.0], chunks, top_k=1)

    assert results[0]["topic_boost"] == 0.0
