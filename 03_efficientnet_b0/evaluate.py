"""Standard retrieval-style evaluation for Stage 1 embeddings."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from shared_retrieval_pipeline import evaluate_from_embedding_files


def main():
    required = [
        os.path.join(EMBEDDING_DIR, "test_embeddings.pkl"),
        os.path.join(EMBEDDING_DIR, "val_embeddings.pkl"),
    ]

    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        print("Missing embeddings:")
        for path in missing:
            print(f"  - {path}")
        print("Run extract_embeddings.py first.")
        raise SystemExit(1)

    results = evaluate_from_embedding_files(EMBEDDING_DIR, RESULTS_DIR, stage_name=STAGE_NAME)

    print("\n" + "=" * 80)
    print(f"{STAGE_NAME} - EVALUATION SUMMARY")
    print("=" * 80)
    for key, value in results["recall"].items():
        print(f"{key}: {value:.4f}")
    print(f"mAP: {results['mAP']:.4f}")
    print(f"Classification Accuracy: {results['classification_accuracy']:.4f}")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
