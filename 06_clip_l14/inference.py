"""
Inference script for stolen jewelry retrieval
Easy-to-use interface for query and retrieval
"""

import os
import sys
import argparse
from PIL import Image

from config import Config
from retrieval import JewelryRetrievalSystem


def main():
    """Main inference function with CLI"""
    parser = argparse.ArgumentParser(description='Stolen Jewelry Retrieval System')
    parser.add_argument('--query', type=str, required=True,
                       help='Path to query jewelry image')
    parser.add_argument('--top_k', type=int, default=10,
                       help='Number of results to retrieve (default: 10)')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize retrieval results')
    parser.add_argument('--save_viz', type=str, default=None,
                       help='Path to save visualization (if --visualize is set)')
    
    args = parser.parse_args()
    
    # Initialize config
    config = Config()
    
    # Paths
    model_path = os.path.join(config.CHECKPOINT_DIR, config.BEST_MODEL_NAME)
    embeddings_path = os.path.join(config.EMBEDDING_DIR, 'stolen_embeddings.pkl')
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using: python src/train.py")
        return
    
    if not os.path.exists(embeddings_path):
        print(f"Error: Embeddings not found at {embeddings_path}")
        print("Please extract embeddings first using: python src/extract_embeddings.py")
        return
    
    if not os.path.exists(args.query):
        print(f"Error: Query image not found at {args.query}")
        return
    
    # Initialize retrieval system
    print("Initializing Jewelry Retrieval System...")
    retrieval_system = JewelryRetrievalSystem(
        model_path=model_path,
        embeddings_path=embeddings_path,
        config=config
    )
    
    # Perform retrieval
    print(f"\nSearching for matches to: {args.query}")
    print("="*80)
    
    top_k_paths, top_k_scores = retrieval_system.retrieve_top_k(args.query, k=args.top_k)
    
    # Display results
    print(f"\nTop-{args.top_k} Matches:")
    print("-"*80)
    for i, (path, score) in enumerate(zip(top_k_paths, top_k_scores), 1):
        print(f"{i:2d}. Score: {score:.4f} | {os.path.basename(path)}")
        print(f"    Path: {path}")
    
    # Visualize if requested
    if args.visualize:
        save_path = args.save_viz or os.path.join(config.RESULTS_DIR, 'retrieval_result.png')
        print(f"\nGenerating visualization...")
        retrieval_system.visualize_results(args.query, k=args.top_k, save_path=save_path)
    
    print("\n" + "="*80)
    print("Retrieval completed!")


if __name__ == "__main__":
    main()

