"""
Image retrieval system for stolen jewelry identification
Implements Top-K retrieval using cosine similarity
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple
import pickle

from config import Config
from model import build_model
from dataset import get_transforms
from utils import get_top_k_matches


class JewelryRetrievalSystem:
    """
    Jewelry retrieval system for finding stolen jewelry matches
    """
    
    def __init__(self, model_path: str, embeddings_path: str, config: Config, device='cuda'):
        """
        Initialize retrieval system
        
        Args:
            model_path: Path to trained model checkpoint
            embeddings_path: Path to stolen jewelry embeddings
            config: Configuration object
            device: Device to run on
        """
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"Initializing Jewelry Retrieval System on {self.device}")
        
        # Load model
        print("Loading model...")
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Load stolen jewelry embeddings
        print("Loading stolen jewelry database...")
        self.database_embeddings, self.database_paths = self._load_database(embeddings_path)
        print(f"Database size: {len(self.database_embeddings)} items")
        
        # Get transform for query images
        self.transform = get_transforms(config.INPUT_SIZE, is_train=False, use_augmentation=False)
        
        print("Retrieval system ready!")
    
    def _load_model(self, model_path: str):
        """Load trained model"""
        model = build_model(self.config, num_classes=4, use_classifier=True)
        model = model.to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def _load_database(self, embeddings_path: str):
        """Load stolen jewelry embeddings database"""
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
        
        embeddings = torch.from_numpy(data['embeddings']).to(self.device)
        paths = data['paths']
        
        return embeddings, paths
    
    def extract_query_embedding(self, image_path: str) -> torch.Tensor:
        """
        Extract embedding for query image
        
        Args:
            image_path: Path to query image
        
        Returns:
            embedding: Query embedding tensor
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract embedding
        with torch.no_grad():
            if isinstance(self.model, torch.nn.DataParallel):
                if hasattr(self.model.module, 'base_model'):
                    embedding = self.model.module.base_model(image_tensor)
                elif hasattr(self.model.module, 'get_embeddings'):
                    embedding = self.model.module.get_embeddings(image_tensor)
                else:
                    embedding = self.model.module(image_tensor)
            else:
                if hasattr(self.model, 'base_model'):
                    embedding = self.model.base_model(image_tensor)
                elif hasattr(self.model, 'get_embeddings'):
                    embedding = self.model.get_embeddings(image_tensor)
                else:
                    embedding = self.model(image_tensor)
        
        return embedding.squeeze(0)
    
    def retrieve_top_k(self, query_image_path: str, k: int = 10) -> Tuple[List[str], List[float]]:
        """
        Retrieve top-k most similar stolen jewelry items
        
        Args:
            query_image_path: Path to query jewelry image
            k: Number of results to return
        
        Returns:
            top_k_paths: List of paths to top-k matches
            top_k_scores: List of similarity scores
        """
        # Extract query embedding
        query_embedding = self.extract_query_embedding(query_image_path)
        
        # Find top-k matches
        top_k_indices, top_k_scores = get_top_k_matches(
            query_embedding,
            self.database_embeddings,
            k=k,
            metric=self.config.SIMILARITY_METRIC
        )
        
        # Get corresponding paths
        top_k_paths = [self.database_paths[idx] for idx in top_k_indices.cpu().numpy()]
        top_k_scores = top_k_scores.cpu().numpy().tolist()
        
        return top_k_paths, top_k_scores
    
    def visualize_results(self, query_image_path: str, k: int = 10, save_path: str = None):
        """
        Visualize retrieval results
        
        Args:
            query_image_path: Path to query image
            k: Number of results to retrieve
            save_path: Optional path to save visualization
        """
        # Retrieve top-k
        top_k_paths, top_k_scores = self.retrieve_top_k(query_image_path, k)
        
        # Create visualization
        num_cols = min(5, k + 1)
        num_rows = (k + 1 + num_cols - 1) // num_cols
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
        axes = axes.flatten() if k > 0 else [axes]
        
        # Display query image
        query_img = Image.open(query_image_path)
        axes[0].imshow(query_img)
        axes[0].set_title("Query Image", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Display top-k results
        for i, (path, score) in enumerate(zip(top_k_paths, top_k_scores), 1):
            if i < len(axes):
                try:
                    img = Image.open(path)
                    axes[i].imshow(img)
                    axes[i].set_title(f"Rank {i}\nScore: {score:.3f}", fontsize=10)
                    axes[i].axis('off')
                except Exception as e:
                    print(f"Error loading image {path}: {e}")
        
        # Hide unused subplots
        for i in range(k + 1, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def batch_retrieve(self, query_image_paths: List[str], k: int = 10) -> List[Tuple[List[str], List[float]]]:
        """
        Retrieve for multiple query images
        
        Args:
            query_image_paths: List of query image paths
            k: Number of results per query
        
        Returns:
            results: List of (top_k_paths, top_k_scores) for each query
        """
        results = []
        for query_path in query_image_paths:
            top_k_paths, top_k_scores = self.retrieve_top_k(query_path, k)
            results.append((top_k_paths, top_k_scores))
        
        return results


def main():
    """Demo retrieval system"""
    config = Config()
    
    # Paths
    model_path = os.path.join(config.CHECKPOINT_DIR, config.BEST_MODEL_NAME)
    embeddings_path = os.path.join(config.EMBEDDING_DIR, 'stolen_embeddings.pkl')
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using train.py")
        return
    
    if not os.path.exists(embeddings_path):
        print(f"Error: Embeddings not found at {embeddings_path}")
        print("Please extract embeddings first using extract_embeddings.py")
        return
    
    # Initialize retrieval system
    retrieval_system = JewelryRetrievalSystem(
        model_path=model_path,
        embeddings_path=embeddings_path,
        config=config
    )
    
    # Example: Retrieve for a test image
    print("\n" + "="*80)
    print("DEMO: Retrieval System")
    print("="*80)
    
    # Get a random test image
    import glob
    test_images = glob.glob(os.path.join(config.TEST_DIR, "*/*.jpg"))
    
    if test_images:
        query_image = test_images[0]
        print(f"\nQuery image: {query_image}")
        
        # Retrieve top-10
        top_k_paths, top_k_scores = retrieval_system.retrieve_top_k(query_image, k=10)
        
        print("\nTop-10 matches:")
        for i, (path, score) in enumerate(zip(top_k_paths, top_k_scores), 1):
            print(f"{i}. {os.path.basename(path)} - Score: {score:.4f}")
        
        # Visualize
        save_path = os.path.join(config.RESULTS_DIR, 'retrieval_demo.png')
        retrieval_system.visualize_results(query_image, k=10, save_path=save_path)
    else:
        print("No test images found for demo")


if __name__ == "__main__":
    main()

