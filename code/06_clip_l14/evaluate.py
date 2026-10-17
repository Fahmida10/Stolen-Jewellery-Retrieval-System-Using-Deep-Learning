"""
Evaluation script for jewelry retrieval system
Computes retrieval metrics: Recall@K, mAP, etc.
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from config import Config
from utils import compute_cosine_similarity, get_top_k_matches


def compute_recall_at_k(query_embeddings, query_labels, database_embeddings, database_labels, k_values=[1, 5, 10, 20]):
    """
    Compute Recall@K for retrieval
    
    Args:
        query_embeddings: Query embeddings [N_q, D]
        query_labels: Query labels [N_q]
        database_embeddings: Database embeddings [N_d, D]
        database_labels: Database labels [N_d]
        k_values: List of K values to compute recall for
    
    Returns:
        recall_dict: Dictionary of Recall@K values
    """
    query_embeddings = torch.from_numpy(query_embeddings)
    database_embeddings = torch.from_numpy(database_embeddings)
    
    # Compute similarity matrix
    similarity_matrix = compute_cosine_similarity(query_embeddings, database_embeddings)
    
    # Get top-k indices for each query
    recall_dict = {}
    
    for k in k_values:
        correct = 0
        total = len(query_embeddings)
        
        for i in range(total):
            # Get top-k matches
            top_k_scores, top_k_indices = torch.topk(similarity_matrix[i], k=min(k, len(database_embeddings)))
            top_k_labels = [database_labels[idx] for idx in top_k_indices.cpu().numpy()]
            
            # Check if any of the top-k matches has the same label as query
            if query_labels[i] in top_k_labels:
                correct += 1
        
        recall = correct / total
        recall_dict[f'Recall@{k}'] = recall
        print(f"Recall@{k}: {recall:.4f} ({correct}/{total})")
    
    return recall_dict


def compute_mean_average_precision(query_embeddings, query_labels, database_embeddings, database_labels):
    """
    Compute Mean Average Precision (mAP)
    
    Args:
        query_embeddings: Query embeddings
        query_labels: Query labels
        database_embeddings: Database embeddings
        database_labels: Database labels
    
    Returns:
        mAP: Mean Average Precision
    """
    query_embeddings = torch.from_numpy(query_embeddings)
    database_embeddings = torch.from_numpy(database_embeddings)
    
    # Compute similarity matrix
    similarity_matrix = compute_cosine_similarity(query_embeddings, database_embeddings)
    
    average_precisions = []
    
    for i in range(len(query_embeddings)):
        # Sort by similarity (descending)
        sorted_indices = torch.argsort(similarity_matrix[i], descending=True).cpu().numpy()
        sorted_labels = [database_labels[idx] for idx in sorted_indices]
        
        # Compute average precision for this query
        relevant_count = 0
        precision_sum = 0.0
        
        for rank, label in enumerate(sorted_labels, 1):
            if label == query_labels[i]:
                relevant_count += 1
                precision_at_k = relevant_count / rank
                precision_sum += precision_at_k
        
        if relevant_count > 0:
            avg_precision = precision_sum / relevant_count
            average_precisions.append(avg_precision)
    
    mAP = np.mean(average_precisions) if average_precisions else 0.0
    print(f"Mean Average Precision (mAP): {mAP:.4f}")
    
    return mAP


def compute_classification_accuracy(embeddings, labels, num_classes=4):
    """
    Compute classification accuracy using nearest centroid classifier
    
    Args:
        embeddings: Embeddings
        labels: Labels
        num_classes: Number of classes
    
    Returns:
        accuracy: Classification accuracy
    """
    embeddings = torch.from_numpy(embeddings)
    
    # Compute class centroids
    centroids = []
    for c in range(num_classes):
        class_mask = labels == c
        if class_mask.sum() > 0:
            centroid = embeddings[class_mask].mean(dim=0)
            centroids.append(centroid)
        else:
            centroids.append(torch.zeros_like(embeddings[0]))
    
    centroids = torch.stack(centroids)
    
    # Compute similarity to centroids
    similarity = compute_cosine_similarity(embeddings, centroids)
    predictions = torch.argmax(similarity, dim=1).numpy()
    
    accuracy = (predictions == labels).sum() / len(labels)
    print(f"Classification Accuracy (Nearest Centroid): {accuracy:.4f}")
    
    return accuracy, predictions


def plot_confusion_matrix(labels, predictions, categories, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_embedding_space(embeddings, labels, categories, save_path):
    """Plot embedding space using t-SNE"""
    from sklearn.manifold import TSNE
    
    # Use t-SNE to reduce to 2D
    print("Computing t-SNE projection...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    for i, category in enumerate(categories):
        mask = labels == i
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   label=category, alpha=0.6, s=50)
    
    plt.legend()
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('Embedding Space Visualization (t-SNE)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Embedding visualization saved to {save_path}")
    plt.close()


def evaluate_retrieval(config):
    """Main evaluation function"""
    print("="*80)
    print("JEWELRY RETRIEVAL SYSTEM - EVALUATION")
    print("="*80)
    
    # Load embeddings
    print("\nLoading embeddings...")
    
    # Load test embeddings
    test_path = os.path.join(config.EMBEDDING_DIR, 'test_embeddings.pkl')
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)
    test_embeddings = test_data['embeddings']
    test_labels = test_data['labels']
    categories = test_data['metadata']['categories']
    
    print(f"Test set: {len(test_embeddings)} samples")
    
    # Load validation embeddings
    val_path = os.path.join(config.EMBEDDING_DIR, 'val_embeddings.pkl')
    with open(val_path, 'rb') as f:
        val_data = pickle.load(f)
    val_embeddings = val_data['embeddings']
    val_labels = val_data['labels']
    
    print(f"Validation set: {len(val_embeddings)} samples")
    
    # Evaluation 1: Retrieval Performance (Test queries on Test database)
    print("\n" + "="*80)
    print("1. RETRIEVAL PERFORMANCE (Test set)")
    print("="*80)
    
    recall_dict = compute_recall_at_k(
        test_embeddings, test_labels,
        test_embeddings, test_labels,
        k_values=[1, 5, 10, 20]
    )
    
    mAP = compute_mean_average_precision(
        test_embeddings, test_labels,
        test_embeddings, test_labels
    )
    
    # Evaluation 2: Classification Performance
    print("\n" + "="*80)
    print("2. CLASSIFICATION PERFORMANCE (Test set)")
    print("="*80)
    
    accuracy, predictions = compute_classification_accuracy(
        test_embeddings, test_labels, num_classes=len(categories)
    )
    
    # Classification report
    print("\nClassification Report:")
    # Handle category mismatch (test might have different categories than training)
    unique_labels = sorted(np.unique(np.concatenate([test_labels, predictions])))
    if len(unique_labels) <= len(categories):
        # Use available categories
        report_categories = [categories[i] if i < len(categories) else f"Class_{i}" for i in unique_labels]
        print(classification_report(test_labels, predictions, labels=unique_labels, target_names=report_categories))
    else:
        print(classification_report(test_labels, predictions))
    
    # Plot confusion matrix
    cm_path = os.path.join(config.RESULTS_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(test_labels, predictions, categories, cm_path)
    
    # Evaluation 3: Cross-dataset retrieval (Val queries on Test database)
    print("\n" + "="*80)
    print("3. CROSS-DATASET RETRIEVAL (Val queries -> Test database)")
    print("="*80)
    
    cross_recall_dict = compute_recall_at_k(
        val_embeddings, val_labels,
        test_embeddings, test_labels,
        k_values=[1, 5, 10, 20]
    )
    
    # Evaluation 4: Embedding space visualization
    print("\n" + "="*80)
    print("4. EMBEDDING SPACE VISUALIZATION")
    print("="*80)
    
    vis_path = os.path.join(config.RESULTS_DIR, 'embedding_space.png')
    plot_embedding_space(test_embeddings, test_labels, categories, vis_path)
    
    # Save evaluation results
    results = {
        'recall': recall_dict,
        'mAP': mAP,
        'classification_accuracy': accuracy,
        'cross_dataset_recall': cross_recall_dict
    }
    
    results_path = os.path.join(config.RESULTS_DIR, 'evaluation_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nEvaluation results saved to {results_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Test Set Size: {len(test_embeddings)}")
    print(f"Embedding Dimension: {test_embeddings.shape[1]}")
    print(f"\nRetrieval Performance:")
    for k, v in recall_dict.items():
        print(f"  {k}: {v:.4f}")
    print(f"  mAP: {mAP:.4f}")
    print(f"\nClassification Accuracy: {accuracy:.4f}")
    print("="*80)


def main():
    """Main function"""
    config = Config()
    
    # Check if embeddings exist
    if not os.path.exists(os.path.join(config.EMBEDDING_DIR, 'test_embeddings.pkl')):
        print("Error: Embeddings not found!")
        print("Please run extract_embeddings.py first")
        return
    
    # Run evaluation
    evaluate_retrieval(config)


if __name__ == "__main__":
    main()

