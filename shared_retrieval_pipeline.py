"""Shared embedding extraction and retrieval evaluation helpers for all stages."""

import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE

try:
    import seaborn as sns
except Exception:
    sns = None


def _get_model_for_embedding(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def default_embedding_getter(model, images):
    """Extract embeddings using the most compatible interface for a given model."""
    model_ref = _get_model_for_embedding(model)

    if hasattr(model_ref, "get_embeddings"):
        return model_ref.get_embeddings(images)
    if hasattr(model_ref, "base_model"):
        return model_ref.base_model(images)

    outputs = model_ref(images)
    if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        return outputs[0]
    return outputs


def extract_embeddings(model, dataloader, device, desc="Extracting embeddings", embedding_getter=None):
    """Extract embeddings, labels and paths from a dataloader."""
    model.eval()
    embed_fn = embedding_getter or default_embedding_getter

    all_embeddings = []
    all_labels = []
    all_paths = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            images = batch[0].to(device)

            labels = None
            paths = None
            if len(batch) == 3:
                labels = batch[1]
                paths = batch[2]
            elif len(batch) == 2:
                second = batch[1]
                if isinstance(second, (list, tuple)) and len(second) > 0 and isinstance(second[0], str):
                    paths = second
                else:
                    labels = second

            embeddings = embed_fn(model, images)
            all_embeddings.append(embeddings.detach().cpu().numpy())

            if labels is not None:
                all_labels.append(labels.detach().cpu().numpy())

            if paths is not None:
                all_paths.extend(list(paths))

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0) if all_labels else None

    return embeddings, labels, all_paths


def save_embeddings(embeddings, labels, paths, save_path, metadata=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    payload = {
        "embeddings": embeddings,
        "labels": labels,
        "paths": paths,
        "metadata": metadata or {},
    }
    with open(save_path, "wb") as f:
        pickle.dump(payload, f)


def load_embeddings(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_cosine_similarity(query_embeddings, database_embeddings):
    query_norm = F.normalize(query_embeddings, p=2, dim=1)
    database_norm = F.normalize(database_embeddings, p=2, dim=1)
    return torch.matmul(query_norm, database_norm.t())


def compute_recall_at_k(query_embeddings, query_labels, database_embeddings, database_labels, k_values=None):
    if k_values is None:
        k_values = [1, 5, 10, 20]

    query_embeddings = torch.from_numpy(query_embeddings)
    database_embeddings = torch.from_numpy(database_embeddings)
    similarity_matrix = compute_cosine_similarity(query_embeddings, database_embeddings)

    recall_dict = {}
    for k in k_values:
        correct = 0
        total = len(query_embeddings)

        for i in range(total):
            _, top_k_indices = torch.topk(similarity_matrix[i], k=min(k, len(database_embeddings)))
            top_k_labels = [database_labels[idx] for idx in top_k_indices.cpu().numpy()]
            if query_labels[i] in top_k_labels:
                correct += 1

        recall_dict[f"Recall@{k}"] = correct / total if total else 0.0

    return recall_dict


def compute_mean_average_precision(query_embeddings, query_labels, database_embeddings, database_labels):
    query_embeddings = torch.from_numpy(query_embeddings)
    database_embeddings = torch.from_numpy(database_embeddings)
    similarity_matrix = compute_cosine_similarity(query_embeddings, database_embeddings)

    average_precisions = []
    for i in range(len(query_embeddings)):
        sorted_indices = torch.argsort(similarity_matrix[i], descending=True).cpu().numpy()
        sorted_labels = [database_labels[idx] for idx in sorted_indices]

        relevant_count = 0
        precision_sum = 0.0
        for rank, label in enumerate(sorted_labels, 1):
            if label == query_labels[i]:
                relevant_count += 1
                precision_sum += relevant_count / rank

        if relevant_count > 0:
            average_precisions.append(precision_sum / relevant_count)

    return float(np.mean(average_precisions)) if average_precisions else 0.0


def compute_classification_accuracy(embeddings, labels, num_classes):
    embeddings = torch.from_numpy(embeddings)

    centroids = []
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            centroids.append(embeddings[mask].mean(dim=0))
        else:
            centroids.append(torch.zeros_like(embeddings[0]))

    centroids = torch.stack(centroids)
    similarity = compute_cosine_similarity(embeddings, centroids)
    predictions = torch.argmax(similarity, dim=1).numpy()
    accuracy = (predictions == labels).sum() / len(labels)
    return float(accuracy), predictions


def plot_confusion_matrix(labels, predictions, categories, save_path):
    cm = confusion_matrix(labels, predictions)

    plt.figure(figsize=(10, 8))
    if sns is not None:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=categories, yticklabels=categories)
    else:
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xticks(range(len(categories)), categories, rotation=45, ha="right")
        plt.yticks(range(len(categories)), categories)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_embedding_space(embeddings, labels, categories, save_path):
    if len(embeddings) < 2:
        return

    perplexity = min(30, max(2, len(embeddings) // 10))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(12, 10))
    for i, category in enumerate(categories):
        mask = labels == i
        if np.any(mask):
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=category, alpha=0.6, s=35)

    plt.legend()
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.title("Embedding Space Visualization (t-SNE)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def evaluate_from_embedding_files(embedding_dir, results_dir, stage_name="stage"):
    os.makedirs(results_dir, exist_ok=True)

    test_data = load_embeddings(os.path.join(embedding_dir, "test_embeddings.pkl"))
    val_data = load_embeddings(os.path.join(embedding_dir, "val_embeddings.pkl"))

    test_embeddings = test_data["embeddings"]
    test_labels = test_data["labels"]
    categories = test_data.get("metadata", {}).get("categories", ["Class_0", "Class_1", "Class_2", "Class_3"])

    val_embeddings = val_data["embeddings"]
    val_labels = val_data["labels"]

    recall = compute_recall_at_k(test_embeddings, test_labels, test_embeddings, test_labels)
    mAP = compute_mean_average_precision(test_embeddings, test_labels, test_embeddings, test_labels)

    cls_acc, predictions = compute_classification_accuracy(
        test_embeddings,
        test_labels,
        num_classes=len(categories),
    )

    cross_recall = compute_recall_at_k(val_embeddings, val_labels, test_embeddings, test_labels)

    plot_confusion_matrix(
        test_labels,
        predictions,
        categories,
        os.path.join(results_dir, "confusion_matrix.png"),
    )
    plot_embedding_space(
        test_embeddings,
        test_labels,
        categories,
        os.path.join(results_dir, "embedding_space.png"),
    )

    report = classification_report(test_labels, predictions, target_names=categories, zero_division=0)

    results = {
        "stage": stage_name,
        "test_set_size": int(len(test_embeddings)),
        "embedding_dim": int(test_embeddings.shape[1]),
        "recall": recall,
        "mAP": mAP,
        "classification_accuracy": cls_acc,
        "cross_dataset_recall": cross_recall,
        "classification_report": report,
    }

    with open(os.path.join(results_dir, "evaluation_results.pkl"), "wb") as f:
        pickle.dump(results, f)

    summary_lines = [
        f"Stage: {stage_name}",
        f"Test Set Size: {len(test_embeddings)}",
        f"Embedding Dimension: {test_embeddings.shape[1]}",
        "",
        "Retrieval Performance:",
    ]
    for key, value in recall.items():
        summary_lines.append(f"{key}: {value:.4f}")
    summary_lines.extend([
        f"mAP: {mAP:.4f}",
        "",
        f"Classification Accuracy (Nearest Centroid): {cls_acc:.4f}",
        "",
        "Cross-Dataset Retrieval (Val -> Test):",
    ])
    for key, value in cross_recall.items():
        summary_lines.append(f"{key}: {value:.4f}")
    summary_lines.extend(["", "Classification Report:", report])

    with open(os.path.join(results_dir, "evaluation_summary.txt"), "w") as f:
        f.write("\n".join(summary_lines))

    return results
