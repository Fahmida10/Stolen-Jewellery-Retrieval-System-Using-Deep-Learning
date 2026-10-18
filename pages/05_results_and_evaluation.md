# Section 5. Results and evaluation

**Section 5** of **11**. **Index:** [Contents](README.md) | **Previous:** [Section 4](04_implementation.md) | **Next:** [Section 6](06_management.md)

This document is the **portfolio chapter for empirical findings**: it states the **evaluation protocol**, reproduces the **headline metric tables**, and summarises **issues and mitigations**. Nothing listed below replaces the **technical report**; rather, the report’s **Results** chapter remains the **authoritative** source for **extended tables**, **class-wise analysis**, **retrieval proof** (Top-K filenames and scores), **inference latency**, **overfitting diagnostics**, **confusion-matrix and t-SNE comparisons**, and **pipeline-integrity** discussion. Continue reading in the report from [**Results**](../documents/README_DETAILED_REPORT.md#results) through [**Discussion**](../documents/README_DETAILED_REPORT.md#discussion), [**Limitations**](../documents/README_DETAILED_REPORT.md#limitations), and [**Conclusion**](../documents/README_DETAILED_REPORT.md#conclusion).

## On this page

- [5.1 Evaluation protocol](#51-evaluation-protocol)
- [5.2 Functional verification](#52-functional-verification)
- [5.3 User-centred evaluation](#53-user-centred-evaluation)
- [5.4 Quantitative results](#54-quantitative-results)
- [5.5 Issues identified](#55-issues-identified)
- [5.6 Improvements and outlook](#56-improvements-and-outlook)

## 5.1 Evaluation protocol

All models were trained for **100 epochs** unless early stopping via **validation-based checkpoint selection** intervened. **Primary metrics** are **mean average precision (mAP)** and **nearest-centroid embedding accuracy**. **Recall@K** is reported under a **validation-to-test** retrieval protocol. **Supplementary analyses** in the technical report include **class-wise** performance, **confusion matrices**, **embedding-space visualisation**, and **train–validation divergence** diagnostics.

## 5.2 Functional verification

Each stage exposes an **`evaluate.py`** entry point that consumes **frozen embeddings** produced by **`extract_embeddings.py`**, thereby ensuring that **reported numbers** correspond to a **reproducible pipeline**. The final stage additionally documents a **retrieval exemplar** in which a **fixed query image** yields a **Top-five** neighbour list with **cosine similarities** (see the report’s **Results** subsection on retrieval proof and the figure [`retrieval_proof_top5.png`](../images/retrieval_proof_top5.png)).

## 5.3 User-centred evaluation

The present submission does **not** include a separate **human-factors study**; claims are therefore limited to **algorithmic** performance on the **specified protocol**.

## 5.4 Quantitative results

The following tables reproduce the **headline quantitative outcomes** documented in the [technical report — Results](../documents/README_DETAILED_REPORT.md#results).

| Model | Epochs | mAP | Embedding accuracy |
|---|---:|---:|---:|
| ResNet18 (single-phase) | 100 | 0.6860 | 0.8086 |
| VGG16 | 100 | 0.6817 | 0.8810 |
| EfficientNet-B0 | 100 | 0.8966 | 0.9257 |
| ResNet18 (two-phase) | 100 | 0.6617 | 0.9033 |
| OpenCLIP ViT-B/32 (CE-only) | 100 | 0.3267 | 0.4015 |
| OpenCLIP ViT-L/14 (final) | 100 | **0.9235** | **0.9368** |

| Model | R@1 | R@5 | R@10 | R@20 |
|---|---:|---:|---:|---:|
| ResNet18 (single-phase) | 0.8135 | 0.9452 | 0.9717 | 0.9817 |
| VGG16 | 0.8647 | 0.9570 | 0.9799 | 0.9890 |
| EfficientNet-B0 | 0.9223 | 0.9753 | 0.9845 | 0.9899 |
| ResNet18 (two-phase) | 0.8583 | 0.9790 | **0.9945** | **0.9973** |
| OpenCLIP ViT-B/32 (CE-only) | 0.3455 | 0.7952 | 0.9442 | 0.9890 |
| OpenCLIP ViT-L/14 (final) | **0.9442** | **0.9854** | 0.9890 | 0.9927 |

Figure assets supporting these conclusions include [`results_stagewise_map_acc.png`](../images/results_stagewise_map_acc.png) and [`results_cross_dataset_recall.png`](../images/results_cross_dataset_recall.png).

## 5.5 Issues identified

The **transitional ViT-B/32** model trained with **cross-entropy alone** exhibited **weak class-wise** behaviour relative to **strong CNN baselines**, particularly on **necklace** and **ring** categories. **Overfitting signatures** appeared in **long training runs** on the relatively **clean** catalogue-style imagery. **Directory metadata** unrelated to images briefly compromised **class discovery** until remediated.

## 5.6 Improvements and outlook

The **final ViT-L/14** configuration combines **supervised contrastive** pressure with **cross-entropy** in a **late training phase**, following **stabilisation** of the classification head. **Embedding regeneration** after **pipeline fixes** restored **internal consistency** among metrics. **Further research directions** are summarised in [Section 10](10_conclusion.md) and in the report’s [future work](../documents/README_DETAILED_REPORT.md#future-work) section.

---

**Navigation:** [Repository home](../README.md) · [Results in report](../documents/README_DETAILED_REPORT.md#results) · [Discussion in report](../documents/README_DETAILED_REPORT.md#discussion)
