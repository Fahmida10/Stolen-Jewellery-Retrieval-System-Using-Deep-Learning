# Section 1. Project overview and scope

This document constitutes **Section 1** of the portfolio sequence (Sections 1–11). **Index:** [Portfolio contents](README.md) | **Next:** [Section 2. Literature](02_literature.md)

## 1.1 Aim of the project

The project addresses **content-based image retrieval** in a jewellery domain. The operational objective is to retrieve, for a given **query image**, the **K** nearest neighbours from a **gallery** of images labelled as stolen jewellery, together with **metadata sufficient for audit** (including filenames and similarity scores). The investigation compares **convolutional neural network (CNN)** baselines with **OpenCLIP** visual encoders under a **single evaluation protocol**, in order to identify a configuration that optimises **retrieval fidelity** rather than classification accuracy alone.

## 1.2 Problem background

Jewellery instances frequently exhibit **high inter-class visual similarity** at coarse scales, while **intra-class** variation arises from viewpoint, illumination, and subtle geometric differences in settings and stones. Conventional classifiers optimise **decision boundaries** that need not align with a **metric structure** suited to **ranked retrieval**. Recent **vision–language** pretraining yields embeddings in which **cosine similarity** correlates more closely with **semantic proximity**, which motivates the comparative design adopted here.

## 1.3 Intended users and deployment context

The artefacts produced by this work are intended for **analyst-facing** or **investigator-support** settings in which a **single query image** must be matched against a **moderate-sized, curated gallery**, and in which **transparent ranking** (Top-K lists with traceable file identifiers) is required. The scope is **demonstration and evaluation** within the constraints of the available dataset; **enterprise-scale** deployment is not claimed.

## 1.4 Principal technical elements

The repository implements **six** sequentially numbered model packages, spanning **ResNet18**, **VGG16**, **EfficientNet-B0**, a **two-phase ResNet18** curriculum, **OpenCLIP ViT-B/32**, and a **final OpenCLIP ViT-L/14** configuration. **Training, validation, and test** partitions support supervised learning; a **distinct gallery partition** is reserved for retrieval and is **excluded from gradient-based fitting**. The final system combines **multi-phase optimisation** (cross-entropy phases followed by a **supervised contrastive** term together with cross-entropy) with **L2-normalised** embeddings for **cosine** retrieval.

## 1.5 Reported outcomes (summary)

Quantitative results are reported under a **100-epoch** training budget. The selected **OpenCLIP ViT-L/14** model attains the strongest headline **mean average precision (mAP)** and **nearest-centroid embedding accuracy**, together with competitive **Recall@K** on a **validation-to-test** protocol. The headline **metric tables** are in [Section 5 — Results and evaluation](05_results_and_evaluation.md); the **full Results chapter**, extended analyses, and figures are in the [technical report](../documents/README_DETAILED_REPORT.md#results).

## 1.6 Scope and limitations of claims

**In scope** are controlled comparison of architectures, **split governance**, **embedding-space diagnostics**, and **integrity** of the data-ingestion pipeline (including mitigation of spurious directory entries). **Out of scope** are guarantees of **real-time** performance at arbitrary gallery scale and **uniform** generalisation beyond the **catalogue-style** imagery that dominates the present dataset. These limitations are developed analytically in the [discussion and limitations](../documents/README_DETAILED_REPORT.md#discussion) of the technical report.

---

**Navigation:** [Repository home](../README.md) · [Report table of contents](../documents/README_DETAILED_REPORT.md#table-of-contents) · [Documents](../documents/README.md) · [Source code](../code/README.md)
