# Section 4. Implementation and system architecture

**Section 4** of 11. **Index:** [Contents](README.md) | **Previous:** [Section 3](03_methodology.md) | **Next:** [Section 5](05_results_and_evaluation.md)

## 4.1 Architectural overview

The retrieval system separates **offline indexing** from **online query processing**. During indexing, each gallery image is mapped to a **dense vector representation**; at query time, the **query image** is encoded with the **same encoder**, **L2-normalised**, and compared to the gallery by **cosine similarity**. The **K** largest similarities define the ranked output, which includes **filenames** and **scores** for audit purposes. The diagrams below summarise the **pipeline** and the **training-time** configuration of the final encoder.

![Unified retrieval pipeline](../images/unified_retrieval_pipeline.png)

![Final model architecture in training mode](../images/final_model_architecture.png)

## 4.2 Principal software components

The implementation is organised into **six** numerically ordered directories under `code/`, each corresponding to one experimental stage, together with a **shared utility module** (`shared_retrieval_pipeline.py`). The **final** stage extends the common pattern with **retrieval-specific** modules and an **inference** script suitable for **demonstration**. A concise index is maintained in [`code/README.md`](../code/README.md).

## 4.3 Repository layout (implementation artefacts)

| Location | Function |
|---|---|
| `code/01_resnet18`–`06_clip_l14` | Configuration, data loading, model definition, training, embedding extraction, and evaluation |
| `code/06_clip_l14/inference.py` | Top-K retrieval with optional visualisation |
| `code/06_clip_l14/retrieval.py` | Retrieval utilities supporting evaluation |

## 4.4 Visual evidence

Figures supporting the thesis including **dataset overviews**, **class-distribution** plots, **architecture schematics** for each family, **quantitative comparison** charts, **retrieval proofs**, **confusion-matrix** comparisons, and **t-SNE** embeddings are stored under [`images/`](../images/README.md). Stage-specific exports use a **consistent filename prefix** (`stage01_`–`stage06_`) to avoid ambiguity.

## 4.5 Attribution and provenance

**Pretrained parameters** are obtained through **torchvision** (CNNs) and **OpenCLIP** (ViT-B/32 and ViT-L/14) according to the checkpoint documentation cited in the technical report. **Third-party libraries** are listed in the per-stage **requirements** files. The **dataset** resides outside the repository; only **relative path configuration** is committed. **Binary artefacts** (weights, caches, embeddings) are **excluded** from version control in line with [`SUBMISSION_NOTES.md`](../documents/SUBMISSION_NOTES.md).

## 4.6 Challenges and mitigations encountered in implementation

**Fine-tuning stability** on a modest dataset required **curriculum-style** optimisation in the final stage. **Equivalence of evaluation** across stages required **strict control** of augmentation and normalisation. **Filesystem metadata** unrelated to class labels posed a **subtle risk** to label indexing; mitigation consisted of **explicit filtering** and **regeneration** of derived embeddings once the issue was identified.

---

**Navigation:** [Repository home](../README.md) · [Methodology in report](../documents/README_DETAILED_REPORT.md#methodology) · [Images index](../images/README.md)
