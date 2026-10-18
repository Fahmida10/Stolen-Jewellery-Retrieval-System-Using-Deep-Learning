# Section 3. Methodology

**Section 3** of 11. **Index:** [Contents](README.md) | **Previous:** [Section 2](02_literature.md) | **Next:** [Section 4](04_implementation.md)

## 3.1 Development approach

The investigation follows a **progressive model-selection** strategy. **Convolutional** baselines establish reference performance under a **shared training budget** and **identical preprocessing rules** wherever architecturally permissible. A **transitional OpenCLIP** configuration isolates the effect of **encoder family** before introducing the **final multi-phase** curriculum on **ViT-L/14**. This ordering reduces confounding between **architecture** and **objective-function** design.

## 3.2 Tools and technologies

Experiments were implemented in **Python** using **PyTorch** and, for the final stages, the **OpenCLIP** library. **Torchvision** supplies ImageNet-pretrained CNNs. Dependency pins are recorded in the `requirements.txt` files under `code/05_clip_b32` and `code/06_clip_l14`. **Mixed-precision** training is enabled where supported by hardware.

## 3.3 System and experimental design

The dataset is partitioned into **training**, **validation**, and **test** subsets for **supervised learning**, together with a **non-overlapping gallery** reserved exclusively for **retrieval evaluation**. This separation is **methodologically necessary** to prevent **information leakage** from gallery images into **parameter updates**. **Input resolution** is fixed at **224×224** RGB; **normalisation constants** follow **ImageNet** statistics for CNN stages and **CLIP** statistics for OpenCLIP stages. **Augmentation** is applied only during training and is deliberately **conservative** in order to preserve **fine detail**.

## 3.4 Workflow

For each numbered code package, the workflow comprises **configuration** of the dataset root, **training**, **embedding extraction**, and **evaluation** scripts executed in that order. The final package additionally exposes **retrieval** and **inference** entry points. **Offline** indexing of the gallery precedes **online** query encoding and **cosine** ranking.

## 3.5 Data description

The technical report documents **3320** training images, **1094** validation images, **538** test images, and a **413-image** retrieval gallery, distributed across four **jewellery categories**. **Class proportions** differ modestly across splits; the evaluation strategy therefore supplements **global** metrics with **class-aware** and **embedding-space** analyses.

## 3.6 Design decisions and risk management

**OpenCLIP ViT-L/14** was selected as the **final encoder** on the basis of **empirical superiority** on **mAP** and **embedding accuracy** under the common protocol. **Risks** addressed include **overfitting** (mitigated by **validation checkpoints**, **weight decay**, and **phased** optimisation), **semantic drift** during fine-tuning (mitigated by **differential learning rates**), and **pipeline integrity** (mitigated by **deterministic indexing** and **filtering** of non-image files during class discovery).

Further methodological detail appears in the [methodology chapter](../documents/README_DETAILED_REPORT.md#methodology) of the technical report.

---

**Navigation:** [Repository home](../README.md) · [Methodology in report](../documents/README_DETAILED_REPORT.md#methodology) · [Source code](../code/README.md)
