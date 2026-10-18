# Section 2. Literature review

**Section 2** of 11. **Index:** [Contents](README.md) | **Previous:** [Section 1](01_overview.md) | **Next:** [Section 3](03_methodology.md)

## 2.1 Introduction

The literature relevant to this project spans **fine-grained visual retrieval**, **domain-specific ornament and jewellery datasets**, **adaptation of CLIP-style models for image-to-image retrieval**, **metric-learning objectives**, and **evaluation practice** under **distribution shift** and **data leakage** constraints. The portfolio does not reproduce the full survey; instead, it highlights **representative sources** and defers the **complete bibliography** to the [Technical Report](../documents/README_DETAILED_REPORT.md#related-work).

## 2.2 Annotated sources (illustrative)

### Source 1

**Reference:** Nayeem, M.D., Sraboni, S.Z., Biswas, S.S. and Islam, M.M. (2025) ‘OrnAsia: A dataset of asian ornaments for image classification and cultural identification’, Data in Brief, 63, p. 112195. doi: https://doi.org/10.1016/j.dib.2025.112195

**Summary.** The article introduces a **curated ornament dataset** with realistic variability in viewpoint and illumination, supporting **fine-grained** analysis at modest scale.

**Relevance.** The present work treats **limited, ornament-adjacent imagery** as a methodological constraint analogous to that discussed in OrnAsia, and therefore emphasises **rigorous evaluation** rather than reliance on a single aggregate accuracy figure.

### Source 2

**Reference:** Alcalde-Llergo, J.M., Ruiz-Mezcua, A., Avila-Ramirez, R., Zingoni, A., Taborri, J. and Yeguas-Bolivar, E. (2025) ‘Automatic identification and description of jewelry through computer vision and neural networks for translators and interpreters’, Applied Sciences, 15(10), p. 5538. doi: https://doi.org/10.3390/app15105538

**Summary.** The authors propose a neural network-based approach for automatic jewellery identification and description using image captioning techniques. The model generates natural language descriptions at multiple levels of detail, enabling structured and interpretable understanding of jewellery items.

**Relevance:** The retrieval-first formulation adopted here (Top-K lists with **traceable identifiers**) responds to the same **accountability** imperative, albeit in a retrieval rather than captioning task.

### Source 3

**Reference:** Schall, K., Barthel, K.U., Hezel, N. and Jung, K. (2024) Optimizing CLIP models for image retrieval with maintained joint-embedding alignment. arXiv preprint. Available at: https://arxiv.org/abs/2409.01936
(Accessed: 1 February 2026).

**Summary.** The paper studies **fine-tuning strategies** that improve **image retrieval** while preserving desirable properties of the **joint embedding**.

**Relevance.** The project’s **final-stage training design** (progressive unfreezing and **retrieval-aware** objectives on OpenCLIP) is conceptually aligned with the observation that **naïve** fine-tuning can degrade **instance-level** retrieval geometry.

### Source 4

**Reference:** K. Schall, K. U. Barthel, N. Hezel, and A. Moelle, “A comprehensive approach to improving CLIP-based image retrieval while maintaining joint-embedding alignment,” Information Systems, 2025. Available: https://www.sciencedirect.com/science/article/pii/S0306437925000651

**Summary.** The paper presents strategies to improve CLIP-based retrieval performance while preserving the alignment between visual and semantic embeddings.

**Relevance.** This supports the project’s use of OpenCLIP and careful fine-tuning, ensuring that retrieval improvements do not distort the embedding space required for similarity-based ranking.

### Source 5

**Reference:** Fahes, M., Vu, T.-H., Bursuc, A., Perez, P. and de Charette, R. (2026) Fine-tuning CLIP’s last visual projector: A few-shot cornucopia. arXiv preprint. Available at: https://arxiv.org/abs/2410.05270
(Accessed: 1 February 2026).

**Summary.** The study explores efficient fine-tuning of CLIP under limited data conditions, focusing on adapting specific layers to improve performance.

**Relevance.** While few-shot techniques were not directly implemented in this project, the study highlights strategies for adapting CLIP under limited data scenarios. This supports the project’s use of controlled fine-tuning and staged training to improve embedding quality on a small jewellery dataset.

### Source 6

**Reference:** Agarwal, A., Karanam, S. and Gandhi, V. (2026) LiteEmbed: Adapting CLIP to rare classes. arXiv preprint. Available at: https://arxiv.org/abs/2601.09661 (Accessed: 1 February 2026).

**Summary.** The paper proposes lightweight adaptation techniques to improve CLIP performance on rare or underrepresented classes.

**Relevance.** This is relevant to the project’s small and imbalanced dataset, supporting the need for careful model adaptation when training data is limited.

## Source 7

**Reference:** eighami, S., Wellmer, Z. and Parameswaran, A. (2024) NUDGE: Lightweight non-parametric fine-tuning of embeddings for retrieval. arXiv preprint. Available at: https://arxiv.org/abs/2409.02343 (Accessed: 1 February 2026).

**Summary.** The work introduces efficient methods for improving retrieval embeddings without heavy retraining.

**Relevance.** This supports the importance of embedding quality in retrieval tasks and reinforces the project’s focus on improving embedding representations rather than only classification accuracy.

## Source 8

**Reference:** Gao, C., Xue, S., Peng, Y., Fu, J., Gu, T., Li, S. and Zhou, F. (2026) LookBench: A live and holistic open benchmark for fashion image retrieval. arXiv preprint. Available at: https://arxiv.org/abs/2601.14706 (Accessed: 3 February 2026).

**Summary.** The paper introduces a benchmark for evaluating fashion and product-based image retrieval systems under realistic conditions.

**Relevance.** This reinforces the importance of robust evaluation protocols, aligning with the project’s use of mAP and Recall@K for assessing retrieval performance.


## 2.3 Synthesis

The **related work** chapter of the [technical report](../documents/README_DETAILED_REPORT.md#related-work) integrates the above themes with **benchmark** and **composed-retrieval** literature, and explains how those works informed **split design**, **loss composition**, and **diagnostic reporting**. The reader is referred to that chapter for the **complete argumentation** and for **citations [1]–[23]**.

---

**Navigation:** [Repository home](../README.md) · [Related work in report](../documents/README_DETAILED_REPORT.md#related-work) · [References in report](../documents/README_DETAILED_REPORT.md#references)
