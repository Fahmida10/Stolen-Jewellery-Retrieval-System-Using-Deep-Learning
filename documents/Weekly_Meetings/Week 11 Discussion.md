### Week 11 – Final training and performance improvement

**Date:** 07/04/2026 (Zoom)
**Attendance:** All five members

**Objective**
To execute the final multi-phase training strategy and optimise retrieval performance using the OpenCLIP ViT-L/14 model.

**Tasks Completed**

* Executed multi-phase training (freeze → unfreeze → refinement)
* Applied combined objectives (cross-entropy + contrastive learning)
* Generated final L2-normalised embeddings for retrieval
* Evaluated performance using mAP and Recall@K

**Previous Learning Applied**
From Week 9–10, it was established that cross-entropy alone is insufficient for retrieval. This stage implemented a retrieval-aware objective to improve embedding structure while retaining classification guidance.

**Discussion**
The group observed that contrastive learning improved intra-class compactness and inter-class separation in the embedding space. Combining objectives led to more stable optimisation and better alignment with similarity-based retrieval goals.

**Challenges Faced**

* Balancing multiple loss functions during training
* Ensuring stable convergence across phases
* Interpreting trade-offs between metrics (mAP vs Recall@K)

**Decisions**

* The combined loss (CE + contrastive) was retained as the final training objective
* Multi-phase training was adopted as the standard procedure for the final model
* The ViT-L/14 configuration was confirmed as the final system for evaluation

**Outcome**

By the end of Week 11, the final model achieved the strongest retrieval performance observed in the project, outperforming all CNN baselines.

The results confirmed that both model capacity and training objective design are critical for high-quality retrieval embeddings.
