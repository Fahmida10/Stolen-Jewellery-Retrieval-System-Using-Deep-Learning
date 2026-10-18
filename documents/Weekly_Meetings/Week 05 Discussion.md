### Week 5 – Initial baseline training (ResNet18)

**Date:** 24/02/2026 (Zoom)
**Attendance:** All five members

**Objective**
To begin implementation and training of baseline models and establish an initial performance reference.

**Tasks**

* Implemented the first CNN baseline (ResNet18)
* Trained the model using the prepared dataset and preprocessing pipeline
* Generated feature embeddings for retrieval
* Performed initial evaluation using retrieval metrics (e.g., mAP, Recall@K)

**Discussion**

The group discussed the importance of starting with a simple and controlled baseline before scaling to more complex models. It was observed that while ResNet18 provided stable training behaviour, its ability to capture fine-grained visual differences appeared limited in early results.

The team also noted that training configuration (e.g., optimizer choice and lack of augmentation) could significantly influence performance and should be revisited in subsequent iterations.

**Decisions**

* ResNet18 was retained as a controlled baseline for comparison
* Embedding-based evaluation was prioritised over classification accuracy
* Further refinement of training setup and exploration of stronger CNN architectures (e.g., VGG, EfficientNet) were planned for the next stage
