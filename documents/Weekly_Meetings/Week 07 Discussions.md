### Week 7 – Efficient CNN baseline (EfficientNet-B0)

**Date:** 10/03/2026 (Zoom)
**Attendance:** All five members

**Objective**
To implement and evaluate a CNN model that balances strong feature representation with computational efficiency.

**Tasks Completed**

* Implemented EfficientNet-B0 architecture
* Applied improved training configuration:

  * AdamW optimizer
  * Learning rate: 1e-4
  * Cosine annealing scheduler
  * Data augmentation enabled
  * Mixed precision training enabled
* Trained the model using the prepared dataset
* Extracted embeddings (1280-dimensional feature vectors)
* Evaluated retrieval performance using mAP and Recall@K

**Previous Learning Applied**
Earlier experiments showed that the initial ResNet18 baseline produced weak retrieval performance, while VGG16 improved results but required higher computational cost.

This stage explored EfficientNet-B0 as a model that could provide strong feature representation while maintaining efficiency.

**Discussion**
The group discussed how training configuration and architecture design jointly affect embedding quality. EfficientNet-B0, with its compound scaling strategy, was expected to capture fine-grained visual details more effectively than earlier CNN baselines.

It was observed that the improved training setup (AdamW, augmentation, scheduler) contributed to better feature learning and more stable training behaviour.

**Outcome**
By the end of Week 7, EfficientNet-B0 established the strongest CNN baseline in terms of retrieval performance.

The results confirmed that while CNN-based approaches can perform well, further improvements would likely require more advanced embedding methods, motivating the transition to OpenCLIP in the next stage.
