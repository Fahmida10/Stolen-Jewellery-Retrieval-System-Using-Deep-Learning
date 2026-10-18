### Week 8 – Two-phase fine-tuning strategy (ResNet18)

**Date:** 17/03/2026 (Zoom)
**Attendance:** All five members

**Objective**
To improve the ResNet18 baseline using a structured fine-tuning strategy and evaluate whether staged training can enhance retrieval performance.

**Tasks Completed**

* Implemented a two-phase training strategy on ResNet18:

  * Phase 1: trained the classifier head with the backbone frozen
  * Phase 2: fine-tuned the full network with reduced learning rates
* Applied differential learning rates for backbone and classifier
* Generated embeddings and evaluated performance using retrieval metrics

**Previous Learning Applied**
Earlier experiments showed that direct end-to-end training of ResNet18 resulted in weak retrieval performance. Although EfficientNet-B0 provided a stronger CNN baseline, this stage revisited ResNet18 to explore whether improved training strategy alone could enhance performance.

**Discussion**
The group discussed how staged fine-tuning can preserve pretrained feature representations while allowing gradual adaptation to the jewellery dataset. Separating head training and backbone tuning was found to provide better control over optimisation and stability.

However, it was observed that while this approach improved certain metrics (particularly higher Recall@K), the overall retrieval performance remained below the strongest CNN baseline.

**Decisions**

* Two-phase fine-tuning was retained as an alternative training strategy
* EfficientNet-B0 remained the primary CNN reference model
* Limitations of CNN-based approaches were acknowledged, motivating transition to transformer-based models

**Outcome**

By the end of Week 8, the two-phase fine-tuning strategy improved the ResNet18 baseline but did not surpass EfficientNet-B0.

This confirmed that while training strategy can enhance performance, model capacity and representation quality play a more significant role in retrieval tasks.
