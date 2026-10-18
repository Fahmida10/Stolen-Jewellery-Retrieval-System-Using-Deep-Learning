### Week 9 – OpenCLIP baseline (ViT-B/32)

**Date:** 24/03/2026 (Zoom)
**Attendance:** All five members

**Objective**
To introduce a transformer-based model and evaluate its baseline performance for image retrieval tasks.

**Tasks Completed**

* Implemented OpenCLIP ViT-B/32 model
* Applied CLIP-specific preprocessing and normalization
* Trained the model using a single-phase cross-entropy objective
* Extracted L2-normalised embeddings for retrieval
* Evaluated performance using retrieval metrics (mAP, Recall@K)

**Previous Learning Applied**
Earlier CNN-based experiments showed strong performance but limitations in capturing fine-grained visual similarity. This stage explored whether transformer-based representations could improve embedding quality for retrieval tasks.

**Discussion**
The group observed that although OpenCLIP provides powerful pretrained embeddings, the use of a standard cross-entropy objective was not sufficient for optimising retrieval performance.

It was identified that classification-based training does not directly enforce similarity relationships in the embedding space, which is critical for retrieval tasks. As a result, the model underperformed compared to the strongest CNN baseline (EfficientNet-B0) under this configuration.

**Challenges Faced**

* Understanding CLIP-specific preprocessing and normalization
* Adapting the training pipeline from CNNs to transformer-based models
* Interpreting performance differences between classification and retrieval objectives

**Decisions**

* The ViT-B/32 stage was treated as a controlled transformer baseline
* Limitations of cross-entropy training for retrieval were acknowledged
* A more advanced training strategy combining classification and metric learning objectives was planned

**Outcome**
By the end of Week 9, the OpenCLIP baseline was successfully implemented and evaluated.

The results demonstrated that while transformer-based models have strong representational capacity, appropriate training objectives are essential for effective retrieval performance. This motivated the development of a retrieval-aware training strategy in the next stage.
