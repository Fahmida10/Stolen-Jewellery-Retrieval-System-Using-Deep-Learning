### Week 10 – Final model setup (OpenCLIP ViT-L/14)

**Date:** 31/03/2026 (Zoom)
**Attendance:** All five members

**Objective**
To prepare the final transformer-based model and design an improved training strategy for retrieval tasks.

**Tasks Completed**

* Initiated implementation of the OpenCLIP ViT-L/14 model
* Defined training pipeline structure with a temporary classifier head
* Applied CLIP-specific preprocessing and normalization
* Designed a multi-phase training strategy for subsequent stages

**Previous Learning Applied**
From Week 9, it was observed that a simple cross-entropy training setup was insufficient for optimising retrieval performance.

This stage focused on addressing that limitation by planning a structured training approach that better aligns with embedding-based retrieval objectives.

**Discussion**
The group discussed how larger transformer models require careful training design to fully utilise their representational capacity. A staged training approach was proposed to improve learning stability and gradually adapt the pretrained model to the jewellery dataset.

It was agreed that combining classification and retrieval-oriented objectives would be explored in the following stage.

**Decisions**

* OpenCLIP ViT-L/14 was selected as the final model architecture
* A multi-phase training strategy was defined but not yet fully executed
* Training objectives would be extended beyond cross-entropy in subsequent stages
* The classifier head was treated as a temporary component for training guidance

**Outcome**
By the end of Week 10, the final model architecture and training framework were prepared, providing a structured foundation for full training and optimisation in the following stage.
