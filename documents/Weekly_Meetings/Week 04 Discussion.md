### Week 4 – Preprocessing, normalization, and pipeline consistency  
**Date:** 17/02/2026 (Zoom)  
**Attendance:** All five members  

**Objective**  
To define a consistent preprocessing pipeline compatible with pretrained CNN models and ensure no data leakage across dataset splits.

**Tasks**  
- Converted all images to RGB format  
- Resized images to 224 × 224 resolution  
- Applied ImageNet normalization (mean and standard deviation)  
- Defined data augmentation strategy:
  - Horizontal flipping  
  - Small rotations  
  - Mild colour adjustments  

**Discussion**  
The group discussed the importance of maintaining consistency between training and inference pipelines. It was emphasised that preprocessing must remain identical across all dataset splits to avoid discrepancies in embedding generation. The need to prevent data leakage between supervised data and the retrieval gallery was also highlighted. Augmentation was applied conservatively to preserve fine-grained jewellery features.

**Decisions**  
- A standard preprocessing pipeline (RGB + 224×224) was finalised  
- ImageNet normalization was adopted for CNN stages  
- Conservative data augmentation was introduced to improve generalisation  
- A **no-leakage pipeline** was enforced, ensuring strict separation between training, validation, test, and retrieval (gallery) data  
- Identical preprocessing was applied across all splits to maintain embedding consistency  
