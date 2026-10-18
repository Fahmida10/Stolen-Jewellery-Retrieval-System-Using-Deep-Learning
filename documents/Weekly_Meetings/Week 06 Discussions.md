### Week 6 – CNN refinement and improved baseline configuration  
**Date:** 03/03/2026 (Zoom)  
**Attendance:** All five members  

**Objective**  
To improve baseline model performance after observing limitations in the initial ResNet18 setup.

**Tasks**  
- Evaluated performance of initial ResNet18 baseline  
- Identified limitations in feature extraction and retrieval quality  
- Implemented VGG16 as a stronger CNN baseline  
- Applied improved training configuration (AdamW, scheduler, augmentation, mixed precision)  
- Generated embeddings and performed retrieval-based evaluation  

**Discussion**  
The group observed that the initial ResNet18 configuration produced suboptimal results, particularly in retrieval performance. It was noted that the training setup (e.g., optimizer choice, lack of augmentation) limited the model’s ability to learn fine-grained visual features.  

To address this, a higher-capacity architecture (VGG16) with improved training parameters was introduced. The group discussed that better optimization strategies and regularisation could significantly improve feature representation even within CNN-based approaches.

**Decisions**  
- VGG16 was adopted as a stronger baseline model  
- Improved training configuration (AdamW, cosine scheduling, augmentation) was standardised  
- ResNet18 limitations were documented for later refinement  
- CNN limitations in retrieval tasks were acknowledged, motivating further exploration in later stages  
