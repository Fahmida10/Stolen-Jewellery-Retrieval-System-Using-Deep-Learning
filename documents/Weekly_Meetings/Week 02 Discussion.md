### Week 2 – Data collection and dataset structuring  
**Date:** 03/02/2026 (Zoom)  
**Attendance:** All five members  

**Objective**  
To identify, collect, and structure a dataset suitable for fine-grained jewellery image retrieval.

**Tasks**  
- Identified and reviewed potential data sources  
- Collected images from:
  - Tanishq Jewellery dataset (Kaggle)  
  - Roboflow Jewellery dataset  
  - Etsy (custom-collected images)  
- Combined sources into a unified dataset  
- Organised dataset into training, validation, test, and retrieval (gallery) partitions  

**Discussion**  
The group evaluated dataset quality requirements for retrieval tasks, emphasising the need for fine-grained visual differences and diverse real-world imagery. The importance of combining multiple sources to improve variation and robustness was discussed.

**Decisions**  
- A **multi-source dataset** was adopted to increase diversity  
- A **retrieval-specific structure** was defined: train / validation / test / stolen set  
- Strict separation between supervised data and retrieval gallery was established to prevent leakage  
