# Section 8. Peer review and formal feedback

**Section 8** of 11. **Index:** [Contents](README.md) | **Previous:** [Section 7](07_meetings_and_review_cycles.md) | **Next:** [Section 9](09_contributions.md)

## 8.1 Internal peer review

Code and configuration changes were pushed to GitLab and reviewed informally within the team. Team members prioritised numerical consistency of normalisation and augmentation, deterministic evaluation, exclusion of large binary artefacts from version control, and traceability of filenames through the embedding and retrieval pipeline. Consensus was required before applying changes that altered evaluation logic or dataset enumeration.

## 8.2 Tutor and module feedback

This section summarises key instances of supervisory feedback that informed major development decisions

**1.Feedback (Supervisor, in-class discussion, Week 2):** Guidance was provided on selecting appropriate models based on dataset size, computational constraints, and project objectives, including the transition from CNNs to OpenCLIP.
**Action:** The team implemented CNN baselines followed by OpenCLIP to improve embedding quality for retrieval tasks.

**2.Feedback (Supervisor, in-class discussion, Week 4):** Clarification was given on dataset structuring, including train, validation, and test splits, and the use of Google Colab for training and inference.
**Action:** The dataset was organised into disjoint splits, and the pipeline was configured for Colab-based execution.

**3.Feedback (Supervisor, in-class discussion, Week 6):** Guidance was provided on selecting appropriate hyperparameters, including learning rates and training settings, with reference to prior jewellery-related studies.
**Action:** The team reviewed relevant literature and adopted suitable hyperparameters, including learning rate configurations and training schedules, aligned with established practices.

**4.Feedback (Supervisor, in-class discussion, Week 7):** A multi-phase training strategy was recommended to improve model performance.
**Action:** A staged training approach with freezing and unfreezing of layers was implemented.

**5.Feedback (Supervisor, in-class discussion, Week 9):** Evaluation should reflect retrieval performance rather than relying solely on classification metrics.
**Action:** Retrieval-specific metrics, including mAP and Recall@K, were incorporated.

## 8.3 Response to feedback and alignment with the thesis

Internal and supervisory feedback informed several key improvements to the project. Issues such as class-index drift caused by hidden filesystem entries were addressed through explicit filtering, regeneration of embeddings, and validation of results using both mAP and Recall@K.

Feedback on model selection led to the transition from CNN baselines to OpenCLIP, improving embedding quality for retrieval tasks. Guidance on dataset structuring resulted in the use of disjoint train, validation, and test splits, ensuring reliable evaluation.

Recommendations on hyperparameter selection and multi-phase training were incorporated through literature-informed configurations and staged training strategies, improving model adaptation and performance.

Feedback on evaluation led to the adoption of retrieval-specific metrics, including mAP and Recall@K, aligning the implementation with the retrieval-focused objective of the thesis. Broader **ethical** and **limitation** arguments were integrated into the [discussion chapter](../documents/README_DETAILED_REPORT.md#discussion) of the technical report. **Synthesis** of outcomes is presented in [Section 10](10_conclusion.md).

---

**Navigation:** [Repository home](../README.md) · [Discussion in report](../documents/README_DETAILED_REPORT.md#discussion) · [Conclusion (Section 10)](10_conclusion.md)
