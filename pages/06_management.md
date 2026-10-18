# Section 6. Project management

**Section 6** of **11.** **Index:** [Contents](README.md) | **Previous:** [Section 5](05_results_and_evaluation.md) | **Next:** [Section 7](07_meetings_and_review_cycles.md)

## 6.1 Chronological structure of the work

The work was organised into **six technical phases** followed by a **documentation phase**. The technical phases align with the **numbered code directories** (`01`–`06`), progressing from **CNN baselines** through **OpenCLIP** to the **final retrieval-capable** package. Calendar dates for each phase are reflected through the version-control history on GitLab, where commits were made continuously throughout development.

## 6.2 Allocation of responsibilities

A **tabular summary** of contributions appears in [Section 9](09_contributions.md). In brief, **coordination and synthesis** were separated from **implementation of individual model stages**, **evaluation scripting**, and **repository hygiene**, in order to reduce **single points of failure** during integration.

## 6.3 Management method

Development proceeded **iteratively**, with the **evaluation protocol** held **constant** across stages unless a **documented methodological amendment** (such as **hidden-file filtering**) was introduced for **validity**. **Binary artefacts** were excluded from Git in accordance with [`SUBMISSION_NOTES.md`](../documents/SUBMISSION_NOTES.md).

## 6.4 Instrumentation and infrastructure

Primary tools include **GitLab** for **source control**, **Python** and **PyTorch** for **experimentation**, and **OpenCLIP** for **transformer-based** encoders. **Compute** assumptions follow those stated in the technical report (e.g. **single-GPU** training comparable to **Colab**-class hardware).

## 6.5 Progress monitoring

**Merge requests**, **commit messages**, and **per-stage README files** jointly document **incremental progress**. The **technical report** consolidates **final metrics** and **figures**.

## 6.6 Risk register (abridged)

| Risk | Mitigation strategy |
|---|---|
| Overfitting on limited data | Validation-driven checkpointing; phased optimisation; conservative augmentation |
| Information leakage | Strict exclusion of the retrieval gallery from supervised updates |
| Repository size inflation | `.gitignore` patterns and pre-submission scans for prohibited binaries |
| Inconsistent class indexing | Filtering of non-image and hidden files; recomputation of embeddings |

Formal **meeting records** and **cyclical process reflection** are provided in [Section 7](07_meetings_and_review_cycles.md).

---

**Navigation:** [Repository home](../README.md) · [Meeting record (Section 7)](07_meetings_and_review_cycles.md) · [Contributions (Section 9)](09_contributions.md)
