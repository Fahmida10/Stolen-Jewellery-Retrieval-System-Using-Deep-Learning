# Section 10. Conclusion and critical synthesis

**Section 10** of 11. **Index:** [Contents](README.md) | **Previous:** [Section 9](09_contributions.md) | **Next:** [Section 11](11_checklist.md)

## 10.1 What was delivered

The project delivers a comparative study and a reference implementation for jewellery image retrieval under controlled data splits, culminating in an OpenCLIP ViT-L/14 encoder trained using a multi-phase objective and integrated Top-K retrieval functionality. Quantitative and visual results are archived in the **`images/`** and developed analytically in the **technical report**.

## 10.2 Principal strengths

The **unified protocol** permits **fair comparison** across **CNN** and **transformer** families. The **final model** attains the strongest **mAP** and **embedding accuracy** in the study, with **competitive Recall@K** at moderate **K**. The **pipeline** supports **auditable** ranked lists suitable for **demonstration**.

## 10.3 Principal difficulties

**Fine-grained** discrimination on **limited**, **catalogue-style** imagery increases **memorisation risk**. **Preventing leakage** between **supervised** data and the **retrieval gallery** required **disciplined directory design**. **Interpretation** of **multiple metrics** simultaneously was non-trivial when **ranking** improvements did not coincide with **uniform** gains at **all K**.

## 10.4 Scholarly and practical lessons learned

**Retrieval quality** depends on **embedding geometry**; **classification accuracy** is therefore an **incomplete** proxy. **Transformer** capacity must be matched to **objective design** and **regularisation** when data are **scarce**. **Low-level software hygiene** can dominate **metric stability** if neglected.

## 10.5 Reflection on collaboration

The **division of labour** described in [Section 9](09_contributions.md) was intended to **parallelise** independent model stages while preserving a **single evaluation standard**. **Integration** relied on **frequent merges** and **shared documentation**.

## 10.6 Recommended future work

Priority directions include **larger and more diverse** imagery, **attribute-level** supervision, **approximate nearest-neighbour** indexing for **large galleries**, and **governance mechanisms** for **dual-use** retrieval technology. These points are elaborated in the [future work section](../documents/README_DETAILED_REPORT.md#future-work) of the technical report.

---

**Navigation:** [Repository home](../README.md) · [Results and evaluation (Section 5)](05_results_and_evaluation.md) · [Report conclusion](../documents/README_DETAILED_REPORT.md#conclusion)
