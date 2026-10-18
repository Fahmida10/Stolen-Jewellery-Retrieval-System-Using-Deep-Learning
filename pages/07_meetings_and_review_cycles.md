# Section 7. Meeting record and cyclical process review

**Section 7** of 11. **Index:** [Contents](README.md) | **Previous:** [Section 6](06_management.md) | **Next:** [Section 8](08_feedback.md)

## 7.1 Purpose of this section

This section documents **coordination decisions** that shaped the technical trajectory and records a **structured retrospective** on each major development cycle. Together, these materials demonstrate **governance** of the collaboration in a form suitable for **academic assessment**. **Dates** for the earliest commits may be recovered from **GitLab**; where **exact calendar dates** are required by the module, they should be **inserted by the team** alongside the headings below.

## 7.2 Record of coordination meetings

This subsection provides an abridged summary of key coordination meetings. Full weekly meeting records, including detailed notes and timelines, are available in the repository ([Weekly Meeting Records](../documents/Weekly_Meetings/)).

### 7.2.1 Initiation of the project and definition of protocol

**Attendance:** All five contributing members.

**Summary:** The group formalised a **retrieval-first** objective with **Top-K** outputs and **filename traceability**, agreed on **disjoint** supervision and gallery partitions, and fixed a **common epoch budget** across model families for **comparability**.

**Decisions:** Dataset root conventions were frozen; CNN stages 01–03 and the OpenCLIP trajectory were assigned jointly to Ananna and Abdi; model training and evaluation were conducted collaboratively across all stages.

### 7.2.2 Review of convolutional baselines

**Summary:** The group compared **mAP** and **embedding accuracy** for **ResNet18**, **VGG16**, and **EfficientNet-B0**, concluding that **EfficientNet-B0** provided the strongest **CNN** reference prior to transformer experiments.

**Decisions:** A **two-phase ResNet18** variant was retained as an **alternative curriculum**; metrics were written into **stage-level README** files for traceability.

### 7.2.3 Transition to OpenCLIP

**Summary:** The **ViT-B/32** configuration trained with **cross-entropy alone** proved **insufficient** on **class-wise** analyses relative to strong CNNs.

**Decisions:** The group treated that stage as a controlled bridge and authorised a multi-phase objective on ViT-L/14, with Ananna and Abdi jointly leading implementation and evaluation across this stage.

### 7.2.4 Final retrieval pipeline and data integrity

**Summary:** **Metadata files** unrelated to class labels were found to perturb **directory-based class indexing**.

**Decisions:** **Filtering rules** were enforced; **embeddings** were regenerated; **retrieval proof** and **latency** measurements for the **small gallery** were added to the evidence set.

### 7.2.5 Preparation for submission

**Summary:** The repository was reorganised into **`pages/`**, **`images/`**, **`documents/`**, and **`code/`** to meet **portfolio readability** requirements.

**Decisions:** David led documentation structuring and cross-link verification; David also reviewed overall report integration; Laknath was responsible for conclusion, ethics, and limitations; all members attested to the [submission checklist (Section 11)](11_checklist.md).

## 7.3 Retrospective review by development cycle

### 7.3.1 Cycle I (convolutional foundations)

The first cycle established **reproducible baselines** and **shared data-loading** logic. A **limitation** was the **absence** of a formal architecture diagram for the earliest stage; the group subsequently **standardised** figure naming and placement under **`images/`**.

### 7.3.2 Cycle II (efficiency-oriented and curriculum CNNs)

**EfficientNet-B0** delivered strong **mAP**, whereas the **two-phase ResNet18** configuration exhibited **higher Recall at large K** despite **lower mAP**. The group concluded that **no single scalar** should dominate interpretation, which informed later **discussion** drafting.

### 7.3.3 Cycle III (transitional transformer)

The **ViT-B/32** experiment confirmed that **architecture change alone** is insufficient without **objective alignment** to **retrieval geometry**.

### 7.3.4 Cycle IV (final OpenCLIP and retrieval tooling)

The **ViT-L/14** system achieved the **best headline metrics**; however, **train–validation divergence** indicated **overfitting risk**, which was **mitigated** through **checkpoint discipline** rather than exhaustive use of the nominal epoch budget.

### 7.3.5 Cycle V (documentation consolidation)

The final cycle **integrated** evidence into a **single technical report** and a **navigable portfolio**. The principal tension was **conciseness** in portfolio sections versus **depth** in the report; the resolution was **cross-referencing** rather than **duplication**.

---

**Navigation:** [Repository home](../README.md) · [Management (Section 6)](06_management.md) · [Checklist (Section 11)](11_checklist.md)
