# Technical and Responsive Design Audit: FedEquiMed

**Project:** FedEquiMed — Fairness-Aware Federated Learning in Healthcare  
**Audit Date:** September 2026  
**Auditor:** Core Engineering & Research Team  
**Scope:** Mathematics, Algorithms, Privacy/Security, Data Pipeline, Responsive Frontend, APIs  

---

## 1. Executive Summary

A comprehensive audit was performed across the complete FedEquiMed application stack:
1. Pure Python Machine Learning & FL Engine (`fl_engine/`)
2. Python Standard Library HTTP REST API Server (`server.py`)
3. Single Page Frontend (`static/index.html`, `static/style.css`, `static/app.js`)
4. Data generation and persistence layers (`data/`)

This audit documents every identified issue, explains why it was mathematically, technically, or architecturally problematic, details the corrected implementation, identifies the modified files, and outlines the testing protocol.

---

## 2. Mathematical & Algorithmic Issues

### Issue 2.1: Demographic Parity Calculation & Misleading Definition
- **Why problematic:** Demographic Parity was previously described in docstrings and UI explanations as "equalizing $P(Y, A)$". Equalizing the joint probability $P(Y, A)$ is not Demographic Parity; Demographic Parity requires equal positive prediction rates across demographic groups:
  $$P(\hat{Y}=1 \mid A=g_1) = P(\hat{Y}=1 \mid A=g_2)$$
  Furthermore, the difference calculation only compared groups 0 and 1, falling back to an unweighted multi-group difference without standardized definitions.
- **Current implementation:** In `fl_engine/bias_detector.py`, pairwise difference $|p_0 - p_1|$ for binary groups, or unnormalized multi-group differences.
- **Correct implementation:** Explicitly define and calculate Demographic Parity Difference ($\Delta\text{DP}$):
  $$\Delta\text{DP} = \max_{g \in \mathcal{G}} P(\hat{Y}=1 \mid A=g) - \min_{g \in \mathcal{G}} P(\hat{Y}=1 \mid A=g)$$
  Where $\hat{Y}$ is model prediction, $A$ is the sensitive attribute, and $g$ is a demographic group.
- **File(s) changed:** `fl_engine/bias_detector.py`, `fl_engine/hospital_node.py`, `static/app.js`, `static/index.html`.
- **How tested:** Tested with synthetic cohorts having known positive prediction rates (e.g. Group 0: 0.8, Group 1: 0.4, Group 2: 0.2 $\rightarrow \Delta\text{DP} = 0.60$).

---

### Issue 2.2: Equalized Odds Calculation Formula
- **Why problematic:** Equalized Odds requires conditional independence of prediction $\hat{Y}$ and protected attribute $A$ given actual outcome $Y$:
  $$P(\hat{Y}=1 \mid A=g, Y=y) = P(\hat{Y}=1 \mid Y=y), \quad \forall y \in \{0, 1\}$$
  This requires both True Positive Rate ($\text{TPR}$) parity and False Positive Rate ($\text{FPR}$) parity. The previous implementation computed `max(fpr_diff, tpr_diff)` for binary groups, which hid disparities when one rate was balanced but the other was severely skewed.
- **Current implementation:** Max scalar across binary TPR and FPR diffs.
- **Correct implementation:** Formulate Equalized Odds Difference ($\Delta\text{EO}$) as:
  $$\Delta\text{EO} = \frac{1}{2} \left[ \left( \max_g \text{TPR}_g - \min_g \text{TPR}_g \right) + \left( \max_g \text{FPR}_g - \min_g \text{FPR}_g \right) \right]$$
  and also report individual $\Delta\text{TPR}$ and $\Delta\text{FPR}$ in detailed audits.
- **File(s) changed:** `fl_engine/bias_detector.py`, `fl_engine/hospital_node.py`.
- **How tested:** Unit tests with controlled contingency tables verifying $\Delta\text{TPR}$ and $\Delta\text{FPR}$ independence.

---

### Issue 2.3: Disparate Impact Ratio (DIR) Zero Denominator Vulnerability
- **Why problematic:** Disparate Impact Ratio was computed as:
  $$\text{DIR} = \frac{\min(p_0, p_1)}{\max(p_0, p_1)}$$
  If both groups had zero positive predictions ($\max(p_0, p_1) = 0$), the code fell back to `1.0`, but if one group had 0 and the other had $>0$, division by zero could occur or return 0.0 without numerical smoothing.
- **Current implementation:** Basic ternary operator without numerical epsilon stabilization.
- **Correct implementation:**
  $$\text{DIR} = \begin{cases} 1.0 & \text{if } \max_g P(\hat{Y}=1 \mid A=g) = 0 \text{ and } \min_g P(\hat{Y}=1 \mid A=g) = 0 \\ \frac{\min_g P(\hat{Y}=1 \mid A=g) + \epsilon}{\max_g P(\hat{Y}=1 \mid A=g) + \epsilon} & \text{otherwise} \end{cases}$$
  where $\epsilon = 10^{-7}$. A DIR of 1.0 denotes perfect demographic parity, and values $< 0.80$ violate the 80% four-fifths rule.
- **File(s) changed:** `fl_engine/bias_detector.py`.
- **How tested:** Tested zero positive predictions for both groups, one group zero, and identical positive rates.

---

### Issue 2.4: FairFed Weighting Normalization & Disparity Sign
- **Why problematic:** In FairFed, client weights are assigned as:
  $$\alpha_k = \frac{n_k}{N} \exp(-\beta \cdot \Delta_k)$$
  If $\Delta_k$ is signed or relative, negative values could exponentially inflate the weight of an unfair client instead of penalizing it. Furthermore, when client disparity was zero or near-zero, weight normalization needed to guarantee $\sum_k w_k = 1.0$.
- **Current implementation:** Evaluated $| \Delta_k - \bar{\Delta} |$ without guaranteeing non-negativity for arbitrary disparity scores, and lacked defensive checks for missing metrics or single-client aggregation.
- **Correct implementation:** Explicitly enforce $\Delta_k = |\Delta\text{DP}_k| \ge 0$, calculate $\alpha_k = \frac{n_k}{N} \exp(-\beta \cdot \Delta_k)$, and normalize:
  $$w_k = \frac{\alpha_k}{\sum_{j=1}^K \alpha_j}$$
  Verify that $\sum_k w_k = 1.0 \pm 10^{-6}$.
- **File(s) changed:** `fl_engine/global_server.py`.
- **How tested:** Tested single client ($w_1 = 1.0$), equal disparity clients, highly unfair client ($w_k \to 0$ as $\beta \to \infty$), zero disparity clients, and raw signed inputs.

---

## 3. Privacy, Differential Privacy & Security Issues

### Issue 3.1: Overclaimed Privacy Guarantees
- **Why problematic:** The application repeatedly claimed "100% Zero-Leakage Privacy Contract", "Zero data leakage risk", and "prevents reconstruction attacks". In federated learning, gradient and weight updates can leak training data under gradient inversion attacks (e.g. Geiping et al., DLG). Simple Gaussian perturbation without formal privacy budgeting does not provide an absolute mathematical zero-leakage guarantee.
- **Current implementation:** Hardcoded claims of "Zero raw data storage", "100% privacy", and "reconstruction attacks prevented" in HTML and payload dictionaries.
- **Correct implementation:** Replaced with technically accurate language:
  - "Raw patient records are not transmitted to the central aggregator."
  - "Differential privacy perturbation is applied to local weight updates to reduce information leakage risks."
  - "Simulation/Prototype: Calibrated Gaussian mechanism with $L_2$ gradient clipping."
- **File(s) changed:** `fl_engine/hospital_node.py`, `server.py`, `static/index.html`, `static/app.js`, `README.md`.
- **How tested:** Full text search verification removing unsupported claims.

---

### Issue 3.2: Differential Privacy Transparency & Formal Parameters
- **Why problematic:** The clipping threshold ($C = 1.0$) and noise standard deviation ($\sigma = 0.005$) were hardcoded inside `hospital_node.py` without tracking sampling ratio $q$, communication rounds $T$, target $\delta$, or resulting $\epsilon$. A noise multiplier of $\sigma = 0.005$ is exceptionally low and provides virtually no formal theoretical $\epsilon$ guarantee.
- **Current implementation:** Ad-hoc Gaussian noise generator in `train_local_model`.
- **Correct implementation:** Created `fl_engine/privacy_engine.py` with explicit configuration:
  - Clipping Norm $C$
  - Noise Multiplier $\sigma$
  - Sampling Rate $q = B / N$
  - Local Epochs $E$ and Federated Rounds $T$
  - Target $\delta = 10^{-5}$
  - Analytical $(\epsilon, \delta)$ estimation via Moments / Renyi DP accountant.
  - Added Privacy Configuration panel to frontend and backend.
- **File(s) changed:** `fl_engine/privacy_engine.py`, `fl_engine/hospital_node.py`, `server.py`, `static/index.html`, `static/app.js`.
- **How tested:** Verified numerical output of $(\epsilon, \delta)$ accountant across varying rounds and noise levels.

---

### Issue 3.3: Hardcoded Admin Credential (`admin123`)
- **Why problematic:** The string `"admin123"` was hardcoded in `server.py`, exposed in plaintext in `app.js` and `index.html`, and documented in `README.md`. Anyone inspecting the page source could access admin functions.
- **Current implementation:** `if admin_key != "admin123"` in `server.py`.
- **Correct implementation:**
  - Read admin secret from environment variable `ADMIN_PASSKEY`.
  - Use constant-time comparison `secrets.compare_digest` to prevent timing attacks.
  - Remove all client-side password checks and hardcoded strings from `app.js` and `index.html`.
  - Validate admin credentials strictly server-side on protected endpoints (`/api/delete_hospital`, `/api/reset`, etc.).
  - Added `.env.example` with placeholder credentials only.
- **File(s) changed:** `server.py`, `static/app.js`, `static/index.html`, `README.md`, `.env.example`.
- **How tested:** Verified rejected requests with wrong passkey, verified accepted requests with environment passkey.

---

### Issue 3.4: Data Flow & Raw Data Privacy Verification
- **Why problematic:** Uploading custom hospital CSV files sent raw patient microdata over HTTP `POST /api/load_custom_data` and `POST /api/register_hospital` to the central backend. Claiming "100% zero network transmission" while accepting CSV payloads on the central server is contradictory.
- **Current implementation:** Server received CSV content, parsed it, and held it in memory on the server.
- **Correct implementation:** Clearly documented the architecture as a **simulated federated network**:
  - In production, hospital nodes run as independent processes behind hospital firewalls.
  - In this demonstration environment, hospital nodes run in separate virtualized instances within the backend runtime.
  - Explicitly labeled client-side browser testing vs backend node onboarding.
- **File(s) changed:** `docs/ARCHITECTURE.md`, `docs/PRIVACY.md`, `static/index.html`.
- **How tested:** End-to-end data flow tracing.

---

## 4. Experiment Pipeline & Evaluation Issues

### Issue 4.1: Hardcoded Benchmark Numbers & Missing Experiment Persistence
- **Why problematic:** In `fl_engine/global_server.py`, `run_benchmark_comparison` contained hardcoded fallback values for DIR (`0.81` for FedAvg, `0.92` for FairFed, `0.98` for FedEquiMed) rather than calculating them dynamically from evaluation predictions. There was no experiment persistence, no experiment IDs, and no multi-seed support.
- **Current implementation:** Hardcoded dictionaries in benchmark comparison return.
- **Correct implementation:**
  - Created reproducible file-based experiment system under `data/experiments/{experiment_id}/`.
  - Generated standard experiment IDs (e.g. `FEQ-2026-001`).
  - Saved `config.json`, `metrics.csv`, and `results.json`.
  - Dynamically evaluated all metrics (Accuracy, Precision, Recall, F1, AUROC, $\Delta\text{DP}$, $\Delta\text{EO}$, $\text{DIR}$) directly on held-out test data.
  - Display "N/A — experiment not yet executed" for unexecuted runs.
- **File(s) changed:** `fl_engine/experiments.py`, `fl_engine/global_server.py`, `server.py`, `static/app.js`, `static/index.html`.
- **How tested:** Executed multiple experiments, verified file creation on disk, verified that results match computed predictions.

---

### Issue 4.2: Multi-Seed Benchmark Evaluation
- **Why problematic:** Machine learning results with single random seeds suffer from stochastic variance. No standard deviations or statistical confidences were provided.
- **Current implementation:** Fixed seed `100` only.
- **Correct implementation:** Added multi-seed runner across default seeds `[42, 123, 456, 789, 999]`, computing and displaying:
  $$\text{Mean} \pm \text{Std Dev}$$
  for Accuracy, F1, AUROC, $\Delta\text{DP}$, $\Delta\text{EO}$, and DIR.
- **File(s) changed:** `fl_engine/experiments.py`, `server.py`, `static/app.js`, `static/index.html`.
- **How tested:** Ran 5-seed benchmark evaluation, verified statistical calculations.

---

### Issue 4.3: Missing Component Ablation Study
- **Why problematic:** FedEquiMed proposes multiple novel components (ClinDisentangleNet, CARP-Loss, D-TAMA momentum aggregator), but previously offered no ablation framework demonstrating what each individual component contributed.
- **Current implementation:** None.
- **Correct implementation:** Added a 5-way ablation experiment framework:
  1. Component A: Baseline FedAvg
  2. Component B: FedAvg + ClinDisentangleNet (Representation Disentanglement)
  3. Component C: FedAvg + CARP-Loss (Covariate-Adjusted Risk Parity)
  4. Component D: FedAvg + D-TAMA (Dual-Tier Adaptive Momentum Aggregation)
  5. Component E: Full FedEquiMed System (All components integrated)
- **File(s) changed:** `fl_engine/experiments.py`, `fl_engine/models.py`, `fl_engine/global_server.py`, `server.py`, `static/app.js`, `static/index.html`.
- **How tested:** Executed full ablation pipeline across identical splits and verified output metrics for each component.

---

### Issue 4.4: Primary Healthcare Task Definition
- **Why problematic:** The interface listed five unrelated datasets (MIMIC-IV, eICU, MedMNIST, COVID-19 X-ray, TCGA Cancer) in the same pipeline as if they represented interchangeable runs of the same model architecture.
- **Current implementation:** Synthetic generator loosely imitating 5 datasets with the same 10 linear features.
- **Correct implementation:**
  - Designate **MIMIC-IV ICU Clinical Risk & 30-Day Adverse Event Prediction** as the primary canonical benchmark task.
  - Formally document the 10 physiological and demographic features, prediction target, sensitive attributes, and scaling parameters.
  - Relegate other datasets to "Additional Benchmarks / Research References".
- **File(s) changed:** `docs/EXPERIMENTS.md`, `README.md`, `fl_engine/dataset.py`, `static/index.html`.
- **How tested:** Verified dataset documentation and primary task consistency.

---

## 5. Responsive Design Audit

### Breakpoints Used
- **Mobile Extra-Small:** $320\text{px} - 374\text{px}$ (iPhone SE 1st gen, small smartphones)
- **Mobile Standard:** $375\text{px} - 413\text{px}$ (iPhone 12/13/14 mini, standard smartphones)
- **Mobile Large / Phablet:** $414\text{px} - 767\text{px}$ (iPhone Plus/Max, Google Pixel Pro)
- **Tablet:** $768\text{px} - 1023\text{px}$ (iPad mini, iPad Air, Galaxy Tab)
- **Laptop:** $1024\text{px} - 1439\text{px}$ (MacBook Air, small laptops)
- **Desktop:** $\ge 1440\text{px}$ (Large monitors, iMacs)

### Responsive Issues Found & Corrected

| Component | Issue on Narrow Viewports (<768px) | Corrective Action Taken |
|---|---|---|
| **Header** | Brand title and badge buttons wrapped awkwardly and caused horizontal overflow. | Applied responsive flex-wrap, converted badges into compact status pills, moved full-width actions below title. |
| **Navigation Tabs** | Fixed-width tabs overflowed horizontally off-screen without touch scroll affordance. | Added mobile-friendly collapsible hamburger drawer / horizontal scroll tab bar with hidden scrollbars and active indicators. |
| **Metric Stat Cards** | Fixed 5-column grid crushed metric numbers into overlapping text on screens $< 600\text{px}$. | Converted to responsive grid: 1 column on $\le 480\text{px}$, 2 columns on $481-800\text{px}$, fluid on $\ge 801\text{px}$. |
| **Data Tables** | Technical comparison tables (6-9 columns) clipped off-screen and broke page width. | Wrapped in `.table-responsive` with custom touch-scroll indicator and sticky headers, or auto-converted into stacked cards on mobile $\le 640\text{px}$. |
| **Canvas Charts** | HTML5 Canvas charts rendered with fixed pixel dimensions or failed to resize upon screen rotation. | Wrapped in responsive aspect-ratio containers, attached `ResizeObserver` / `window.resize` handlers to re-render dynamically. |
| **Forms & Controls** | Multi-column inputs (e.g. hospital onboarding, DP config) squished labels and inputs. | Switched to single-column vertical stacking on mobile viewports $< 640\text{px}$. |
| **Touch Targets** | Small reset and toggle buttons had clickable areas $< 28\text{px}$, causing misclicks. | Enforced minimum touch target height and width of $44\text{px} \times 44\text{px}$ for touch interfaces. |
| **Modals & Dialogs** | Fixed-width dialogs overflowed viewport on 320px screens. | Applied `max-width: 92vw`, `max-height: 90vh`, and responsive internal scrolling. |
| **Flowchart Board** | Fixed horizontal arrows (`➔`) forced wide horizontal scroll on mobile. | Switched from horizontal flex row to vertical stacked cards with downward arrows (`⬇`) on screens $< 768\text{px}$. |

### Viewports Tested
- **320px**: Verified on small mobile layout (no horizontal page scrolling, all cards stack cleanly).
- **375px**: Verified on standard mobile layout (iPhone SE/13 mini).
- **390px**: Verified on iPhone 12/13/14/15 standard width.
- **414px**: Verified on iPhone 8 Plus / XR / 11.
- **768px**: Verified on iPad portrait.
- **1024px**: Verified on iPad landscape / MacBook 13".
- **1440px**: Verified on desktop display.
