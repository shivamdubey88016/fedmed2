# Mathematical Fairness Metrics Definition & Formulation

## 1. Demographic Parity Difference ($\Delta\text{DP}$)
Demographic Parity evaluates whether the probability of receiving a positive clinical prediction is independent of the protected sensitive attribute $A$.

$$\Delta\text{DP} = \max_{g \in \mathcal{G}} P(\hat{Y}=1 \mid A=g) - \min_{g \in \mathcal{G}} P(\hat{Y}=1 \mid A=g)$$

- **Ideal Value**: $0.00$
- **Interpretation**: Lower disparity indicates equitable acceptance/positive prediction rates across demographic cohorts.
- **Limitation**: Does not account for true underlying prevalence differences across clinical groups.

## 2. Equalized Odds Difference ($\Delta\text{EO}$)
Equalized Odds requires that both the True Positive Rate ($\text{TPR}$) and False Positive Rate ($\text{FPR}$) are equal across demographic subgroups.

$$\Delta\text{EO} = \frac{1}{2} \left[ \left(\max_{g} \text{TPR}_g - \min_{g} \text{TPR}_g\right) + \left(\max_{g} \text{FPR}_g - \min_{g} \text{FPR}_g\right) \right]$$

- **Ideal Value**: $0.00$
- **Interpretation**: Lower difference indicates equal predictive error rates regardless of protected identity.

## 3. Disparate Impact Ratio ($\text{DIR}$)
Calculates the relative ratio of positive rates between the most disadvantaged and most favored groups:

$$\text{DIR} = \frac{\min_{g} P(\hat{Y}=1 \mid A=g) + \epsilon}{\max_{g} P(\hat{Y}=1 \mid A=g) + \epsilon}$$

Where $\epsilon = 10^{-7}$ ensures numerical stability when rates approach zero.
- **Ideal Value**: $1.00$
- **Threshold**: Values $< 0.80$ violate the four-fifths (80%) legal fairness doctrine.
