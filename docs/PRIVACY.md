# Differential Privacy & Data Protection Specification

## 1. Threat Model & Privacy Scope
- **Threat Model**: Honest-but-curious central aggregator and potential eavesdroppers observing communicated model parameter updates.
- **Protection Scope**: Raw Electronic Health Records (EHR) never leave hospital nodes. Only clipped and perturbed weight deltas are transmitted.

## 2. Differential Privacy Mechanism (DP-SGD)
1. **$L_2$ Norm Clipping**:
   $$\Delta \bar{W} = \Delta W \cdot \min\left(1, \frac{C}{\|\Delta W\|_2}\right)$$
   Default clipping threshold: $C = 1.0$.

2. **Calibrated Gaussian Perturbation**:
   $$\tilde{\Delta W} = \Delta \bar{W} + \mathcal{N}(0, \sigma^2 C^2 \mathbf{I})$$
   Default noise multiplier: $\sigma = 0.02 - 0.05$.

3. **Renyi Differential Privacy (RDP) Budgeting**:
   Estimated analytical privacy bound $(\epsilon, \delta)$ is tracked across communication rounds $T$ and sampling ratio $q = B / N$.

## 3. Technically Honest Statements
- Claims of "100% privacy" or "zero leakage guarantee" are invalid under empirical gradient inversion attacks and are explicitly avoided.
- The platform provides a transparent privacy configuration panel where researchers can inspect parameters, evaluate trade-offs, and monitor cumulative analytical privacy consumption.
