# Future Research & Engineering Roadmap

1. **Empirical Privacy Attack Demonstrator**:
   - Integrate simulated Membership Inference Attacks (MIA, Shokri et al.) and Gradient Inversion Attacks (Geiping et al.) to empirically demonstrate patient reconstruction resistance across noise multipliers $\sigma \in [0.001, 0.5]$.

2. **Decentralized Secure Multiparty Computation (SMC)**:
   - Combine DP-SGD with additive secret sharing or Homomorphic Encryption (Paillier) to guarantee that the server inspects only encrypted aggregates without seeing individual hospital weight updates.

3. **Multi-Modal Clinical Extensions**:
   - Extend the linear clinical classifier to convolutional and transformer-based architectures for joint radiological imaging (CheXpert, MIMIC-CXR) and longitudinal EHR modeling.

4. **Dynamic Client Dropout & Asynchronous FL**:
   - Model real-world hospital network dropouts, intermittent connectivity, and asynchronous aggregation routines.
