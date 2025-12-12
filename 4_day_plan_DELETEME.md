# AM226 Final Project — Full 4-Day Experimental Plan

## **Overview**

This project reproduces and extends a subset of results from the BioSystems AANN protein-network paper by implementing:

- A synthetic 12-protein discrete-time dynamical model
- Linear, sigmoid, and recurrent AANNs with fixed connectivity
- **Structure–function non-identifiability** analysis tied to **Gardner’s capacity framework**
- **Fixed-point and Jacobian eigenvalue stability** analysis
- **Contrastive Hebbian Learning (CHL)** vs SGD
- *(Optional)* Hybrid AANN
- *(Optional)* Self-learning divergence experiment

The plan below is scoped precisely to fit a **4-day development window**.

---

# **Day 1 — Dataset + Linear vs Sigmoid AANN (Required)**

### **1.1 Implement synthetic protein dynamics dataset**

- State dimension: `d = 12`
- Connectivity mask `S` from paper (signed ±1) or manually-specified mask with ∼3–5 regulators per protein
- Sample magnitude matrix:
    - `|W|_ij ~ LogNormal(mu=-1.2, sigma=0.6)` for edges in S
- Effective weights:
    - `W_eff = S * |W|`
- Rescale spectral radius of `W_eff` to ~0.8–1.0
- Use discrete-time dynamics:
    
    ```
    x_{t+1} = sigmoid( α W_eff x_t + b + noise )
    ```
    
    - α = 0.9
    - noise ~ N(0, 0.02²)
- Simulate:
    - T = 400 timesteps × 5 sequences
    - x₀ ~ Beta(2,2) per dimension
    - Train/val/test split: 70% / 15% / 15%

---

### **1.2 Implement masked AANN (linear + sigmoid)**

Model:

```
z = W_eff @ x + b
y = z          (linear)
y = sigmoid(z) (sigmoid)
```

- `W` and `b` are learnable
- Mask via `W_eff = W * M`
- Use MSE loss, manual gradients, SGD+momentum

---

### **1.3 Train and evaluate linear + sigmoid models**

Training:

- Epochs ≤ 300
- Early stopping with patience 20
- Learning rate ~ 1e-3
- Batch or full-sequence training

Outputs:

- Loss curves (train + validation)
- Test MSE comparison
- Per-protein correlation
- Predicted vs true trajectories for 3–4 proteins

Interpretation:

- Sigmoid should outperform linear → reproduces the paper’s core finding.

---

### **1.4 *(Optional)* Hybrid AANN**

*If time remains*:

Steps:

1. Fit single-neuron regressions for each protein (linear and sigmoid variants).
2. Select best activation per node.
3. Build a hybrid AANN with mixed activations.
4. Train and compare to linear/sigmoid.

Outputs:

- Hybrid vs sigmoid MSE table
- Barplot showing which proteins prefer linear vs sigmoid

---

# **Day 2 — Recurrent AANN + Non-Identifiability & Gardner (Required)**

## **2.1 Implement and train recurrent AANN (Required)**

Use lagged-input formulation:

```
tilde_x_t = concat(x_t, x_{t-1}) ∈ R^{24}
hat_x_{t+1} = sigmoid( W @ tilde_x_t + b )
```

- Train on shifted dataset with `(tilde_x_t → x_{t+1})`
- Use same loss and optimizer as Day 1

Outputs:

- MSE comparison: recurrent vs non-recurrent AANN
- Trajectory overlays for several proteins

Interpretation:

- Recurrent model should improve fidelity → fulfills reproduction requirement.

---

## **2.2 Structure–Function Non-Identifiability + Gardner Capacity (Required)**

### **A. Collect many solutions**

Train the **non-recurrent sigmoid AANN** N = 10–20 times with different random initializations.

Record for each run `i`:

- Final test MSE `L_i`
- Weights `W^(i)` and biases `b^(i)`
- A trajectory rollout `x_{0:T}^{(i)}` from a fixed initial state

---

### **B. Define “near-identical dynamics”**

Pick a reference solution (best or first).

For each solution `i`:

1. Generate predicted trajectory:
    
    ```
    x̂_{t+1}^{(i)} = sigmoid( W^(i) @ x̂_t^(i) + b^(i) )
    ```
    
2. Compute dynamic distance:
    
    ```
    D_i = (1/T) * Σ || x̂_t^(i) - x̂_t^(ref) ||₂
    ```
    

Select all solutions with:

- `L_i ≤ L_best + ε`
- `D_i` small (close to reference)

These form the *near-identical dynamics* solution set.

---

### **C. Quantify diversity in weight space**

For the selected subset:

- Compute:
    - Pairwise Euclidean distances: `||W^(i) - W^(j)||`
    - Pairwise cosine similarities
- Optionally plot:
    - Histogram of `||W^(i) - W^(ref)||`
    - Distribution of cosine similarities

Interpretation:

- If distances vary widely while MSE and dynamics remain similar → **non-identifiability**.

---

### **D. Connect to Gardner’s capacity framework**

Discussion points to include:

- In Gardner’s perceptron analysis, constraints define a **version space**: a high-dimensional manifold of weight vectors that satisfy the same functional behavior.
- Your empirical findings reveal:
    - Many different `W` realize almost the same dynamical behavior.
    - The “solution space” for your protein dynamics is a **broad region** in weight space.
- This suggests:
    - High structural non-identifiability
    - Architecture + data support **large solution manifolds**, reminiscent of high-capacity regimes

This fulfills your proposal requirement verbatim:

> Quantify structure–function non-identifiability by testing how many weight configurations yield near-identical dynamics, relating this to Gardner’s capacity framework.
> 

---

## **2.3 *(Optional)* Self-Learning Divergence**

Run the trained sigmoid AANN in two modes:

1. **Teacher-forced:**
    
    feed true `x_t` → predict `x_{t+1}`
    
2. **Self-generated:**
    
    feed predicted `x̂_t` back into the next step
    

Compute error:

```
E_t = || x̂_t^(self) - x_t^(true) ||
```

Plot:

- Teacher-forced vs self-generated trajectories
- Error vs rollout length

Interpretation:

- AANN may drift or diverge → matches paper’s findings that self-learning is unstable.

---

# **Day 3 — Fixed Points + Jacobian Stability (Required)**

## **3.1 Find fixed points**

For the non-recurrent sigmoid AANN:

- Iteratively compute:
    
    ```
    x_{k+1} = sigmoid( W @ x_k + b )
    ```
    
- Stop when:
    - `||x_{k+1} - x_k|| < 1e-5`
    - or max iterations reached

Record 1–3 fixed points (could be identical).

---

## **3.2 Compute Jacobian at fixed points**

For sigmoid activation:

```
z = W @ x* + b
D = diag( sigmoid(z) * (1 - sigmoid(z)) )
J = D @ W
```

For linear:

```
J = W
```

Compute eigenvalues:

```
λ = eigvals(J)
ρ = max(|λ|)
```

---

## **3.3 Analyze and visualize stability**

Plots:

- Eigenvalue scatter in complex plane (linear vs sigmoid)
- Histogram of |λ|
- Spectral radius comparison

Interpretation (connect to AM226):

- Sigmoid activation → shrinks effective Jacobian
- Stability corresponds to |λ| < 1 (local attractor)
- Dynamics resemble an asymmetric Hopfield-like attractor system
- Supports interpretation of AANNs as stable associative maps

---

# **Day 4 — Contrastive Hebbian Learning + Write-Up (Required)**

## Implement and compare with paper's Hebbian learning rule

- Paper uses delta hebbian rule
- Impplement this, CHL, and compare to SGD

## **4.1 Implement CHL**

For each training sample:

1. **Free phase**
    
    ```
    r ← initial state (e.g., x_t)
    run K_free iterations:  r = sigmoid(W r + b)
    record r^F
    ```
    
2. **Clamped phase**
    
    ```
    clamp output toward x_{t+1}
    run K_clamp iterations → r^C
    ```
    
3. **Weight update**
    
    ```
    ΔW ∝ (r^C r^Cᵀ - r^F r^Fᵀ)
    Δb ∝ (r^C - r^F)
    ```
    
- Mask ΔW using connectivity mask M
- Use small learning rate (1e-3 to 1e-4)

Train CHL for 10–20 epochs on a **small subset** of training data.

---

## **4.2 Compare CHL vs SGD**

Produce:

- MSE vs iteration (SGD vs CHL)
- Notes on:
    - Convergence speed
    - Stability
    - Biological plausibility

Interpretation:

- CHL approximates gradient descent in energy-based models
- CHL converges more slowly but follows biologically meaningful updates

---

## **4.3 Write-Up and Final Figures**

Sections:

1. **Introduction**
2. **Methods**
3. **Reproduction Results**
4. **Non-Identifiability & Gardner Capacity**
5. **Fixed-Point & Stability Analysis**
6. **Contrastive Hebbian Learning**
7. **Discussion**
8. **Conclusion**

Figures to include:

- Linear vs sigmoid MSE curves
- Recurrent vs non-recurrent comparison
- Trajectory overlays
- Weight-distance histogram (Gardner section)
- Dynamic-distance histogram
- Jacobian eigenvalue plots
- CHL vs SGD curves
- Optional: hybrid and self-learning divergence plots

---

# **Appendix — Optional Components**

If time allows (not required for proposal fulfillment):

### **Hybrid AANN**

- Per-node activation selection
- Hybrid vs sigmoid test MSE comparison

### **Self-Learning Divergence**

- Rollouts showing instability
- Error accumulation plots

Both are plug-and-play and do not affect the main schedule.