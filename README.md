# Black-Box Optimisation Capstone Project

## Section 1: Project Overview

### What is this project?

This project is a sequential black-box optimisation (BBO) challenge completed as the capstone component of a machine learning programme. The challenge involves querying eight unknown mathematical functions — one query per function per round — with the goal of finding inputs that maximise each function's output. The functions are entirely opaque: no formula, gradient, or structural information is provided. The only feedback available is the output value returned after each query, making this a pure trial-and-learn problem.

The eight functions vary in dimensionality (from 2 to 8 input dimensions), domain (all inputs constrained to [0, 1]), and behaviour — some are smooth and well-behaved, others are volatile, multimodal, or near-zero across most of their input space. This variety means no single strategy works uniformly across all functions; each requires its own evolving approach based on accumulated evidence.

### Why is this relevant in real-world ML?

Black-box optimisation is one of the most practically important problems in applied machine learning and engineering. Anytime a system must be tuned without access to its internals — hyperparameter search for neural networks, drug dosage optimisation in clinical trials, material composition search in chemistry, or A/B testing in product development — the underlying problem is BBO. Gradients are unavailable, evaluations are expensive, and decisions must be made sequentially under uncertainty.

The core tension in all of these settings is the same as in this challenge: how do you extract maximum information from a limited number of expensive evaluations? Solving this well requires combining probabilistic modelling, principled uncertainty quantification, and adaptive decision-making — exactly the skills this project develops.

### Career relevance

The practical skills built here map directly onto roles in data science, ML engineering, and quantitative research. The ability to design and justify a sequential decision-making strategy — documenting why each choice was made, what evidence drove it, and how the approach adapted to new information — is precisely the kind of structured thinking that distinguishes strong ML practitioners from those who simply run standard pipelines. This project also provides a concrete, end-to-end example of Bayesian optimisation implemented from scratch, which is directly applicable to AutoML, experimental design, and any domain where evaluations carry a real cost.

---

## Section 2: Inputs and Outputs

### Inputs

Each function accepts a vector of continuous values, with all dimensions constrained to the interval [0, 1]. The dimensionality varies by function:

| Function | Dimensions | Domain   | Context Label                         |
|----------|------------|----------|---------------------------------------|
| F1       | 2          | [0,1]²   | Radiation field (2 spatial inputs)    |
| F2       | 2          | [0,1]²   | ML log-likelihood (2 parameters)      |
| F3       | 3          | [0,1]³   | Drug discovery (3 compounds)          |
| F4       | 4          | [0,1]⁴   | Warehouse optimisation (4 params)     |
| F5       | 4          | [0,1]⁴   | Chemical yield (4 chemicals)          |
| F6       | 5          | [0,1]⁵   | Cake recipe (5 ingredients)           |
| F7       | 6          | [0,1]⁶   | ML hyperparameters (6 settings)       |
| F8       | 8          | [0,1]⁸   | Neural network config (8 dimensions)  |

Each submission provides one query vector per function per round. Example submission format:

```
Function 1: [0.606061, 0.626263]
Function 2: [0.694429, 0.853313]
Function 3: [0.454254, 0.950329, 0.514089]
Function 4: [0.409969, 0.408553, 0.459897, 0.407688]
Function 5: [0.451234, 0.902540, 0.919484, 0.997644]
Function 6: [0.524435, 0.433417, 0.639900, 0.760149, 0.000000]
Function 7: [0.040000, 0.154068, 0.994523, 0.199378, 0.334133, 0.818653]
Function 8: [0.166499, 0.141044, 0.140668, 0.000000, 1.000000, 0.711852, 0.173018, 0.251532]
```

### Outputs

Each query returns a single scalar value — the function's output at the submitted input point. No gradient, uncertainty estimate, or structural information accompanies the return value. The output may be positive, negative, near-zero, or extreme depending on the function and the query location. Example outputs for the above submission:

```
Function 1: 1.176e-177   (effectively zero)
Function 2: 0.6117
Function 3: -0.0303
Function 4: -0.3250
Function 5: 2639.20
Function 6: -0.3889
Function 7: 1.1721
Function 8: 9.8898
```

The goal is to drive these output values as high as possible over the course of the challenge.

---

## Section 3: Challenge Objectives

### Primary objective

The goal is to **maximise** the output of each function by selecting input vectors that return progressively higher values. There is no single combined score — each function is optimised independently, and progress is measured by how close each function's best observed output comes to its (unknown) true maximum.

### Constraints and limitations

**Query budget.** One query per function per round. This is the defining constraint of the challenge. Every query must be justified — there is no opportunity to probe a region speculatively and correct immediately. The cost of a poorly chosen query is an entire round of lost information.

**No function access.** The true function, its formula, its derivatives, and its global structure are entirely unknown. The only information available is the set of (input, output) pairs accumulated across all previous rounds. This rules out gradient-based optimisation and any approach that requires evaluating the function at many points cheaply.

**Fixed input domain.** All inputs must lie in [0, 1] for every dimension. Queries outside this range are invalid.

**Response delay.** Results from one round are required before the next round's queries can be submitted. The optimisation is strictly sequential — batch or parallel strategies within a round are not possible.

**Unknown noise structure.** It is not known in advance whether the functions are deterministic or stochastic. Some functions appear to return consistent values for similar inputs; others exhibit variability that suggests either noise or high sensitivity to small input changes.

---

## Section 4: Technical Approach

### Overview

The core framework is **Bayesian optimisation (BO)** — a principled approach to sequential decision-making under uncertainty that is well-suited to expensive, black-box function evaluation. BO works by maintaining a probabilistic surrogate model of the unknown function, using that model to estimate the expected value of querying any point in the input space, and selecting the query that maximises an acquisition function balancing exploration and exploitation.

### Surrogate model: Gaussian Process Regression

A **Gaussian Process (GP)** is used as the surrogate model for each function. A GP places a probability distribution over possible functions consistent with the observed data, providing not just a predicted mean at any unobserved point but also a calibrated uncertainty estimate. This uncertainty is what makes the acquisition function meaningful — it allows the model to identify regions that are either predicted to be good (exploitation) or uncertain enough to be worth exploring (exploration).

The GP kernel is **Matérn (ν = 2.5)** for most functions, chosen because it assumes a moderate degree of smoothness — twice differentiable — which is more appropriate for real-world functions than the infinitely smooth RBF kernel. For functions whose output history suggests a smooth, well-behaved landscape, the RBF kernel is used instead. Both kernels incorporate **automatic relevance determination (ARD)**, meaning a separate length scale is fitted per input dimension. This allows the GP to infer which dimensions are most sensitive — dimensions with small fitted length scales drive the function's behaviour, while those with large length scales contribute little and can be searched more coarsely.

A **WhiteKernel** noise term is added to account for output variability, with its magnitude tuned per function based on the observed consistency of outputs across rounds.

### Acquisition function: Expected Improvement

**Expected Improvement (EI)** is used as the acquisition function. EI computes the expected amount by which a candidate query would exceed the current best observed output, integrating over the GP's predictive uncertainty. It is zero for points the model is confident are worse than the current best, and positive for points where either the mean prediction is high or the uncertainty is large enough to make improvement plausible.

The **exploration-exploitation trade-off** is governed by the `xi` parameter in the EI formula. High `xi` inflates the improvement threshold, favouring exploration of uncertain regions. Low `xi` favours tight exploitation near the current best. This parameter is set per function and updated each round based on observed convergence:

- Functions with clean, consistent improvement receive progressively lower `xi` values as confidence in the optimum's location grows.
- Functions with volatile or non-monotone output histories receive higher `xi` values to prevent overcommitting to a region that may sit on a ridge or near a discontinuity.
- Functions that fail to improve over multiple consecutive rounds receive an emergency increase in `xi`, deliberately breaking out of the local exploitation trap.

EI is maximised numerically using **L-BFGS-B** with multiple random restarts, scaled with dimensionality (150 restarts for 2D functions, up to 400 for 8D) to reduce the risk of the optimiser settling in a local maximum of the acquisition surface.

### Search space management: zoom and hard constraints

Rather than searching the full [0, 1] hypercube at every round, a **zoom** strategy narrows the search bounds to a region around the current best-known point once promising regions are identified. The zoom radius is set per function and shrinks as convergence evidence accumulates. This concentrates the query budget in regions the data supports, rather than wasting queries in areas already shown to be poor.

Where domain knowledge or consistent data patterns support it, certain dimensions are **hard-constrained** to subregions regardless of what the GP recommends. For example, an ingredient that has been near-zero in every good result is constrained to remain near zero; a dimension where the boundary value consistently outperforms interior values is pinned to that boundary. These constraints encode accumulated evidence in a way that prevents the GP from reversing well-established findings due to limited data in later rounds.

### Per-function hyperparameter differentiation

A key feature of the approach is that every function has its own hyperparameter profile — `xi`, zoom radius, kernel type, noise level, and EI restart count — rather than a shared configuration. This reflects the fundamental reality that the eight functions have different dimensionalities, noise levels, landscape shapes, and convergence behaviours. Treating them uniformly would over-exploit volatile functions and under-exploit well-behaved ones.

The hyperparameter choices are updated each round based on the function's observed behaviour, following explicit rules:

| Signal observed               | Adjustment made                               |
|-------------------------------|-----------------------------------------------|
| Consistent improvement        | Reduce `xi`, narrow zoom                      |
| Single regression             | Return to prior best; raise noise level       |
| Repeated regressions          | Raise `xi`, widen zoom — escape local trap    |
| Output plateau (3+ rounds)    | Moderate `xi` increase — probe adjacent region|
| Boundary value performs best  | Hard-constrain that dimension near boundary   |
| Large ARD length scale        | Reduce zoom in that dimension (near-irrelevant)|

### Evolution across rounds

**Rounds 1–2 (Orientation):** Queries were distributed broadly across each function's input space, informed by the initial training data provided at the start of the challenge. The primary goal was to build enough observations to fit a meaningful GP posterior. Kernel hyperparameters were kept at defaults; no zoom was applied.

**Rounds 3–5 (Model-guided exploitation):** With 10–15 observations per function, the GP posteriors became informative enough to direct queries meaningfully. Zoom was introduced for functions showing clear promising regions. Per-function kernel and noise settings were differentiated for the first time. EI restart counts were scaled with dimensionality.

**Rounds 6–9 (Adaptive refinement):** Per-function strategies diverged significantly. Functions with strong convergence signals moved to tight exploitation with hard-constrained dimensions. Functions with volatile histories shifted to robustness-weighted settings. Functions failing to improve for multiple rounds received emergency widening of search bounds. ARD length scales were inspected each round as an emergence detector — sudden changes in fitted length scales signal a shift in which dimensions are driving the function, prompting a strategy review.

### Reflection on assumptions and limitations

The approach rests on a smoothness assumption — the GP kernel assumes nearby inputs produce similar outputs. This holds for well-behaved functions but fails for landscapes with sharp ridges or discontinuities, producing overconfident recommendations. With only 8–15 observations per function, kernel hyperparameters are underdetermined and posterior uncertainty estimates cannot be fully trusted. High-dimensional functions (6–8 inputs) are massively undersampled relative to their space, meaning the GP is extrapolating rather than interpolating across most of its domain. These limitations are partially mitigated by conservative noise settings and wider exploration parameters for the more complex functions, but they cannot be fully resolved within the query budget available.

---

## Repository Structure

```
├── notebooks/
│   ├── Submission1.ipynb
│   ├── Submission2.ipynb
│   └── ...
├── initial_data/
│   └── function_{1..8}/
│       ├── initial_inputs.npy
│       └── initial_outputs.npy
├── results/
│   └── round_history.csv
└── README.md
```

---

## Dependencies

```
numpy
scipy
scikit-learn
matplotlib
```

---


