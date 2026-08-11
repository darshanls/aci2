# Design Document - PS11: Bayesian Network for Predictive Maintenance

**Group:** G091  
**Course:** Artificial and Computational Intelligence - Assignment 2  
**Institution:** BITS Pilani WILP

**Contributors:**
| Name | ID | Contribution |
|---|---|---|
| Darshan L S | 2025aa05828 | 100% |
| MAHADEVA SWAMY B N | 2025ab05081 | 100% |
| RANJITA PATEL | 2025ab05117 | 100% |
| SIHAAM S | 2025aa05354 | 100% |
| SPOORTHY N KUMAR | 2025ab05027 | 100% |

---

## Task A: PEAS Description for Predictive Maintenance Agent

### Performance Measure
- **Fault Detection Accuracy:** Percentage of true machine faults correctly identified before failure (precision and recall of posterior probability predictions).
- **False Alarm Rate:** Minimizing false positives where maintenance is triggered unnecessarily, reducing unplanned downtime costs.
- **Mean Time to Detection (MTTD):** Speed of identifying emerging faults from partial sensor data, measured in seconds from anomaly onset.
- **Posterior Probability Calibration:** How well the predicted probabilities (e.g., P(Bearing Failure | High Temp, High Vibration)) match the observed failure frequencies.
- **Maintenance Cost Reduction:** Measurable reduction in unplanned corrective maintenance events versus scheduled preventive actions.

### Environment
- **Industrial manufacturing floor** with rotating machinery (motors, compressors, pumps) subject to wear-based degradation.
- **Partially observable:** Not all machine states are directly measurable; hidden faults (bearing wear, cooling blockage) must be inferred from observable sensor readings.
- **Stochastic:** Sensor readings are noisy and probabilistic; identical fault conditions may produce varying sensor outputs.
- **Sequential:** Machine conditions evolve over time; current readings depend on historical degradation patterns.
- **Multi-agent (cooperative):** Multiple sensor channels operate simultaneously; their combined evidence is fused via Bayesian inference.

### Actuators
- **Maintenance Alert System:** Generates prioritized work orders when posterior failure probability exceeds configurable thresholds.
- **Dashboard Display:** Visual interface showing real-time posterior probabilities, joint probability tables, and conditional independence analysis.
- **Automated Shutdown Signal:** Emergency stop command issued when P(Failure | Evidence) exceeds a critical safety threshold.
- **Report Generator:** Produces structured output files (outputPS11.txt) with inference results for audit and review.

### Sensors
- **Temperature Sensors:** Continuous thermal monitoring of bearings, motors, and cooling systems (binary thresholded: High/Normal).
- **Vibration Sensors:** Accelerometers mounted on rotating equipment detecting abnormal vibration patterns (binary thresholded: High/Normal).
- **Coolant Flow Sensors:** Monitoring cooling system operation, triggering alarms on flow/temperature anomalies.
- **Redundant Sensor Channels:** Independent sensor arrays (S1, S2) providing parallel monitoring for fault-tolerant detection.

---

## Task B: Bayesian Network Implementation Design

### Architecture Overview

The implementation uses a single reusable `BayesianNetwork` class that models any three-variable network with the structure **Root -> Child1, Root -> Child2**. This design is parameterized and not hard-coded to specific scenarios.

### Data Structures

1. **Joint Probability Table:** A list of tuples `[(root_val, child1_val, child2_val, probability), ...]` storing all 2^3 = 8 rows of the complete joint distribution. Constructed once during initialization using the factorization P(Root, C1, C2) = P(Root) * P(C1|Root) * P(C2|Root).

2. **Conditional Probability Tables (CPTs):** Python dictionaries mapping parent values {True, False} to conditional probabilities. Example: `{True: P(T|B=True), False: P(T|B=False)}`.

3. **Evidence Dictionary:** A dict mapping variable names to observed values, enabling flexible query specification: `{"T": True, "V": True}`.

### Inference Algorithm

The inference engine uses **exact inference via joint enumeration**:

```
P(Query=True | Evidence) = SUM[consistent rows where Query=True] / SUM[all consistent rows]
```

For each row in the joint table:
1. Check if the row is consistent with all evidence variables.
2. If consistent, add the probability to the denominator.
3. If consistent AND the query variable is True, add to the numerator.
4. Return numerator / denominator.

**Time Complexity:** O(2^n) where n is the number of variables (n=3 in our case, so 8 iterations per query).

**Space Complexity:** O(2^n) for storing the joint table.

### Independence Testing

- **Marginal Independence:** Check whether P(X, Y) = P(X) * P(Y) by computing marginals from the joint table with floating-point tolerance (epsilon = 1e-9).
- **Conditional Independence:** For each value of the conditioning variable, check P(X, Y | Z=z) = P(X|Z=z) * P(Y|Z=z). Both Z=True and Z=False must satisfy this condition.

### Input Validation

The parser validates:
- File existence and readability
- Correct scenario identifiers (rejects unknown identifiers)
- Key=value format on all parameter lines
- No duplicate keys within a scenario
- Probability values are numeric and within [0, 1]
- All required keys are present for each scenario

---

## Task C: Comparative Analysis

### 1. Probabilistic Reasoning vs. Fixed IF-THEN Rules

| Aspect | Bayesian Network | Rule-Based System |
|---|---|---|
| **Uncertainty Handling** | Quantifies uncertainty with probabilities (e.g., P(B\|T)=0.27) | Binary decisions only (IF temp>threshold THEN fault) |
| **Gradual Degradation** | Captures incremental probability changes as evidence accumulates | Rigid thresholds; either fires or does not |
| **Parameter Learning** | CPTs can be learned from data using MLE or Bayesian estimation | Rules must be hand-crafted by domain experts |
| **Output Interpretability** | Produces calibrated probabilities for risk assessment | Produces deterministic yes/no verdicts |

### 2. Handling Uncertain, Incomplete, or Overlapping Symptoms

Bayesian Networks excel when sensor data is **partial or ambiguous**:
- With only temperature evidence: P(B|T) = 0.2698 (moderate concern).
- Adding vibration evidence: P(B|T,V) = 0.6892 (high concern).
- This **evidence accumulation** is impossible in pure rule-based systems without combinatorial rule explosion.

When symptoms overlap (e.g., high temperature could indicate bearing failure OR cooling failure), Bayesian Networks use the network structure to correctly attribute probabilities, while rule-based systems would need explicit disambiguation rules for every combination.

### 3. Impact of Conditional Dependencies on Inference Accuracy

In our Scenario 1, Temperature and Vibration are **conditionally independent given Bearing Failure** but **not marginally independent**. This structure means:
- Observing high temperature **increases** the probability of high vibration (via the explaining-away effect through the common cause B).
- If a rule-based system ignores this dependency, it would either double-count evidence or miss the correlation entirely.
- The factorization P(B,T,V) = P(B)*P(T|B)*P(V|B) exploits conditional independence to accurately represent the joint distribution with only 5 parameters instead of 7.

### 4. Effect of Evidence Propagation on Posterior Probabilities

Evidence propagation demonstrates the **explaining-away** phenomenon:
- **Single evidence:** P(B|T)=0.2698, P(B|V)=0.3429
- **Combined evidence:** P(B|T,V)=0.6892

The combined posterior is significantly higher than either individual posterior, demonstrating how Bayesian inference correctly fuses multiple evidence sources. In Scenario 2, P(C|S1,S2) = 0.8115 is much higher than P(C|S1)=0.3286 or P(C|S2)=0.3597, showing how redundant sensor confirmation dramatically increases diagnostic confidence.

### 5. Role of Conditional Independence and d-Separation

**d-Separation** in our networks (B->T, B->V): T and V are d-separated given B (the common cause is observed), making them conditionally independent. This has critical computational benefits:

- **Parameter Reduction:** The joint distribution P(B,T,V) requires only 5 parameters (P(B), P(T|B), P(T|~B), P(V|B), P(V|~B)) instead of 2^3 - 1 = 7 for a full joint table.
- **Scalability:** For n sensor variables with a single root cause, the Bayesian Network needs O(n) parameters vs O(2^n) for a full joint distribution.
- **Inference Efficiency:** Conditional independence enables variable elimination and belief propagation algorithms that avoid enumerating the entire joint space in larger networks.

---

## Alternate Modeling Approach: Naive Bayes Classifier

### Description
An alternative approach models the problem as a **Naive Bayes Classifier** where the root variable (Bearing Failure / Cooling Failure) is the class label and sensor readings are features. This assumes all features are conditionally independent given the class.

### Implementation Difference
Instead of constructing the full joint table, the Naive Bayes approach directly computes:

```
P(Class | Features) = P(Class) * PRODUCT[P(Feature_i | Class)] / P(Features)
```

### Performance Implications
- **Advantage:** O(n) computation per query (linear in number of features) without building the joint table. For our 3-variable network this is negligible, but for networks with 20+ sensors this becomes significant.
- **Advantage:** Trivial parameter estimation from data; each CPT is estimated independently.
- **Limitation:** Cannot represent networks where children have multiple parents or where features have inter-dependencies beyond the single root cause.
- **Limitation:** For our specific problem structure (single root, two children, conditional independence holds), the Naive Bayes result is **identical** to the Bayesian Network result. However, the Bayesian Network framework generalizes to more complex structures (e.g., chains, polytrees) where Naive Bayes cannot.

In summary, the Bayesian Network approach provides a more general and expressive framework at the cost of slightly higher computational complexity, making it the preferred choice for real-world predictive maintenance systems where fault relationships are complex and multi-layered.
