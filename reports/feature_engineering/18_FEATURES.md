# PathSnap — 18-Feature Map Matching Specification

**Architecture:** Distributed Map-Matching & Route Reconstruction  
**Total Features:** 18 (14 Observation-Level Classifier Emission Features + 4 Transition-Level HMM Features)  
**Target Variable:** `label` $\in \{0, 1\}$ (Observation-candidate route membership)  
**Leakage Policy:** Strictly causal ($t, t-1$ observations and directed road graph topology only; zero future GPS or reference-route leakage).

---

## 1. System Architecture Overview

The PathSnap map-matching pipeline decomposes into two complementary inference stages:

```
+--------------------------------------------------------------------------------------------------+
|                               STAGE 1: POINTWISE CANDIDATE SCORING                                |
| Input: GPS observation (t) + Spatial Road Candidates (R=50m)                                     |
| Model: Calibrated Random Forest / HistGradientBoosting Classifier                                |
| Features: 14 Observation-Level Kinematic & Geometric Features                                     |
| Output: Calibrated Emission Posterior Likelihoods P(y=1 | x_cand)                                |
+--------------------------------------------------------------------------------------------------+
                                                |
                                                v
+--------------------------------------------------------------------------------------------------+
|                               STAGE 2: HMM / VITERBI SEQUENTIAL DECODING                          |
| Input: Calibrated Emissions + Directed Road Graph Topology + Trajectory Steps (t-1 -> t)         |
| Engine: Dynamic Programming Trellis with Bounded Dijkstra Shortest Paths                         |
| Features: 4 Transition-Level Movement Plausibility Features (F15 - F18)                           |
| Output: Topologically Continuous, Globally Optimal Route Arc Sequence                            |
+--------------------------------------------------------------------------------------------------+
```

---

## 2. Complete 18-Feature Catalog

| # | Feature Name | Level | Category | Units | Mathematical Formula | Physical Purpose |
| :-: | :--- | :---: | :---: | :---: | :--- | :--- |
| **1** | `dist_to_arc` | Observation | Geometric Proximity | meters ($\text{m}$) | $\min_{s \in \text{arc}} \|\mathbf{p}_{\text{gps}} - s\|_2$ | Primary geometric distance to the road centerline. |
| **2** | `dist_to_from_node` | Observation | Geometric Endpoint | meters ($\text{m}$) | $\|\mathbf{p}_{\text{gps}} - \mathbf{u}_{\text{from}}\|_2$ | Proximity to intersection entry node; captures longitudinal position. |
| **3** | `dist_to_to_node` | Observation | Geometric Endpoint | meters ($\text{m}$) | $\|\mathbf{p}_{\text{gps}} - \mathbf{v}_{\text{to}}\|_2$ | Proximity to intersection exit node; captures segment convergence. |
| **4** | `proj_ratio` | Observation | Longitudinal Position | ratio $[0, 1]$ | $\frac{(\mathbf{p}_{\text{gps}} - \mathbf{u}) \cdot (\mathbf{v} - \mathbf{u})}{\|\mathbf{v} - \mathbf{u}\|_2^2}$ (clamped) | Normalized position along segment: $0$ (start), $0.5$ (mid), $1$ (end). |
| **5** | `gps_heading` | Observation | Kinematics | degrees $[0, 360)$ | $\text{atan2}(\Delta x_{\text{gps}}, \Delta y_{\text{gps}}) \pmod{360^\circ}$ | Instantaneous vehicle movement heading using strictly $t$ and $t-1$. |
| **6** | `arc_bearing` | Observation | Network Orientation | degrees $[0, 360)$ | $\text{atan2}(x_{\text{to}} - x_{\text{from}}, y_{\text{to}} - y_{\text{from}}) \pmod{360^\circ}$ | Permissible travel direction along the directed road arc. |
| **7** | `heading_diff` | Observation | Angular Alignment | degrees $[0, 180]$ | $\min(\|h_{\text{gps}} - b_{\text{arc}}\|, 360 - \|h_{\text{gps}} - b_{\text{arc}}\|)$ | Angle between travel heading and road bearing; filters cross-streets. |
| **8** | `is_reverse_direction` | Observation | Directional Filter | binary $\{0, 1\}$ | $\mathbb{I}(\text{heading\_diff} > 90^\circ)$ | Binary flag for counter-flow travel; strongly penalizes wrong-way arcs. |
| **9** | `speed_est` | Observation | Kinematics | $\text{m/s}$ | $\frac{\|\mathbf{p}_t - \mathbf{p}_{t-1}\|_2}{\Delta t}$ | Vehicle speed; separates stationary jitter from arterial cruising. |
| **10** | `time_gap` | Observation | Temporal State | seconds ($\text{s}$) | $t_t - t_{t-1}$ | Elapsed time interval; identifies GPS dropouts and tunnel passages. |
| **11** | `arc_length` | Observation | Network Geometry | meters ($\text{m}$) | $\|\mathbf{v}_{\text{to}} - \mathbf{u}_{\text{from}}\|_2$ | Segment length; distinguishes short intersection connectors from highways. |
| **12** | `candidate_rank` | Observation | Relative Margin | integer $[1, K]$ | $\text{rank}(\text{dist\_to\_arc})$ ascending | Invariant relative rank of candidate arc within observation choice set. |
| **13** | `candidate_count` | Observation | Network Density | count $\mathbb{Z}^+$ | $|\mathcal{C}(z_t)|$ | Local road network density; captures complex multi-level junctions. |
| **14** | `dist_to_alternative` | Observation | Decision Margin | meters ($\text{m}$) | $d_{\text{cand}, 2} - d_{\text{cand}, 1}$ (or search radius margin) | Margin between best and second-best candidate; measures ambiguity. |
| **15** | `gps_displacement_m` | Transition | Spatial Displacement | meters ($\text{m}$) | $\|\mathbf{p}_t - \mathbf{p}_{t-1}\|_2$ (projected UTM) | Physical vehicle displacement between consecutive GPS observations. |
| **16** | `network_transition_distance_m` | Transition | Network Topology | meters ($\text{m}$) | $\text{shortest\_path}(u_{\text{to}}(a_{t-1}) \to v_{\text{from}}(a_t))$ | Directed shortest path distance required to transition between road arcs. |
| **17** | `transition_distance_residual_m` | Transition | Movement Plausibility | meters ($\text{m}$) | $\|d_{\text{net}}(a_{t-1} \to a_t) - d_{\text{gps}}(z_{t-1} \to z_t)\|$ | Primary movement consistency signal: true routes satisfy $d_{\text{net}} \approx d_{\text{gps}}$. |
| **18** | `turn_angle_change_deg` | Transition | Angular Continuity | degrees $[0, 180]$ | $\min(\|b(a_t) - b(a_{t-1})\|, 360 - \|b(a_t) - b(a_{t-1})\|)$ | Directional deviation between consecutive arcs; prevents U-turn toggling. |

---

## 3. Deep-Dive on the Four Transition-Level Features

### Feature 15: `gps_displacement_m`
- **Formula:**
  $$f_{15} = \sqrt{(x_t - x_{t-1})^2 + (y_t - y_{t-1})^2}$$
- **Units:** Meters ($\text{m}$).
- **Handling Edge Cases:**
  - At $t=0$ (trip inception), $f_{15} = 0.0$.
  - When $\Delta t > 30.0\,\text{s}$ (GPS sampling interruption), the sequence resets and $f_{15}$ is not used for continuity.
- **Leakage Prevention:** Uses strictly observation $t$ and observation $t-1$. Zero future observations are referenced.

### Feature 16: `network_transition_distance_m`
- **Formula:**
  $$f_{16} = \begin{cases} 
  f_{15} & \text{if } a_{t-1} == a_t \text{ (same road segment)} \\
  \max(1.0, f_{15}) & \text{if } u_{\text{to}}(a_{t-1}) == v_{\text{from}}(a_t) \text{ (adjacent segments)} \\
  \text{DijkstraShortestPath}(u_{\text{to}}(a_{t-1}), v_{\text{from}}(a_t)) & \text{if path exists in } \mathcal{G}_{\text{directed}} \\
  \infty & \text{if unreachable in directed road graph}
  \end{cases}$$
- **Units:** Meters ($\text{m}$).
- **Handling Edge Cases:**
  - Bounded Dijkstra exploration is constrained to $\text{cutoff} = \max(150.0, 2.5 \times f_{15})$, guaranteeing real-time execution ($<0.5\,\text{s}$ per trip).
  - When unreachable, $f_{16} = \infty$. The system never fabricates an arbitrary path.

### Feature 17: `transition_distance_residual_m`
- **Formula:**
  $$f_{17} = |f_{16} - f_{15}| = |d_{\text{net}} - d_{\text{gps}}|$$
- **Units:** Meters ($\text{m}$).
- **Movement Plausibility Principle (Newson & Krumm 2009):**
  Vehicles drive along roads. Therefore, the road network distance between consecutive matched positions should closely match the physical straight-line GPS displacement. A large discrepancy indicates an impossible detour or shortcut.
- **Handling Edge Cases:**
  - If $f_{16} == \infty$ (unreachable transition), $f_{17} = \infty$.
  - In log-probability space, unreachable transitions incur a calibrated penalty $\lambda_{\text{unreachable}} = 5.0$, which prevents the trellis from being permanently destroyed by single GPS outliers.

### Feature 18: `turn_angle_change_deg`
- **Formula:**
  $$\Delta \theta = |b(a_t) - b(a_{t-1})|$$
  $$f_{18} = \min(\Delta \theta, 360^\circ - \Delta \theta) \in [0^\circ, 180^\circ]$$
- **Units:** Degrees ($^\circ$).
- **Anti-Zigzag & U-Turn Elimination:**
  On bi-directional two-way streets, reverse road arcs share endpoint nodes ($v_{\text{from}} == u_{\text{to}}$). Without Feature 18, a naive graph search considers transitioning from the forward arc to the reverse arc as "adjacent" because they share a node!
  Feature 18 calculates $f_{18} = 180.0^\circ$ (a direct U-turn). By penalizing turns quadratically:
  $$\text{Penalty}_{\text{turn}} = \left(\frac{f_{18}}{\theta_0}\right)^2 \quad (\theta_0 = 55.0^\circ)$$
  A $180^\circ$ U-turn incurs a heavy penalty of $(180/55)^2 = 10.71$ log units, completely eliminating erroneous back-and-forth toggling between opposing lanes.

---

## 4. Integration into HMM Viterbi Scoring Function

The complete log-posterior transition score between candidate $a_{t-1}$ and candidate $a_t$ is computed as:

$$\ln p(a_t \mid a_{t-1}) = \begin{cases}
0.0 & \text{if } \Delta t > 30.0\,\text{s (sequence reset)} \\
\text{stay\_bonus} - \frac{f_{15}}{50.0} & \text{if } a_{t-1} == a_t \\
-\lambda_{\text{unreachable}} - 0.5 \left(\frac{f_{18}}{\theta_0}\right)^2 & \text{if unreachable} \\
\text{adj\_bonus} \cdot \mathbb{I}(u_{\text{to}} == v_{\text{from}} \land f_{18} < 80^\circ) - \frac{f_{17}}{\beta} - \left(\frac{f_{18}}{\theta_0}\right)^2 - \text{pen}_{\text{speed}} & \text{otherwise}
\end{cases}$$

The Viterbi trellis dynamic programming update balances emission evidence with transition plausibility:

$$\text{Trellis}(t, a_t) = \max_{a_{t-1}} \left[ \text{Trellis}(t-1, a_{t-1}) + w_{\text{trans}} \cdot \ln p(a_t \mid a_{t-1}) \right] + w_{\text{emit}} \cdot \ln \hat{P}(y=1 \mid \mathbf{x}_{t, a_t})$$

where:
- $\beta = 20.0\,\text{m}$ (scale parameter for distance residual).
- $\theta_0 = 55.0^\circ$ (characteristic turn scale).
- $w_{\text{emit}} = 1.5$ (calibrated emission weight).
- $w_{\text{trans}} = 1.0$ (transition weight).
- $\lambda_{\text{unreachable}} = 5.0$ (bounded disconnect penalty).

