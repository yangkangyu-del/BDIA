# BDIA

Official MATLAB implementation of the **Bayesian Detection, Identification, and Adaptation (BDIA)** algorithm for linear observation models based on a rigorous **Branch and Bound (B&B)** tree search.

---

## 🌟 Key Features

- **Global Optimality Guaranteed**: Rigorously searches the binary hypothesis space to identify the exact Maximum A Posteriori (MAP) fault combination without relying on ad-hoc search depth caps ($q_{\max}$).
- **Two-Stage Pruning**: Employs monotonic upper bounds ($\overline{J}_{\text{loose}}$ and $\overline{J}_{\text{tight}}$) to safely prune unpromising subtrees (achieving 99.95% pruning rate on the 16-satellite dual-constellation benchmark).
- **Dual-Engine Execution**:
  - **Diagonal Fast Engine**: Algebraic reduction to $O(n^2)$ scalar updates for uncorrelated measurement noise and single-channel gross errors.
  - **General Matrix Engine**: Exact block recursions for block-correlated, cross-block independent faults (with occurrence probabilities $0 < \varepsilon_k < 0.5$).
- **Pure and Efficient**: Compact recursive DFS search engine (~150 lines core recursion, ~380 lines total with local linear algebra kernels). Typical execution time is ~9 ms (warm median; ~90 ms cold start).
- **Zero External Dependencies**: Self-contained pure MATLAB implementation requiring no additional packages.

---

## 🚀 Quick Start

```matlab
% Add BDIA directory to path
addpath('path/to/BDIA');

% 1. Nominal Weighted Least Squares (pass prior = 0):
sol_ls = bdia(y, A, D, 0);

% 2. Bayesian Fault Detection, Identification & Adaptation (pass prior probability or struct):
prior = 0.01; % Or: prior = struct('epsilon', 0.01);
[sol, info] = bdia(y, A, D, prior);

% Inspect results
fprintf('Identified Faulty Observations: %s\n', mat2str(sol.fault_set));
fprintf('Adapted State Vector x_MAP:\n');
disp(sol.x_map);
fprintf('Evaluated %d out of %d hypotheses (%.2f%% pruned)\n', ...
    info.N_computed, 2^length(y), 100 * (1 - info.N_computed / 2^length(y)));
```

To run the demonstration and self-verification suite:
```matlab
demo;
```

---

## 📐 Mathematical Overview

### 1. Observation and Fault Model
$$y = Ax + F(z \odot \nabla) + e, \quad e \sim \mathcal{N}(0, D)$$
where:
- $x \in \mathbb{R}^n$: unknown parameter vector.
- $z_k \in \{0, 1\} \sim \text{Bernoulli}(\varepsilon_k)$: binary fault indicator with prior occurrence probability $0 < \varepsilon_k < 0.5$.
- $\nabla_k \sim \mathcal{N}(\mu_{\nabla_k}, \Sigma_{\nabla_k\nabla_k})$: fault magnitude vector for block $k$, where fault blocks are internally correlated and cross-block independent (block-diagonal prior covariance $\Sigma_\nabla$).
- $e \sim \mathcal{N}(0, D)$: nominal Gaussian observation noise ($D$ positive definite, supporting both diagonal and fully correlated dense noise).

### 2. MAP Objective Function
The optimal fault hypothesis $H_{\text{MAP}}$ maximizes the log-posterior score:
$$H_{\text{MAP}} = \arg\max_{H_i} J_i$$
$$J_i = J_{\text{prior},i} + \frac{1}{2} J_{\text{geo},i} - \frac{1}{2} J_{\text{res},i}$$
where:
- $J_{\text{prior},i} = \ln P(H_i)$: prior fault log-probability.
- $J_{\text{geo},i} = \ln\det(W_i) - \ln\det(N_i)$: geometric information gain.
- $J_{\text{res},i} = \|y_i - A\hat{x}_i\|_{W_i}^2$: weighted residual quadratic form.

---

## 📁 Package Structure

BDIA follows a completely flat, self-contained architecture mirroring classical geodesy software packages (e.g., TU Delft's LAMBDA):

```text
BDIA/
├── bdia.m               % Main entry function
├── bdia_inputs.m        % Input validation and default prior construction
├── bdia_order.m         % Heuristic single-fault posterior ordering
├── bdia_search.m        % DFS Branch and Bound search engine
├── bdia_estimate.m      % State estimation, parameter covariance, and residuals
├── demo.m               % GNSS demonstration & multi-scenario self-verification
├── Contents.m           % Version and directory index
├── LICENSE              % BSD 3-Clause License terms
└── README.md            % Documentation and references
```

---

## 🗺️ Scope and Roadmap

BDIA provides the official reference implementation for Bayesian fault estimation and integrity monitoring based on the Bernoulli-Gaussian model:
- **v1.0 (Current Release)**: Exact Maximum A Posteriori (MAP) hypothesis search and state adaptation via Branch and Bound (Yu et al. 2025, Yu & Tan 2026).
- **Upcoming Releases**: Full Bayesian Receiver Autonomous Integrity Monitoring (BRAIM) module for Protection Level (PL) and Integrity Risk (IR) bounding (Yu et al. 2026).

---

## 📖 Citation

If you use BDIA or reference the Bernoulli-Gaussian framework in your research, please cite the following publications:

```bibtex
@article{yu2025bayesian,
  author={Yu, Yangkang and Yang, Ling and Shen, Yunzhong and El-Mowafy, Ahmed},
  journal={IEEE Transactions on Aerospace and Electronic Systems}, 
  title={Bayesian Fault Detection, Identification, and Adaptation for GNSS Applications}, 
  year={2025},
  volume={61},
  number={2},
  pages={1518--1535},
  doi={10.1109/TAES.2024.3456757}
}

@article{yu2026braim,
  author={Yu, Yangkang and Yang, Ling and Shen, Yunzhong and El-Mowafy, Ahmed and Li, Bofeng and Chen, Wu},
  journal={IEEE Transactions on Aerospace and Electronic Systems}, 
  title={Bayesian Receiver Autonomous Integrity Monitoring ({BRAIM}) Based on Bernoulli--Gaussian Model of Faults}, 
  year={2026},
  volume={62},
  pages={1164--1180},
  doi={10.1109/TAES.2025.3627566}
}

@article{yu2026efficient,
  author={Yu, Yangkang and Tan, Lu},
  journal={Measurement Science and Technology}, 
  title={Efficient Bayesian fault detection, identification, and adaptation based on branch and bound algorithm}, 
  year={2026},
  volume={37},
  number={25},
  pages={256303},
  doi={10.1088/1361-6501/ae7623}
}
```
