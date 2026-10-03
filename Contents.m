% BDIA
% Version 1.0.0 (R2021b) 03-Oct-2026
%
% Bayesian Detection, Identification, and Adaptation (BDIA).
%
% Main Entry Point
%   bdia          - Bayesian detection, identification, and adaptation
%
% Core Modules
%   bdia_inputs   - Validate inputs and configure fault prior models
%   bdia_order    - Heuristic single-fault posterior ordering
%   bdia_search   - Branch and bound tree search for global MAP hypothesis
%   bdia_estimate - State estimation, covariance extraction, and result formatting
%
% Demonstration and Verification
%   demo          - GNSS simulation demonstration and self-test verification
%
% Documentation and License
%   LICENSE       - BSD 3-Clause License terms
%
% References:
%   [1] Yu Y, Yang L, Shen Y, and El-Mowafy A (2025). Bayesian Fault
%       Detection, Identification, and Adaptation for GNSS Applications.
%       IEEE Transactions on Aerospace and Electronic Systems, 61(2): 1518-1535.
%       DOI: 10.1109/TAES.2024.3456757
%   [2] Yu Y, Yang L, Shen Y, El-Mowafy A, Li B, and Chen W (2026).
%       Bayesian Receiver Autonomous Integrity Monitoring (BRAIM) Based on
%       Bernoulli--Gaussian Model of Faults.
%       IEEE Transactions on Aerospace and Electronic Systems, 62: 1164-1180.
%       DOI: 10.1109/TAES.2025.3627566
%   [3] Yu Y, and Tan L (2026). Efficient Bayesian fault detection,
%       identification, and adaptation based on branch and bound algorithm.
%       Measurement Science and Technology, 37(25): 256303.
%       DOI: 10.1088/1361-6501/ae7623
