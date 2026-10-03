function [sol, info] = bdia(y, A, D, prior, options)
%BDIA Bayesian Detection, Identification, and Adaptation via Branch and Bound.
%
%   [sol, info] = bdia(y, A, D, prior, options) evaluates the optimal Maximum A
%   Posteriori (MAP) hypothesis for multiple outlier detection and adaptation
%   using a binary branch-and-bound tree search.
%
%   Model:
%       y = A*x + F*(z .* nabla) + e,   e ~ N(0, D)
%       z_k ~ Bernoulli(epsilon_k)
%       nabla_k ~ N(mu_nabla_k, Sigma_nabla_k)
%
%   Inputs:
%       y       - m-by-1 observation vector
%       A       - m-by-n design/sensitivity matrix
%       D       - m-by-m observation noise covariance matrix, or m-by-1 variances
%       prior   - (Required) Fault prior specification:
%                   * Pass 0: computes nominal Weighted Least Squares (WLS) solution.
%                   * Pass a numeric scalar/vector (e.g., 0.01): interpreted as
%                     prior fault occurrence probability epsilon.
%                   * Pass a struct with fields:
%                     .epsilon     : 1-by-p fault probabilities (e.g., 0.01; or 0 for nominal WLS)
%                     .F           : m-by-p fault design matrix (default: eye(m))
%                     .dim         : 1-by-p fault block dimensions (default: ones)
%                     .mu_nabla    : sum(dim)-by-1 fault means (default: zeros)
%                     .Sigma_nabla : fault prior covariance (default: (100^2)*eye)
%       options - (Optional) Struct with solver settings:
%                   .sort_mode   : 'single' (heuristic sorting, default) or 'original'
%                   .tol         : pruning tolerance (default: 1e-10)
%
%   Outputs:
%       sol     - Struct containing the optimal MAP solution:
%                   .x_map, .x      : n-by-1 adapted state estimate (with .x as convenient alias)
%                   .Dxx_map, .Dxx  : n-by-n posterior covariance matrix (with .Dxx as alias)
%                   .std_map        : n-by-1 standard deviations of x_map
%                   .fault_set      : indices of identified faulty observations/blocks
%                   .fault_detected : true if any fault is identified
%                   .residuals      : m-by-1 post-fit observation residuals
%                   .J_map          : optimal log-posterior objective score
%       info    - Struct with algorithm diagnostics:
%                   .N_computed     : number of evaluated hypotheses
%                   .sort_order     : fault processing order
%                   .diag_fast_mode : boolean indicating if fast diagonal engine was used
%                   .method         : 'branch_and_bound' or 'nominal_wls'
%                   .timing         : elapsed time in seconds
%
%   References:
%       [1] Yu Y, Yang L, Shen Y, and El-Mowafy A (2025). Bayesian Fault
%           Detection, Identification, and Adaptation for GNSS Applications.
%           IEEE Transactions on Aerospace and Electronic Systems, 61(2): 1518-1535.
%           DOI: 10.1109/TAES.2024.3456757
%       [2] Yu Y, Yang L, Shen Y, El-Mowafy A, Li B, and Chen W (2026).
%           Bayesian Receiver Autonomous Integrity Monitoring (BRAIM) Based on
%           Bernoulli--Gaussian Model of Faults.
%           IEEE Transactions on Aerospace and Electronic Systems, 62: 1164-1180.
%           DOI: 10.1109/TAES.2025.3627566
%       [3] Yu Y, and Tan L (2026). Efficient Bayesian fault detection,
%           identification, and adaptation based on branch and bound algorithm.
%           Measurement Science and Technology, 37(25): 256303.
%           DOI: 10.1088/1361-6501/ae7623

    if nargin < 4
        prior = [];
    end
    if nargin < 5 || isempty(options)
        options = struct();
    end

    t_start = tic;

    % 1. Input preprocessing and validation
    [model, prior, options] = bdia_inputs(y, A, D, prior, options);

    % 2. If no fault prior is active, return nominal Least Squares solution
    if ~prior.active
        sol = bdia_nominal_wls(model);
        info = struct();
        info.N_computed = 1;
        info.sort_order = [];
        info.diag_fast_mode = model.diag_fast_mode;
        info.method = 'nominal_wls';
        info.timing = toc(t_start);
        return;
    end

    % 3. Heuristic fault ordering (single-fault posterior sorting)
    [order, root_state, relaxed_root] = bdia_order(model, prior, options);

    % 4. Branch and bound search for global MAP hypothesis
    search_res = bdia_search(model, order, root_state, relaxed_root, options);

    % 5. State estimation and output packing
    [sol, info] = bdia_estimate(model, prior, order, search_res, options);

    info.method = 'branch_and_bound';
    info.timing = toc(t_start);
end

function sol = bdia_nominal_wls(model)
%BDIA_NOMINAL_WLS Compute nominal weighted least squares solution without fault hypotheses.
    y = model.y;
    A = model.A;
    if model.diag_fast_mode
        w = 1 ./ model.D_diag;
        A_white = A .* sqrt(w);
        y_white = y .* sqrt(w);
    else
        [R, flag] = chol(model.D_mat);
        if flag ~= 0
            error('bdia:InvalidCovariance', 'D must be symmetric positive definite.');
        end
        A_white = R' \ A;
        y_white = R' \ y;
    end

    [Q, Rn] = qr(A_white, 0);
    if rcond(Rn) < eps || any(abs(diag(Rn)) <= eps * max(abs(diag(Rn))))
        error('bdia:RankDeficientA', 'Design matrix A is rank deficient under D.');
    end
    x = Rn \ (Q' * y_white);
    Rn_inv = Rn \ eye(model.n);
    Dxx = 0.5 * (Rn_inv * Rn_inv' + (Rn_inv * Rn_inv')');
    residuals = y - A * x;

    if model.diag_fast_mode
        J_res = sum((residuals.^2) ./ model.D_diag);
    else
        J_res = residuals' * (model.D_mat \ residuals);
    end

    sol = struct();
    sol.x = x;
    sol.x_map = x;
    sol.Dxx = Dxx;
    sol.Dxx_map = Dxx;
    sol.std_map = sqrt(max(diag(Dxx), 0));
    sol.fault_set = [];
    sol.fault_detected = false;
    sol.residuals = residuals;
    sol.J_map = -0.5 * J_res;
end
