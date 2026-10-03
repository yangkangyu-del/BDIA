%DEMO Demonstration and verification script for BDIA.
%
%   Part 1: Dual-constellation GNSS outlier detection & adaptation demo
%   Part 2: Multi-scenario self-verification suite (1D, 3D blocks, correlated)

clear; clc; close all;
rng(2026);

fprintf('=================================================================\n');
fprintf('                       BDIA DEMONSTRATION                        \n');
fprintf('=================================================================\n\n');

%% PART 1: Dual-Constellation GNSS Outlier Detection
fprintf('--- PART 1: Dual-Constellation GNSS Positioning Demo ---\n');

% 10 GPS + 6 Galileo satellites; 3 ENU coordinates + 2 receiver clock biases
A = [-0.4804 -0.4041  0.7784 1 0;
     -0.4613 -0.8637 -0.2030 1 0;
     -0.8912 -0.1088 -0.4403 1 0;
      0.2348 -0.7795  0.5808 1 0;
     -0.5795  0.1434  0.8023 1 0;
     -0.1086 -0.4612  0.8806 1 0;
     -0.2755 -0.9339  0.2280 1 0;
      0.6320 -0.7695  0.0916 1 0;
     -0.5121  0.2498  0.8218 1 0;
     -0.9658  0.2451  0.0852 1 0;
     -0.6431 -0.7625 -0.0708 0 1;
     -0.1980 -0.4902 -0.8488 0 1;
      0.3479 -0.8721 -0.3441 0 1;
     -0.9670  0.2012  0.1564 0 1;
     -0.0484 -0.7085 -0.7040 0 1;
     -0.7658 -0.1686  0.6206 0 1];

[m, n] = size(A);
D_diag = [6.1194, 6.1261, 6.6275, 6.2302, 6.4608, 6.1944, 6.0688, 6.8999, ...
          6.6699, 6.7275, 10.0650, 11.4017, 10.6736, 10.5657, 10.7998, 10.1490];
D = diag(D_diag);

x_true = [0.0; 0.0; 0.0; 10.0; 12.0];
y = A * x_true + randn(m, 1) .* sqrt(D_diag(:));

% Inject 2 gross errors
true_outliers = [3, 11];
y(true_outliers) = y(true_outliers) + [35.0; -45.0];

% 1. Conventional Least Squares (prior = 0 indicates zero fault probability)
[sol_ls, info_ls] = bdia(y, A, D, 0);
pos_err_ls = norm(sol_ls.x_map(1:3) - x_true(1:3));

% 2. BDIA via Branch and Bound (With satellite fault prior probability)
prior = struct();
prior.epsilon = 0.01; % Satellite fault prior occurrence probability (1%)
[sol_bdia, info_bdia] = bdia(y, A, D, prior);
pos_err_bdia = norm(sol_bdia.x_map(1:3) - x_true(1:3));

% Verify alias compatibility (.x and .Dxx)
assert(isequal(sol_ls.x, sol_ls.x_map) && isequal(sol_ls.Dxx, sol_ls.Dxx_map), 'LS alias failed.');
assert(isequal(sol_bdia.x, sol_bdia.x_map) && isequal(sol_bdia.Dxx, sol_bdia.Dxx_map), 'BDIA alias failed.');

fprintf('%-25s | %-18s | %-18s\n', 'Metric', 'Conventional LS', 'BDIA');
fprintf('-----------------------------------------------------------------\n');
fprintf('%-25s | %-18s | %-18s\n', 'Fault Prior (epsilon)', '0 (Nominal LS)', num2str(prior.epsilon));
fprintf('%-25s | %-18s | %-18s\n', 'Identified Outliers', 'None (Ignored)', mat2str(sol_bdia.fault_set));
fprintf('%-25s | %-18.4f | %-18.4f\n', '3D Position Error (m)', pos_err_ls, pos_err_bdia);
fprintf('%-25s | %-18d | %-18d\n', 'Evaluated Hypotheses', info_ls.N_computed, info_bdia.N_computed);
fprintf('%-25s | %-18.2f ms | %-18.2f ms\n', 'Elapsed Time', info_ls.timing * 1000, info_bdia.timing * 1000);
fprintf('-----------------------------------------------------------------\n\n');

%% PART 2: Multi-Scenario Self-Verification Suite
fprintf('--- PART 2: Multi-Scenario Self-Verification ---\n');

% Case A: Multi-dimensional 3D Coordinate Anomaly (d_k = 3)
A_a = randn(10, 4);
D_a = diag(rand(10, 1) + 0.5);
x_a = [1; 2; 3; 4];
y_a = A_a * x_a + 0.1 * randn(10, 1);
prior_a = struct('dim', [3, 1], 'F', [eye(3), zeros(3, 1); zeros(7, 3), [1; zeros(6, 1)]], ...
                 'epsilon', [0.05, 0.05], 'mu_nabla', zeros(4, 1), 'Sigma_nabla', diag([100^2, 100^2, 100^2, 50^2]));
y_a(1:3) = y_a(1:3) + [30; -25; 40]; % Inject 3D fault on block 1
sol_a = bdia(y_a, A_a, D_a, prior_a);
assert(isequal(sol_a.fault_set(:), 1), 'Multi-dimensional 3D block test failed.');
fprintf('  [1/3] Multi-dimensional 3D block fault detection: PASSED\n');

% Case B: Correlated Observations (Dense Covariance D)
tmp = randn(8, 8);
D_dense = 0.5 * (tmp * tmp') + eye(8);
A_b = randn(8, 3);
y_b = A_b * [1; 1; 1] + 0.1 * randn(8, 1);
y_b(4) = y_b(4) + 30; % Outlier on obs 4
prior_b = struct('epsilon', 0.02);
sol_b = bdia(y_b, A_b, D_dense, prior_b);
assert(isequal(sol_b.fault_set(:), 4), 'Correlated covariance test failed.');
fprintf('  [2/3] Correlated observation covariance handling: PASSED\n');

% Case C: Distributed Multi-channel Common-mode Fault
A_c = randn(8, 3);
D_c = diag(rand(8, 1) + 1);
y_c = A_c * [2; 0; -1] + 0.1 * randn(8, 1);
F_c = zeros(8, 2); F_c(1:4, 1) = 1; F_c(7, 2) = 1;
prior_c = struct('dim', [1, 1], 'F', F_c, 'epsilon', [0.03, 0.03], ...
                 'mu_nabla', [0; 0], 'Sigma_nabla', diag([100^2, 100^2]));
y_c(1:4) = y_c(1:4) + 35; % Common-mode fault
sol_c = bdia(y_c, A_c, D_c, prior_c);
assert(isequal(sol_c.fault_set(:), 1), 'Distributed fault test failed.');
fprintf('  [3/4] Multi-channel distributed fault detection : PASSED\n');

% Case D: Nominal Weighted Least Squares Verification (prior = 0)
[sol_d, info_d] = bdia(y_c, A_c, D_c, 0);
assert(isempty(sol_d.fault_set) && ~sol_d.fault_detected, 'Nominal LS should report 0 faults.');
assert(strcmp(info_d.method, 'nominal_wls') && info_d.N_computed == 1, 'Nominal LS info mismatch.');
fprintf('  [4/7] Nominal Weighted Least Squares (prior = 0) : PASSED\n');

% Case E: Input Validation Verification (Defense against invalid inputs)
error_missing_prior = false;
try, bdia(y_c, A_c, D_c); catch ME, if strcmp(ME.identifier, 'bdia:MissingPrior'), error_missing_prior = true; end; end
error_rank_deficient = false;
try, bdia(y_c, ones(size(A_c, 1), 2), D_c, 0.01); catch ME, if strcmp(ME.identifier, 'bdia:RankDeficientA'), error_rank_deficient = true; end; end
error_ill_conditioned = false;
try, bdia([0; 1; 0], [1, 1; 0, 1e-8; 0, 0], eye(3), 0.01); catch ME, if strcmp(ME.identifier, 'bdia:IllConditionedSystem'), error_ill_conditioned = true; end; end
error_cross_block = false;
try, bdia(y_c, A_c, D_c, struct('dim', [1 1], 'F', eye(size(y_c, 1), 2), 'Sigma_nabla', [1 0.5; 0.5 1])); catch ME, if strcmp(ME.identifier, 'bdia:InvalidPriorCovariance'), error_cross_block = true; end; end
error_asym_d = false;
try, bdia(y_c, A_c, [1 100 0; -100 1 0; 0 0 1], 0.01); catch ME, if strcmp(ME.identifier, 'bdia:InvalidD'), error_asym_d = true; end; end
error_empty_eps = false;
try, bdia(y_c, A_c, D_c, struct('epsilon', [])); catch ME, if strcmp(ME.identifier, 'bdia:InvalidPrior'), error_empty_eps = true; end; end
error_dep_qmax = false;
try, bdia(y_c, A_c, D_c, 0.01, struct('q_max', 0)); catch ME, if strcmp(ME.identifier, 'bdia:DeprecatedOption'), error_dep_qmax = true; end; end

assert(error_missing_prior && error_rank_deficient && error_ill_conditioned && ...
       error_cross_block && error_asym_d && error_empty_eps && error_dep_qmax, 'Input validation checks failed.');
fprintf('  [5/7] Defensive input validation (fail-fast)     : PASSED\n');

% Case F: Numerical Resilience (Extreme variance & column permutation)
% 1. Extreme variance in diagonal mode
A_num = [randn(5, 2); ones(1, 2)];
y_num = A_num * [1; 2] + 0.1 * randn(6, 1);
y_num(2) = y_num(2) + 30; % Outlier at obs 2
sol_var16 = bdia(y_num, A_num, eye(6), struct('epsilon', 0.05, 'Sigma_nabla', 1e16));
assert(isequal(sol_var16.fault_set(:), 2), 'Extreme variance 1e16 test failed.');

% 2. Coordinate translation invariance (shift 1e8)
y_shifted = y_num + 1e8;
A_shifted = [A_num, ones(6, 1)];
sol_orig = bdia(y_num, A_shifted, eye(6), 0.05);
sol_shift = bdia(y_shifted, A_shifted, eye(6), 0.05);
assert(isequal(sol_orig.fault_set, sol_shift.fault_set), 'Large translation 1e8 test failed.');

% 3. General engine column-permuted counterexample with 1e16 variance
F_perm = eye(3); F_perm = F_perm(:, [2, 1, 3]);
prior_perm = struct('F', F_perm, 'dim', [1, 1, 1], 'epsilon', 0.01, 'Sigma_nabla', 1e16 * eye(3));
sol_perm = bdia([20; 0; 0], ones(3, 1), eye(3), prior_perm);
assert(isequal(sol_perm.fault_set(:), 2) && abs(sol_perm.J_map - (-23.3925251919)) < 1e-8, ...
       'General engine permutation counterexample failed.');
fprintf('  [6/7] Numerical resilience (var=1e16, shift=1e8) : PASSED\n');

% Case G: Independent Full Brute-Force Ground Truth (Non-zero Mean mu_nabla)
p_bf = 6; m_bf = 8; n_bf = 2;
A_bf = randn(m_bf, n_bf);
D_bf = diag(0.8 + 0.4 * rand(m_bf, 1));
prior_bf = struct('dim', ones(1, p_bf), 'F', eye(m_bf, p_bf), 'epsilon', 0.05 * ones(1, p_bf), ...
                  'mu_nabla', 5 * randn(p_bf, 1), 'Sigma_nabla', diag(25 + 10 * rand(p_bf, 1)));
y_bf = A_bf * [1; -1] + 0.2 * randn(m_bf, 1);
y_bf([2, 5]) = y_bf([2, 5]) + prior_bf.mu_nabla([2, 5]) + [25; -30];
sol_bf_test = bdia(y_bf, A_bf, D_bf, prior_bf);

% Run independent full enumeration over all 2^p = 64 hypotheses
best_J_exact = -inf; best_mask_exact = false(1, p_bf);
for h = 0:(2^p_bf - 1)
    mask_h = logical(bitget(h, 1:p_bf));
    f_h = find(mask_h);
    if isempty(f_h)
        W_h = inv(D_bf); y_h = y_bf;
    else
        Sig_h = D_bf; y_h = y_bf;
        for kk = f_h
            Sig_h = Sig_h + prior_bf.F(:, kk) * prior_bf.Sigma_nabla(kk, kk) * prior_bf.F(:, kk)';
            y_h = y_h - prior_bf.F(:, kk) * prior_bf.mu_nabla(kk);
        end
        W_h = inv(Sig_h);
    end
    N_h = A_bf' * W_h * A_bf;
    x_h = N_h \ (A_bf' * W_h * y_h);
    res_h = y_h - A_bf * x_h;
    J_res_h = res_h' * W_h * res_h;
    lp_h = sum(log(1 - prior_bf.epsilon));
    for kk = f_h, lp_h = lp_h + log(prior_bf.epsilon(kk)) - log(1 - prior_bf.epsilon(kk)); end
    [Rw, ~] = chol(W_h); [Rn, ~] = chol(N_h);
    J_h = lp_h + 0.5 * (2*sum(log(diag(Rw))) - 2*sum(log(diag(Rn)))) - 0.5 * J_res_h;
    if J_h > best_J_exact, best_J_exact = J_h; best_mask_exact = mask_h; end
end
mask_got = false(1, p_bf); mask_got(sol_bf_test.fault_set) = true;
assert(isequal(mask_got, best_mask_exact) && abs(sol_bf_test.J_map - best_J_exact) < 1e-8, ...
    'BDIA does not match independent brute force ground truth.');
fprintf('  [7/7] Brute-force full ground-truth regression   : PASSED\n\n');

fprintf('=================================================================\n');
fprintf('                 ALL DEMOS AND CHECKS COMPLETED!                 \n');
fprintf('=================================================================\n');
