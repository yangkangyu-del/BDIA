function [order, root_state, relaxed_root] = bdia_order(model, prior, options)
%BDIA_ORDER Heuristic fault ordering and single-fault caching for BDIA.
%
%   Evaluates the root hypothesis (fault-free) and all p single-fault hypotheses.
%   Reorders faults descending by single-fault posterior scores to rapidly
%   establish a high benchmark J_max for effective subtree pruning.
%   Caches single-fault evaluations so they are reused during the tree search.

    m = model.m;
    p = model.p;
    y = model.y;
    A = model.A;
    D_mat = model.D_mat;
    D_diag = model.D_diag;
    diag_fast = model.diag_fast_mode;

    %% 1. Initialize fault-free root node
    if diag_fast
        Sigma0 = D_diag;
        W0 = 1 ./ Sigma0;
        A_white0 = A .* sqrt(W0);
        y_white0 = y .* sqrt(W0);
        b0 = A' * (W0 .* y);
        c0 = y' * (W0 .* y);
    else
        Sigma0 = D_mat;
        [R_D, p_D] = chol(Sigma0);
        if p_D ~= 0
            error('bdia:InvalidD', 'Observation covariance D must be symmetric positive definite.');
        end
        W0 = safe_inv_spd(Sigma0);
        A_white0 = R_D' \ A;
        y_white0 = R_D' \ y;
        b0 = [];
        c0 = [];
    end

    [Q0, R_A0] = qr(A_white0, 0);
    if rcond(R_A0) < eps || any(abs(diag(R_A0)) <= eps * max(abs(diag(R_A0))))
        error('bdia:RankDeficientA', 'Design matrix A is rank deficient under D.');
    end
    R_A0_inv = R_A0 \ eye(model.n);
    N0_inv = 0.5 * (R_A0_inv * R_A0_inv' + (R_A0_inv * R_A0_inv')');
    x0 = R_A0 \ (Q0' * y_white0);
    res0 = y_white0 - A_white0 * x0;
    J_res0 = sum(res0.^2);

    if diag_fast
        J_geo0 = sum(log(W0)) - 2 * sum(log(abs(diag(R_A0))));
    else
        J_geo0 = -2 * sum(log(diag(R_D))) - 2 * sum(log(abs(diag(R_A0))));
    end
    J_prior0 = sum(log(1 - prior.epsilon));
    J0 = J_prior0 + 0.5 * J_geo0 - 0.5 * J_res0;

    root_state = struct('Sigma', Sigma0, 'W', W0, 'N_inv', N0_inv, 'y', y, ...
                        'J_prior', J_prior0, 'J_geo', J_geo0, 'b', b0, 'c', c0, ...
                        'x', x0, 'J', J0);

    %% 2. Initialize fully relaxed root state (all p faults active)
    if diag_fast
        Sigma_rel0 = D_diag + diag(prior.Sigma_nabla);
        W_rel0 = 1 ./ Sigma_rel0;
        y_rel0 = y - prior.F * prior.mu_nabla;
        A_white_rel = A .* sqrt(W_rel0);
        y_white_rel = y_rel0 .* sqrt(W_rel0);
        b_rel0 = A' * (W_rel0 .* y_rel0);
        c_rel0 = y_rel0' * (W_rel0 .* y_rel0);
    else
        Sigma_rel0 = 0.5 * (D_mat + prior.F * prior.Sigma_nabla * prior.F' + ...
                            (D_mat + prior.F * prior.Sigma_nabla * prior.F')');
        [R_Srel, p_Srel] = chol(Sigma_rel0);
        if p_Srel ~= 0
            error('bdia:InvalidPriorCovariance', 'Relaxed covariance is not positive definite.');
        end
        W_rel0 = safe_inv_spd(Sigma_rel0);
        y_rel0 = y - prior.F * prior.mu_nabla;
        A_white_rel = R_Srel' \ A;
        y_white_rel = R_Srel' \ y_rel0;
        b_rel0 = [];
        c_rel0 = [];
    end

    [Q_rel, R_A_rel] = qr(A_white_rel, 0);
    if rcond(R_A_rel) < eps || any(abs(diag(R_A_rel)) <= eps * max(abs(diag(R_A_rel))))
        error('bdia:RankDeficientA', 'Design matrix A is rank deficient under relaxed model.');
    end
    R_A_rel_inv = R_A_rel \ eye(model.n);
    N_rel0_inv = 0.5 * (R_A_rel_inv * R_A_rel_inv' + (R_A_rel_inv * R_A_rel_inv')');
    x_rel = R_A_rel \ (Q_rel' * y_white_rel);
    res_rel = y_white_rel - A_white_rel * x_rel;
    J_res_rel0_raw = sum(res_rel.^2);

    %% 3. Build per-fault model structures
    fault_data = repmat(struct('F', [], 'Sig', [], 'SigInv', [], 'logdet_Sig', 0, 'Miu', [], ...
                               'DeltaLP', 0, 'BlockDim', 0, 'OriginalIndex', 0, ...
                               'MeasIndex', [], 'ARow', [], 'prior_pen', 0), 1, p);

    col_idx = 0;
    total_prior_pen = 0;
    for k = 1:p
        dk = prior.dim(k);
        cols = col_idx + (1:dk);
        col_idx = col_idx + dk;

        F_k = prior.F(:, cols);
        Sig_k = prior.Sigma_nabla(cols, cols);
        Miu_k = prior.mu_nabla(cols);

        fault_data(k).F = F_k;
        fault_data(k).Sig = Sig_k;
        fault_data(k).SigInv = safe_inv_spd(Sig_k);
        fault_data(k).logdet_Sig = safe_logdet(Sig_k);
        fault_data(k).Miu = Miu_k;
        fault_data(k).DeltaLP = log(prior.epsilon(k)) - log(1 - prior.epsilon(k));
        fault_data(k).BlockDim = dk;
        fault_data(k).OriginalIndex = k;
        fault_data(k).prior_pen = Miu_k' * (fault_data(k).SigInv * Miu_k);
        total_prior_pen = total_prior_pen + fault_data(k).prior_pen;

        if diag_fast && dk == 1
            nz = find(abs(F_k) > 0);
            if numel(nz) == 1
                fault_data(k).MeasIndex = nz;
                fault_data(k).ARow = A(nz, :).';
            end
        end
    end

    J_res_rel0 = J_res_rel0_raw - total_prior_pen;

    relaxed_root = struct('Sigma', Sigma_rel0, 'W', W_rel0, 'N_inv', N_rel0_inv, ...
                          'y', y_rel0, 'J_res', J_res_rel0, 'b', b_rel0, 'c', c_rel0, ...
                          'prior_pen', total_prior_pen);

    %% 4. Single-fault evaluations & heuristic sorting
    J_single = -inf(1, p);
    cached_nodes = repmat(struct('ok', false, 'Sigma', [], 'W', [], 'N_inv', [], ...
                                 'y', [], 'b', [], 'c', [], 'J_prior', [], ...
                                 'J_geo', [], 'x', [], 'J', []), 1, p);

    for k = 1:p
        fk = fault_data(k);
        prep = prepare_faulty_step(W0, Sigma0, N0_inv, J_prior0, J_geo0, fk, A, diag_fast);
        if ~prep.ok, continue; end

        [W_k, Sig_k, N_k_inv, y_k, b_k, c_k, ok] = realize_faulty_step( ...
            W0, Sigma0, N0_inv, y, b0, c0, fk, prep, A, diag_fast);
        if ~ok, continue; end

        [x_k, J_res_k] = solve_state(A, W_k, N_k_inv, y_k, b_k, c_k, diag_fast);
        J_k = prep.J_prior_new + 0.5 * prep.J_geo_new - 0.5 * J_res_k;
        J_single(k) = J_k;

        cached_nodes(k).ok = true;
        cached_nodes(k).Sigma = Sig_k;
        cached_nodes(k).W = W_k;
        cached_nodes(k).N_inv = N_k_inv;
        cached_nodes(k).y = y_k;
        cached_nodes(k).b = b_k;
        cached_nodes(k).c = c_k;
        cached_nodes(k).J_prior = prep.J_prior_new;
        cached_nodes(k).J_geo = prep.J_geo_new;
        cached_nodes(k).x = x_k;
        cached_nodes(k).J = J_k;
    end

    if strcmpi(options.sort_mode, 'single')
        [~, sort_idx] = sort(J_single, 'descend');
    else
        sort_idx = 1:p;
    end

    order = struct();
    order.sort_idx = sort_idx;
    order.fault_data = fault_data(sort_idx);
    order.cached_nodes = cached_nodes(sort_idx);
    order.J_single = J_single(sort_idx);
end

% --- Local Algorithmic Helpers ---
function prep = prepare_faulty_step(W, Sigma, N_inv, J_prior, J_geo, fk, A, diag_fast)
    prep = struct('ok', false, 'J_prior_new', J_prior + fk.DeltaLP, 'J_geo_new', [], ...
                  'diag_fast', false, 'meas_idx', [], 'w_curr', [], 'u_scalar', [], ...
                  'proj', [], 'state_core', [], 'U_k', [], 'R_core', []);

    if diag_fast && fk.BlockDim == 1 && ~isempty(fk.MeasIndex)
        idx = fk.MeasIndex;
        sig_inv = fk.SigInv(1);
        sig_val = fk.Sig(1);
        w_curr = W(idx);
        S_fault = sig_inv + w_curr;
        if S_fault <= 0, return; end

        u_scalar = w_curr / sqrt(S_fault);
        proj = fk.ARow * u_scalar;
        % Algebraically: 1 - u_scalar^2 * Sigma(idx) == sig_inv / S_fault
        meas_core = sig_inv / S_fault;
        state_core = 1 - proj' * N_inv * proj;
        if meas_core <= 0 || state_core <= 0, return; end

        prep.ok = true;
        prep.J_geo_new = J_geo + log(meas_core) - log(state_core);
        prep.diag_fast = true;
        prep.meas_idx = idx;
        prep.w_curr = w_curr;
        prep.sig_inv = sig_inv;
        prep.sig_val = sig_val;
        prep.u_scalar = u_scalar;
        prep.proj = proj;
        prep.state_core = state_core;
        return;
    end

    WFk = W * fk.F;
    S_fault = 0.5 * (fk.SigInv + fk.F' * WFk + (fk.SigInv + fk.F' * WFk)');
    [R_fault, p_flag] = chol(S_fault);
    if p_flag ~= 0, return; end

    U_k = WFk / R_fault;
    proj = A' * U_k;
    state_core = 0.5 * (eye(fk.BlockDim) - proj' * N_inv * proj + ...
                       (eye(fk.BlockDim) - proj' * N_inv * proj)');
    [R_core, p_state] = chol(state_core);
    if p_state ~= 0, return; end

    logdet_meas = -fk.logdet_Sig - 2 * sum(log(diag(R_fault)));
    logdet_state = 2 * sum(log(diag(R_core)));

    prep.ok = true;
    prep.J_geo_new = J_geo + logdet_meas - logdet_state;
    prep.U_k = U_k;
    prep.proj = proj;
    prep.R_core = R_core;
end

function [W_new, Sigma_new, N_inv_new, y_new, b_new, c_new, ok] = realize_faulty_step( ...
        W, Sigma, N_inv, y, b, c, fk, prep, A, diag_fast)

    ok = false;
    W_new = []; Sigma_new = []; N_inv_new = []; y_new = y - fk.F * fk.Miu;
    b_new = []; c_new = [];

    if ~prep.ok, return; end

    if prep.diag_fast
        idx = prep.meas_idx;
        W_new = W;
        W_new(idx) = (prep.w_curr * prep.sig_inv) / (prep.sig_inv + prep.w_curr);

        Sigma_new = Sigma;
        Sigma_new(idx) = Sigma_new(idx) + prep.sig_val;

        tmp = (N_inv * prep.proj) / sqrt(prep.state_core);
        N_inv_new = 0.5 * (N_inv + tmp * tmp' + (N_inv + tmp * tmp')');

        if ~isempty(b)
            old_y = y(idx);
            new_y = y_new(idx);
            b_new = b + fk.ARow * (W_new(idx) * new_y - prep.w_curr * old_y);
            c_new = c + W_new(idx) * (new_y^2) - prep.w_curr * (old_y^2);
        end
        ok = true;
        return;
    end

    W_new = 0.5 * (W - prep.U_k * prep.U_k' + (W - prep.U_k * prep.U_k')');
    Sigma_new = 0.5 * (Sigma + fk.F * fk.Sig * fk.F' + (Sigma + fk.F * fk.Sig * fk.F')');
    tmp = (N_inv * prep.proj) / prep.R_core;
    N_inv_new = 0.5 * (N_inv + tmp * tmp' + (N_inv + tmp * tmp')');
    ok = true;
end

function [x, J_res] = solve_state(A, W, N_inv, y, b, c, diag_fast)
    if diag_fast && ~isempty(b)
        x = N_inv * b;
        quad_diff = c - b' * x;
        if quad_diff > 1e-7 * c
            J_res = quad_diff;
            return;
        end
        % Fallback to direct residual calculation if precision is lost
        res = y - A * x;
        J_res = sum(W(:) .* (res.^2));
        return;
    end

    if isvector(W)
        rhs = A' * (W(:) .* y);
        x = N_inv * rhs;
        res = y - A * x;
        J_res = sum(W(:) .* (res.^2));
    else
        rhs = A' * (W * y);
        x = N_inv * rhs;
        res = y - A * x;
        J_res = res' * (W * res);
    end
end

function val = safe_logdet(M)
    if isvector(M)
        val = sum(log(M));
        return;
    end

    M = 0.5 * (M + M');
    [R, p_flag] = chol(M);
    if p_flag == 0
        val = 2 * sum(log(diag(R)));
    else
        ev = eig(M);
        val = sum(log(ev(ev > 0)));
    end
end

function Minv = safe_inv_spd(M)
    if isvector(M)
        Minv = 1 ./ M;
        return;
    end

    M = 0.5 * (M + M');
    [R, p_flag] = chol(M);
    if p_flag ~= 0
        error('bdia:RankDeficientSystem', 'Matrix must be symmetric positive definite.');
    end
    R_inv = R \ eye(size(M, 1));
    Minv = 0.5 * (R_inv * R_inv' + (R_inv * R_inv')');
end
