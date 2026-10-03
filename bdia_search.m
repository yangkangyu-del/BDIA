function search_res = bdia_search(model, order, root_state, relaxed_root, options)
%BDIA_SEARCH Depth-First Branch and Bound search for global MAP hypothesis.
%
%   Traverses the binary hypothesis tree using monotonic upper bound pruning:
%   - Faulty-child branch: explores suspected faults first, pruned if the
%     tight subtree upper bound cannot exceed current J_max.
%   - Healthy-child branch: uses a fast two-stage screening (cheap loose
%     upper bound first, tight bound only if loose bound survives).

    p = model.p;
    A = model.A;
    diag_fast = model.diag_fast_mode;
    tol = options.tol;

    fault_data = order.fault_data;
    cached_nodes = order.cached_nodes;
    sort_idx = order.sort_idx;

    % Initialize global best from root state
    J_max = root_state.J;
    x_max = root_state.x;
    H_max_sorted = false(1, p);
    N_computed = 1 + p; % Root node + p single-fault heuristic evaluations

    % Check if any pre-evaluated single-fault hypothesis beats the root
    for k = 1:p
        if cached_nodes(k).ok && cached_nodes(k).J > J_max
            J_max = cached_nodes(k).J;
            x_max = cached_nodes(k).x;
            mask_k = false(1, p);
            mask_k(k) = true;
            H_max_sorted = mask_k;
        end
    end

    % Start recursive DFS search
    mask0 = false(1, p);
    dfs(0, root_state, relaxed_root, mask0);

    % Map optimal hypothesis back to original observation ordering
    H_max_original = false(1, p);
    if any(H_max_sorted)
        H_max_original(sort_idx(H_max_sorted)) = true;
    end

    search_res = struct();
    search_res.J_max = J_max;
    search_res.x_max = x_max;
    search_res.H_max = H_max_original;
    search_res.H_max_sorted = H_max_sorted;
    search_res.N_computed = N_computed;

    % --- Nested DFS Tree Traversal ---
    function dfs(level, curr, rel, mask)
        if level >= p
            return;
        end

        k = level + 1;
        fk = fault_data(k);

        % -------------------------------------------------------------
        % 1. Faulty Child Branch (activate fault k)
        % -------------------------------------------------------------
        if ~any(mask) && cached_nodes(k).ok
            % Reuse single-fault state from heuristic sorting
            cached = cached_nodes(k);
            ub_F = cached.J_prior + 0.5 * cached.J_geo - 0.5 * rel.J_res;
            ub_F = min(ub_F, cached.J_prior + 0.5 * cached.J_geo);

            if ub_F >= J_max - tol
                mask_new = mask;
                mask_new(k) = true;

                if cached.J > J_max
                    J_max = cached.J;
                    x_max = cached.x;
                    H_max_sorted = mask_new;
                end

                next_curr = curr;
                next_curr.Sigma = cached.Sigma;
                next_curr.W = cached.W;
                next_curr.N_inv = cached.N_inv;
                next_curr.y = cached.y;
                next_curr.J_prior = cached.J_prior;
                next_curr.J_geo = cached.J_geo;
                next_curr.b = cached.b;
                next_curr.c = cached.c;
                dfs(level + 1, next_curr, rel, mask_new);
            end
        else
            prep = prepare_faulty_step(curr.W, curr.Sigma, curr.N_inv, curr.J_prior, curr.J_geo, fk, A, diag_fast);
            if prep.ok
                % Monotonic tight upper bound for faulty subtree
                ub_F = prep.J_prior_new + 0.5 * prep.J_geo_new - 0.5 * rel.J_res;
                ub_F = min(ub_F, prep.J_prior_new + 0.5 * prep.J_geo_new);

                if ub_F >= J_max - tol
                    [W_new, Sig_new, N_new_inv, y_new, b_new, c_new, ok_full] = realize_faulty_step( ...
                        curr.W, curr.Sigma, curr.N_inv, curr.y, curr.b, curr.c, fk, prep, A, diag_fast);

                    if ok_full
                        [x_new, J_res_new] = solve_state(A, W_new, N_new_inv, y_new, b_new, c_new, diag_fast);
                        J_new = prep.J_prior_new + 0.5 * prep.J_geo_new - 0.5 * J_res_new;
                        N_computed = N_computed + 1;

                        mask_new = mask;
                        mask_new(k) = true;

                        if J_new > J_max
                            J_max = J_new;
                            x_max = x_new;
                            H_max_sorted = mask_new;
                        end

                        next_curr = curr;
                        next_curr.Sigma = Sig_new;
                        next_curr.W = W_new;
                        next_curr.N_inv = N_new_inv;
                        next_curr.y = y_new;
                        next_curr.J_prior = prep.J_prior_new;
                        next_curr.J_geo = prep.J_geo_new;
                        next_curr.b = b_new;
                        next_curr.c = c_new;
                        dfs(level + 1, next_curr, rel, mask_new);
                    end
                end
            end
        end

        % -------------------------------------------------------------
        % 2. Healthy Child Branch (fix fault k inactive)
        % -------------------------------------------------------------
        % Stage 1: Fast loose upper bound check (avoids matrix updates if failed)
        ub_H_loose = curr.J_prior + 0.5 * curr.J_geo;
        if ub_H_loose < J_max - tol
            return;
        end

        % Stage 2: Update relaxed continuation (remove fault k from relaxed model)
        [W_rel_new, Sig_rel_new, N_rel_new_inv, y_rel_new, J_res_rel_new, b_rel_new, c_rel_new, prior_pen_new, ok_rel] = ...
            realize_healthy_relaxed_step(rel.W, rel.Sigma, rel.N_inv, rel.y, rel.b, rel.c, rel.prior_pen, fk, A, diag_fast);

        if ~ok_rel
            next_rel = rel;
        else
            next_rel = rel;
            next_rel.Sigma = Sig_rel_new;
            next_rel.W = W_rel_new;
            next_rel.N_inv = N_rel_new_inv;
            next_rel.y = y_rel_new;
            next_rel.J_res = J_res_rel_new;
            next_rel.b = b_rel_new;
            next_rel.c = c_rel_new;
            next_rel.prior_pen = prior_pen_new;
        end

        % Tight upper bound for healthy subtree
        ub_H = curr.J_prior + 0.5 * curr.J_geo - 0.5 * next_rel.J_res;
        ub_H = min(ub_H, ub_H_loose);

        if ub_H >= J_max - tol
            dfs(level + 1, curr, next_rel, mask);
        end
    end
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
        % Algebraically: 1 - u_scalar^2 * Sigma(idx) == sig_inv / S_fault (avoids cancellation)
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
    prep.diag_fast = false;
    prep.U_k = U_k;
    prep.R_core = R_core;
    prep.proj = proj;
end

function [W_new, Sigma_new, N_inv_new, y_new, b_new, c_new, ok] = ...
        realize_faulty_step(W, Sigma, N_inv, y, b, c, fk, prep, A, diag_fast)

    ok = false;
    W_new = []; Sigma_new = []; N_inv_new = [];
    y_new = y - fk.F * fk.Miu;
    b_new = []; c_new = [];

    if diag_fast && prep.diag_fast
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
            b_new = b + prep.proj * sqrt(prep.sig_inv + prep.w_curr) * ...
                    (W_new(idx) * new_y / prep.w_curr - old_y);
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

function [W_rel_new, Sigma_rel_new, N_inv_rel_new, y_rel_new, J_res_rel_new, b_rel_new, c_rel_new, prior_pen_new, ok] = ...
        realize_healthy_relaxed_step(W_rel, Sigma_rel, N_inv_rel, y_rel, b_rel, c_rel, prior_pen_curr, fk, A, diag_fast)

    ok = false;
    W_rel_new = []; Sigma_rel_new = []; N_inv_rel_new = [];
    y_rel_new = y_rel + fk.F * fk.Miu;
    J_res_rel_new = []; b_rel_new = []; c_rel_new = [];
    prior_pen_new = max(prior_pen_curr - fk.prior_pen, 0);

    if diag_fast && fk.BlockDim == 1 && ~isempty(fk.MeasIndex)
        idx = fk.MeasIndex;
        sig_inv = fk.SigInv(1);
        sig_val = fk.Sig(1);
        w_rel = W_rel(idx);
        S_remove = sig_inv - w_rel;
        if S_remove <= 0, return; end

        u_scalar = w_rel / sqrt(S_remove);
        proj = fk.ARow * u_scalar;
        state_core = 1 + proj' * N_inv_rel * proj;
        if state_core <= 0, return; end

        W_rel_new = W_rel;
        W_rel_new(idx) = w_rel + (w_rel^2) / S_remove;

        Sigma_rel_new = Sigma_rel;
        Sigma_rel_new(idx) = Sigma_rel_new(idx) - sig_val;

        tmp = (N_inv_rel * proj) / sqrt(state_core);
        N_inv_rel_new = 0.5 * (N_inv_rel - tmp * tmp' + (N_inv_rel - tmp * tmp')');

        if ~isempty(b_rel)
            old_y = y_rel(idx);
            new_y = y_rel_new(idx);
            b_rel_new = b_rel + fk.ARow * (W_rel_new(idx) * new_y - w_rel * old_y);
            c_rel_new = c_rel + W_rel_new(idx) * (new_y^2) - w_rel * (old_y^2);
        end

        [~, J_res_raw] = solve_state(A, W_rel_new, N_inv_rel_new, y_rel_new, b_rel_new, c_rel_new, true);
        J_res_rel_new = J_res_raw - prior_pen_new;
        ok = true;
        return;
    end

    WFk = W_rel * fk.F;
    S_remove = 0.5 * (fk.SigInv - fk.F' * WFk + (fk.SigInv - fk.F' * WFk)');
    [R_remove, p_flag] = chol(S_remove);
    if p_flag ~= 0, return; end

    U_k = WFk / R_remove;
    proj = A' * U_k;
    state_core = 0.5 * (eye(fk.BlockDim) + proj' * N_inv_rel * proj + ...
                       (eye(fk.BlockDim) + proj' * N_inv_rel * proj)');
    [R_core, p_flag] = chol(state_core);
    if p_flag ~= 0, return; end

    W_rel_new = 0.5 * (W_rel + U_k * U_k' + (W_rel + U_k * U_k')');
    Sigma_rel_new = 0.5 * (Sigma_rel - fk.F * fk.Sig * fk.F' + ...
                          (Sigma_rel - fk.F * fk.Sig * fk.F')');
    tmp = (N_inv_rel * proj) / R_core;
    N_inv_rel_new = 0.5 * (N_inv_rel - tmp * tmp' + (N_inv_rel - tmp * tmp')');
    [~, J_res_raw] = solve_state(A, W_rel_new, N_inv_rel_new, y_rel_new, [], [], false);
    J_res_rel_new = J_res_raw - prior_pen_new;
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
