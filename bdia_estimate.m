function [sol, info] = bdia_estimate(model, prior, order, search_res, options)
%BDIA_ESTIMATE State estimation, covariance extraction, and result packaging.
%
%   Computes adapted state vector x_map, posterior variance-covariance matrix
%   Dxx_map = N_map^-1, standard deviations, post-fit residuals, and identified
%   fault sets from the optimal hypothesis returned by bdia_search.

    A = model.A;
    y = model.y;
    D_mat = model.D_mat;
    D_diag = model.D_diag;
    diag_fast = model.diag_fast_mode;
    p = model.p;

    H_max = search_res.H_max;
    fault_set = find(H_max);
    fault_detected = ~isempty(fault_set);
    x_map = search_res.x_max;

    % Recompute exact posterior state, covariance, and residuals under H_MAP via whitened QR
    if ~fault_detected
        y_corr = y;
        if diag_fast
            W_map = 1 ./ D_diag;
            A_white = A .* sqrt(W_map);
            y_white = y .* sqrt(W_map);
        else
            [R_D, p_D] = chol(D_mat);
            if p_D ~= 0
                error('bdia:InvalidD', 'Observation covariance D must be symmetric positive definite.');
            end
            A_white = R_D' \ A;
            y_white = R_D' \ y;
        end
    else
        % Map active fault blocks to model columns
        col_mask = false(1, model.total_dim);
        col_offset = 0;
        for k = 1:p
            dk = prior.dim(k);
            if H_max(k)
                col_mask(col_offset + (1:dk)) = true;
            end
            col_offset = col_offset + dk;
        end

        F_act = prior.F(:, col_mask);
        Sig_act = prior.Sigma_nabla(col_mask, col_mask);
        mu_act = prior.mu_nabla(col_mask);
        y_corr = y - F_act * mu_act;

        if diag_fast
            Sigma_map = D_diag;
            Sigma_map(fault_set) = Sigma_map(fault_set) + diag(Sig_act);
            W_map = 1 ./ Sigma_map;
            A_white = A .* sqrt(W_map);
            y_white = y_corr .* sqrt(W_map);
        else
            Sigma_map = 0.5 * (D_mat + F_act * Sig_act * F_act' + (D_mat + F_act * Sig_act * F_act')');
            [R_sig, p_sig] = chol(Sigma_map);
            if p_sig ~= 0
                error('bdia:InvalidCovariance', 'Posterior measurement covariance is not positive definite.');
            end
            A_white = R_sig' \ A;
            y_white = R_sig' \ y_corr;
        end
    end

    [Q_map, R_map] = qr(A_white, 0);
    if rcond(R_map) < eps || any(abs(diag(R_map)) <= eps * max(abs(diag(R_map))))
        error('bdia:RankDeficientSystem', 'System is numerically rank-deficient under optimal hypothesis.');
    end
    R_map_inv = R_map \ eye(model.n);
    N_map_inv = 0.5 * (R_map_inv * R_map_inv' + (R_map_inv * R_map_inv')');
    x_map = R_map \ (Q_map' * y_white);
    v_map = y_corr - A * x_map;

    sol = struct();
    sol.x = x_map;
    sol.x_map = x_map;
    sol.Dxx = N_map_inv;
    sol.Dxx_map = N_map_inv;
    sol.std_map = sqrt(max(diag(N_map_inv), 0));
    sol.fault_detected = fault_detected;
    sol.fault_set = fault_set;
    sol.H_map = H_max;
    sol.residuals = v_map;
    sol.J_map = search_res.J_max;

    info = struct();
    info.N_computed = search_res.N_computed;
    info.sort_order = order.sort_idx;
    info.diag_fast_mode = diag_fast;
    info.options = options;
end
