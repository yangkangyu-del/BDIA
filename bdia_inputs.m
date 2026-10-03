function [model, prior, options] = bdia_inputs(y, A, D, prior, options)
%BDIA_INPUTS Preprocess and validate input arguments for BDIA.
%
%   Internal helper function for BDIA.

    % 1. Validate observations y and design matrix A
    validateattributes(y, {'numeric'}, {'vector', 'real', 'finite', 'nonempty'}, 'bdia', 'y');
    y = y(:);
    m = numel(y);

    validateattributes(A, {'numeric'}, {'2d', 'real', 'finite', 'nrows', m}, 'bdia', 'A');
    n = size(A, 2);
    if n > m
        error('bdia:UnderdeterminedSystem', 'Number of unknowns n (%d) exceeds observations m (%d).', n, m);
    end
    if rank(A) < n
        error('bdia:RankDeficientA', 'Design matrix A must have full column rank n = %d.', n);
    end

    % 2. Validate observation covariance D
    validateattributes(D, {'numeric'}, {'2d', 'real', 'finite', 'nonempty'}, 'bdia', 'D');
    if isvector(D)
        if numel(D) ~= m || any(D <= 0)
            error('bdia:InvalidD', 'Observation variance vector D must have m = %d positive finite elements.', m);
        end
        D_diag = D(:);
        D_mat = diag(D_diag);
    elseif ismatrix(D)
        if size(D, 1) ~= m || size(D, 2) ~= m
            error('bdia:InvalidD', 'Observation covariance matrix D must be m-by-m (%d x %d).', m, m);
        end
        if norm(D - D', 'fro') / max(norm(D, 'fro'), 1) > 1e-10
            error('bdia:InvalidD', 'Observation covariance matrix D must be symmetric.');
        end
        D_mat = 0.5 * (D + D');
        [~, p_chol] = chol(D_mat);
        if p_chol ~= 0
            error('bdia:InvalidD', 'Observation covariance matrix D must be symmetric positive definite.');
        end
        if rcond(D_mat) < 1e-12
            error('bdia:IllConditionedD', 'Observation covariance matrix D is ill-conditioned (rcond < 1e-12).');
        end
        D_diag = diag(D_mat);
    else
        error('bdia:InvalidD', 'Observation covariance D must be a vector or matrix.');
    end

    % 3. Check numerical conditioning of design matrix under D (LAMBDA-style fail-fast)
    if isdiag(D_mat)
        N0_check = A' * (A ./ D_diag);
    else
        N0_check = A' * (D_mat \ A);
    end
    rcond_N0 = rcond(N0_check);
    if rcond_N0 < 1e-12
        error('bdia:IllConditionedSystem', ...
            'Design matrix A is ill-conditioned under observation noise D (rcond = %.2e < 1e-12).', rcond_N0);
    end

    % 4. Validate or populate prior struct
    if nargin < 4 || isempty(prior)
        error('bdia:MissingPrior', ...
            ['BDIA requires prior fault information.\n' ...
             '  * To run nominal Least Squares (zero fault probability), pass: prior = 0\n' ...
             '  * To run Bayesian outlier detection, pass a prior probability (e.g., prior = 0.01) or a prior struct.']);
    end

    if isnumeric(prior)
        validateattributes(prior, {'numeric'}, {'real', 'finite'}, 'bdia', 'prior');
        if isempty(prior)
            error('bdia:InvalidPrior', 'prior cannot be empty.');
        elseif all(prior(:) == 0)
            prior = struct('epsilon', zeros(1, m), 'active', false);
        else
            prior = struct('epsilon', prior, 'active', true);
        end
    elseif isstruct(prior)
        if isfield(prior, 'epsilon') && isempty(prior.epsilon)
            error('bdia:InvalidPrior', 'prior.epsilon cannot be empty.');
        end
        if isfield(prior, 'active') && ~prior.active
            % Explicitly marked inactive
        elseif isfield(prior, 'epsilon') && ~isempty(prior.epsilon) && all(prior.epsilon(:) == 0)
            prior.active = false;
        else
            prior.active = true;
        end
    else
        error('bdia:InvalidPrior', 'prior must be a struct, a numeric probability scalar/vector, or 0.');
    end

    % 4. Solver options defaults
    if nargin < 5 || ~isstruct(options)
        options = struct();
    end
    if isfield(options, 'q_max')
        error('bdia:DeprecatedOption', ...
            'The ''q_max'' option has been removed. BDIA explores the full combinatorial fault space.');
    end
    if ~isfield(options, 'sort_mode') || isempty(options.sort_mode)
        options.sort_mode = 'single';
    end
    if ~ischar(options.sort_mode) || ~ismember(lower(options.sort_mode), {'single', 'original'})
        error('bdia:InvalidOption', 'options.sort_mode must be ''single'' or ''original''.');
    end
    if ~isfield(options, 'tol') || isempty(options.tol)
        options.tol = 1e-10;
    end
    if ~isnumeric(options.tol) || ~isscalar(options.tol) || ~isfinite(options.tol) || options.tol < 0
        error('bdia:InvalidOption', 'options.tol must be a non-negative scalar.');
    end

    % If prior is not active, return nominal Least Squares model immediately
    if ~prior.active
        model = struct();
        model.y = y;
        model.A = A;
        model.D_mat = D_mat;
        model.D_diag = D_diag;
        model.m = m;
        model.n = n;
        model.p = 0;
        model.total_dim = 0;
        model.diag_fast_mode = isdiag(D_mat);
        return;
    end

    if ~isfield(prior, 'dim') || isempty(prior.dim)
        if isfield(prior, 'F') && ~isempty(prior.F)
            prior.dim = ones(1, size(prior.F, 2));
        else
            prior.dim = ones(1, m);
        end
    end
    prior.dim = prior.dim(:).';
    if any(prior.dim < 1) || any(prior.dim ~= round(prior.dim)) || any(~isfinite(prior.dim))
        error('bdia:InvalidPriorDim', 'prior.dim must contain positive integer block dimensions.');
    end
    p = numel(prior.dim);
    total_dim = sum(prior.dim);

    if ~isfield(prior, 'F') || isempty(prior.F)
        if total_dim == m
            prior.F = eye(m);
        else
            error('bdia:InvalidF', 'Prior fault design matrix F must be specified when sum(dim) ~= m.');
        end
    end
    validateattributes(prior.F, {'numeric'}, {'2d', 'real', 'finite', 'nrows', m, 'ncols', total_dim}, 'bdia', 'prior.F');

    if ~isfield(prior, 'epsilon') || isempty(prior.epsilon)
        prior.epsilon = 0.01 * ones(1, p);
    elseif isscalar(prior.epsilon)
        prior.epsilon = repmat(prior.epsilon, 1, p);
    end
    prior.epsilon = prior.epsilon(:).';
    validateattributes(prior.epsilon, {'numeric'}, {'vector', 'real', 'finite'}, 'bdia', 'prior.epsilon');
    if numel(prior.epsilon) ~= p || any(prior.epsilon <= 0) || any(prior.epsilon >= 0.5)
        error('bdia:InvalidEpsilon', 'Fault probabilities epsilon must be in (0, 0.5) with length p = %d.', p);
    end

    if ~isfield(prior, 'mu_nabla') || isempty(prior.mu_nabla)
        prior.mu_nabla = zeros(total_dim, 1);
    elseif isscalar(prior.mu_nabla) && total_dim > 1
        prior.mu_nabla = repmat(prior.mu_nabla, total_dim, 1);
    end
    prior.mu_nabla = prior.mu_nabla(:);
    validateattributes(prior.mu_nabla, {'numeric'}, {'vector', 'real', 'finite', 'numel', total_dim}, 'bdia', 'prior.mu_nabla');

    if ~isfield(prior, 'Sigma_nabla') || isempty(prior.Sigma_nabla)
        prior.Sigma_nabla = (100^2) * eye(total_dim);
    elseif isscalar(prior.Sigma_nabla)
        prior.Sigma_nabla = prior.Sigma_nabla * eye(total_dim);
    elseif isvector(prior.Sigma_nabla)
        prior.Sigma_nabla = diag(prior.Sigma_nabla(:));
    end
    validateattributes(prior.Sigma_nabla, {'numeric'}, {'2d', 'real', 'finite', 'nrows', total_dim, 'ncols', total_dim}, 'bdia', 'prior.Sigma_nabla');
    if norm(prior.Sigma_nabla - prior.Sigma_nabla', 'fro') / max(norm(prior.Sigma_nabla, 'fro'), 1) > 1e-10
        error('bdia:InvalidPriorCovariance', 'prior.Sigma_nabla must be symmetric.');
    end
    prior.Sigma_nabla = 0.5 * (prior.Sigma_nabla + prior.Sigma_nabla');
    [~, p_chol_sig] = chol(prior.Sigma_nabla);
    if p_chol_sig ~= 0
        error('bdia:InvalidPriorCovariance', 'prior.Sigma_nabla must be symmetric positive definite.');
    end
    if rcond(prior.Sigma_nabla) < 1e-12
        error('bdia:IllConditionedPriorCovariance', ...
            'Prior fault covariance Sigma_nabla is ill-conditioned (rcond < 1e-12).');
    end

    % Enforce cross-block independence (assert block-diagonal)
    assert_block_diagonal(prior.Sigma_nabla, prior.dim);

    % 5. Fast diagonal mode detection
    diag_fast_mode = isdiag(D_mat) && ...
                     all(prior.dim == 1) && ...
                     p == m && ...
                     isequal(prior.F, eye(m)) && ...
                     isdiag(prior.Sigma_nabla);

    % 6. Pack model
    model = struct();
    model.y = y;
    model.A = A;
    model.D_mat = D_mat;
    model.D_diag = D_diag;
    model.m = m;
    model.n = n;
    model.p = p;
    model.total_dim = total_dim;
    model.diag_fast_mode = diag_fast_mode;
end

function assert_block_diagonal(Sigma_nabla, dim)
    if isvector(Sigma_nabla) || numel(dim) <= 1
        return;
    end
    block_of = repelem(1:numel(dim), dim);
    [rows, cols] = find(abs(Sigma_nabla) > 0);
    if any(block_of(rows) ~= block_of(cols))
        error('bdia:InvalidPriorCovariance', ...
            ['prior.Sigma_nabla must be block-diagonal with respect to prior.dim: ' ...
             'the Bernoulli-Gaussian model treats fault blocks as independent, ' ...
             'so cross-block fault covariance is not supported.']);
    end
end
