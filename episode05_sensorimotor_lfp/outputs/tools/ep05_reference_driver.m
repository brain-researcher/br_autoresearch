% EP05 direct execution of unchanged pinned publication and TrialData code.
% Only the selected exposed M1 window MAT is available to these entrypoints.
assert(~isempty(getenv('SLURM_JOB_ID')), 'Reference execution requires Slurm');
cfg = jsondecode(fileread(getenv('EP05_REFERENCE_CONFIG')));
addpath(genpath(cfg.trialdata_root));
addpath(fullfile(cfg.analysis_root, 'MainAnalysis'));
assert(startsWith(which('fCCA'), cfg.analysis_root));
assert(startsWith(which('vel_decoder'), cfg.analysis_root));
assert(startsWith(which('getModel'), cfg.trialdata_root));
assert(startsWith(which('smooth_data'), cfg.trialdata_root));

% Meaningful reference-function fixtures run before reading any real window.
fixture = reference_fixtures();
write_json(fullfile(cfg.output_dir, 'reference_fixtures.json'), fixture);
loaded = load(cfg.input_mat, 'trial_data');
trial_data = loaded.trial_data;
assert(all([trial_data.bin_size] == .03));
assert(all(arrayfun(@(t) size(t.M1_lfp,1) == 19 && size(t.vel,1) == 19, trial_data)));
assert(all([trial_data.idx_movement_on] == 5));

t = tic;
[cca_coef, ~, ~] = fCCA(trial_data, struct('array','M1','pca_dims',10, ...
    'surrogate_iter',0,'doPlot',false));
assert(all(isfinite(cca_coef(:))), 'Nonfinite reference CCA coefficient');
cca_seconds = toc(t);
save(fullfile(cfg.output_dir, 'reference_cca.mat'), 'cca_coef', 'cca_seconds', '-v7');

% Preserve every publication whole-trial permutation without editing its code.
% SVD PCA and the fixed decoder are deterministic between the randperm calls.
n = length(trial_data);
rng(cfg.seed, 'twister'); rng_before = rng;
permutations = zeros(500,n);
for k = 1:500, permutations(k,:) = randperm(n); end
rng_expected_after = rng;
rng(rng_before);
t = tic;
velocity = vel_decoder(trial_data, struct('array','M1','pca_dims',10, ...
    'bins_to_past',3,'folds',50,'eval',{{'r2'}},'doPlot',false,'fix_traintest',false));
velocity_seconds = toc(t);
rng_after = rng;
assert(isequal(rng_after, rng_expected_after), 'Unaccounted reference random draws');
assert(length(velocity) == 500, 'Reference did not return all fifty by ten fits');
scores = reshape([velocity.r2],2,[])';
assert(all(isfinite(scores(:))), 'Nonfinite reference correlation-squared score');
test_count = round(n*.1);
assert(all(arrayfun(@(v) isequal(size(v.actual_vel), [test_count*16,2]), velocity)));
save(fullfile(cfg.output_dir, 'reference_velocity.mat'), 'velocity','permutations', ...
    'rng_before','rng_after','test_count','velocity_seconds','-v7');
summary = struct('session_id',cfg.session_id,'matlab_version',version, ...
    'matlab_release',version('-release'),'toolboxes',ver, ...
    'selected_trials',n,'test_trials_per_split',test_count,'train_trials_per_split',n-test_count, ...
    'folds_per_representation',50,'representations',10,'pca_dims',10, ...
    'cca_seconds',cca_seconds,'velocity_seconds',velocity_seconds, ...
    'cca_size',size(cca_coef),'cca_channel_mean_per_band',mean(cca_coef,1), ...
    'velocity_mean_corr_squared_XY_per_representation',reshape(mean(reshape(scores,50,10,2),1),10,2), ...
    'fixtures_passed',true,'all_trial_pca_preserved',true, ...
    'polynomial_applied_to_last_coordinate_only',true, ...
    'whole_paper_reproduction_established',false);
write_json(fullfile(cfg.output_dir,'reference_result.json'),summary);

function checks = reference_fixtures()
    tol = 1e-10;
    impulse = zeros(19,1); impulse(1) = 1; impulse(8) = -.7;
    actual = smooth_data(impulse,.03,.05);
    expected = zeros(19,1);
    half_width = ceil(3*.05/.03);
    for t = 1:19
        idx = max(1,t-half_width):min(19,t+half_width);
        weight = exp(-.5*((idx-t)*.03/.05).^2);
        expected(t) = weight*impulse(idx)/sum(weight);
    end
    smoothing_error = max(abs(actual-expected));
    assert(smoothing_error < tol, 'Gaussian/truncated-mass reference mismatch');
    td = struct('bin_size',.03,'M1_spikes',(1:19)', 'vel',[(101:119)',(201:219)']);
    shifted = dupeAndShift(td,'M1_spikes',[-1,-2,-3]);
    assert(isequal(shifted.M1_spikes_shift,[(3:18)',(2:17)',(1:16)']));
    assert(isequal(shifted.vel,td.vel(4:end,:)));
    x = linspace(-1,1,31)';
    td = struct('bin_size',.03,'M1_spikes',x,'vel',[x+.4*x.^3,2*x-.7*x.^3]);
    fitted = getModel(td,struct('model_type','linmodel','in_signals','M1_spikes', ...
        'out_signals','vel','train_idx',1,'polynomial',3));
    X = [ones(size(x)),x];
    linear = X*(X\td.vel);
    expected_y = polyval(polyfit(linear(:,2),td.vel(:,2),3),linear(:,2));
    assert(max(abs(fitted.linmodel_default(:,1)-linear(:,1))) < tol);
    assert(max(abs(fitted.linmodel_default(:,2)-expected_y)) < tol);
    assert(max(abs(linear(:,1)-td.vel(:,1))) > .01, 'Fixture must distinguish omitted X cubic');
    r = compute_r2(x,2*x+9);
    assert(abs(r-1) < tol, 'Reference score is correlation squared');
    checks = struct('passed',true,'absolute_numerical_tolerance',tol, ...
        'smoothing_boundary_max_error',smoothing_error, ...
        'three_past_lags_and_target_trimming',true,'Y_only_polynomial_quirk',true, ...
        'correlation_squared_not_SSE_R_squared',true, 'scientific_outcomes_read',false);
end

function write_json(path, value)
    assert(~isfile(path), 'Preserve previous attempt outputs');
    stream = fopen(path,'w'); assert(stream ~= -1);
    closer = onCleanup(@() fclose(stream));
    fprintf(stream,'%s\n',jsonencode(value));
end
