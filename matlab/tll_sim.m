clearvars
close all
clc
addpath lib/
% Time-locked loop PLL Implementation
%
% Implements a DSP-based Phase-Locked Loop featured with a PI controller.
%
% Suggested experiments:
%   1) Set Ki = 0 and configure the frequency offset to zero (y0_err_ppb =
%   0). See that the PLL can correct the constant phase error.
%
%   2) Set Ki = 0 and configure a large enough frequency offset. See that
%   the time error converges to a constant value y (in ppb) * N * T, while
%   the filter output converges to this number divided by Kp.
%
%   3) With the same configuration, increase Kp and see that the
%   steady-state filtered time error to which the PLL converges becomes
%   smaller, at the expense of a noisier phase output due to the fact that
%   the PLL bandwidth increases with Kp.
%
%   4) Experiment with the damping ratio. When < 1 (underdamped system),
%   the settling time should be shorter, at the expense of overshoot. In
%   constrast, when zeta > 1 (overdamped), there should be no overshoot (or
%   oscillations) at the expense of a longer settling time.
%
%   5) Experiment with the noise bandwidth. Increase it and see a large
%   steady-state error-variance, but with faster acquisition. Decrease it
%   and see a reduced error variance, but with slower acquisition.
%
% Selected Bibliography:
% [1] Rice, Michael. Digital Communications: A Discrete-Time Approach.
% Appendix C.

%% Global Parameters
n_iterations = 10e3;  % Number of loop iterations
y_ppb        = 60;   % True residual frequency offset in ppb
y0_ppb       = 59;   % Estimated residual frequency offset in ppb
x0           = 6e3;   % Initial time offset (constant time error)
T            = 1/128; % SYNC Period
% Note: "y_ppb" is the true residual freq. offset after the best RTC
% increment value is configured during coarse syntonization. Meanwhile,
% "y0_ppb" is the residual frequency offset that is estimated during fine
% syntonization.

plot_en      = 0;

%% Run parameters

% Simulation configurations:
%
% Bn_times_Tl, zeta, noise_type, N
%
% where noise_type = 0) None; 1) Gaussian; 2) Erlang and "N" is the LS
% block length.

% Varying N
sim_configs = { ...
    0.01, 0.707, 2, 2^7; ...
    0.01, 0.707, 2, 2^8; ...
    0.01, 0.707, 2, 2^9; ...
    0.01, 0.707, 2, 2^10; ...
    0.01, 0.707, 2, 2^11; ...
    0.01, 0.707, 2, 2^12; ...
    0.01, 0.707, 2, 2^13;
    };

% Varying the damping factor:
% sim_configs = { ...
%     0.01, 0.5,   2, 128; ...
%     0.01, 0.707, 2, 128; ...
%     0.01, 1,     2, 128; ...
%     0.01, 2,     2, 128; ...
%     };

% Varying the noise bandwidth:
% sim_configs = { ...
%     1e-5, 0.707,   2, 128; ...
%     1e-4, 0.707,   2, 128; ...
%     1e-3, 0.707,   2, 128; ...
%     5e-3, 0.707,   2, 128; ...
%     0.01, 0.707,   2, 128; ...
%     0.05, 0.707,   2, 128; ...
%     0.1,  0.707,   2, 128; ...
%     };

% For the Gaussian noise:
gaussian_n_var   = 1e4;  % Noise variance

% For the Erlang noise:
queueing_mean_ns = 35e3;  % Mean queueing delay (in ns)
erlang_K         = 8;     % Erlang shape for delay pdf (see [5])

%% Constants

% Initial error in the frequency offset acquired during fine syntonization
% and the actual residual frequency offset (after coarse syntonization of
% the RTC):
y0_err_ppb = y_ppb - y0_ppb;

%% Run all the distinct simulation configurations

% Number of different simulation runs:
n_distinct_runs = length(sim_configs);

% Preallocate
x_f_ns            = zeros(n_iterations, n_distinct_runs);
x_hat_f_ns        = zeros(n_iterations, n_distinct_runs);
x_tilde_f_ns      = zeros(n_iterations, n_distinct_runs);
tll_x_f_err       = zeros(n_iterations, n_distinct_runs);
ls_x_f_err        = zeros(n_iterations, n_distinct_runs);
time_err          = zeros(n_iterations, n_distinct_runs);
time_err_filtered = zeros(n_iterations, n_distinct_runs);
expec_filt_err_ss = zeros(n_iterations, n_distinct_runs);
y_tilde           = zeros(n_iterations, n_distinct_runs);
y_hat             = zeros(n_iterations, n_distinct_runs);
time_track_err_mean = zeros(1, n_distinct_runs);
time_track_err_var  = zeros(1, n_distinct_runs);
time_ls_err_var     = zeros(1, n_distinct_runs);
freq_track_err_mean = zeros(1, n_distinct_runs);
freq_track_err_var  = zeros(1, n_distinct_runs);
freq_ls_err_var     = zeros(1, n_distinct_runs);
crlb_x0           = zeros(1, n_distinct_runs);
crlb_y            = zeros(1, n_distinct_runs);
legend_cell       = cell(n_distinct_runs, 1);

for i_run = 1:n_distinct_runs

    %% Run Parameters
    % Normalized loop bandwidth
    Bn_times_Tl = sim_configs{i_run, 1};

    % Damping factor
    zeta        = sim_configs{i_run, 2};

    % Noise Type:
    noise_type  = sim_configs{i_run, 3};

    % LS block length:
    N           = sim_configs{i_run, 4};

    fprintf('===== Run #%d ====\n', i_run);
    fprintf('Bn_times_Tl=\t %g\n', Bn_times_Tl);
    fprintf('zeta=\t %g\n', zeta);
    fprintf('noise_type=\t %g\n', noise_type);
    fprintf('N=\t %g\n', N);

    %% Get loop constants

    [ Kp, Ki ] = getPiConstants(Bn_times_Tl, zeta);

    %% Generate Noise

    % Then generate the time offset following a linear model, for the given
    % true frequency offset:
    switch (noise_type)
        case 0
            toffset_noise = zeros(N, n_iterations);
        case 1
            % Zero-mean Gaussian noise
            toffset_noise = sqrt(gaussian_n_var)*randn(N, n_iterations);
        case 2
            % Generate zero-mean Erlang noise (null mean implies that the
            % mean delay estimation is perfect):
            erlang_noise = sum(exprnd(queueing_mean_ns/erlang_K, ...
                n_iterations*N, erlang_K), 2).' - queueing_mean_ns;
            toffset_noise = reshape(erlang_noise, N, n_iterations);
    end

    %% Generate the noiseless (true) time offset values
    % The input signal does not depend on the loop processing, so we can
    % generate it before simulating the loop.

    % Consider n_iterations blocks of N time offset estimations and
    % generate the time vector:
    t = (0:(n_iterations*N)-1)*T;

    % Generate the time-offset estimations:
    x_true_ns = x0 + (y_ppb * t);
    % Note that "y_ppb" is the true residual frequency offset between the
    % slave and the master. During fine syntonization, the slave estimates
    % it to be "y0_ppb" and adopts this value in the TLL. However, "y0_ppb"
    % is not perfectly equal to "y_ppb", leading to a difference that is
    % intended to be corrected through the TLL.

    % Group them in blocks of N estimations each:
    x_true_ns_blocked = reshape(x_true_ns, N, numel(x_true_ns)/N);

    % Save the "true" final offset values of each block
    x_f_ns(:,i_run) = x_true_ns_blocked(end, :);

    %% Generate the noisy time offset "measurements"
    % Create vector of noisy time offset estimations:
    x_noisy_ns_blocked = x_true_ns_blocked + toffset_noise;

    %% Generate the time offset estimations using LS

    % Least-squares observation matrix:
    H = [ones(N, 1), (0:(N-1)).'*T];

    % Estimate a single time-offset using LS for each block:
    for i_block = 1:n_iterations
        ls_sol = H\x_noisy_ns_blocked(:, i_block);
        y_hat(i_block, i_run) = ls_sol(2);
        x_hat_f_ns(i_block, i_run) = ls_sol(1) + ls_sol(2)*N*T;
    end
    % Note that although "x_hat_ns" represents the input to the PLL, it has
    % to be computed distinctly for each because the noise type may be
    % adjusted,

    %% Compute the error between the true and LS-estimated final offsets
    ls_x_f_err(:, i_run) = x_hat_f_ns(:, i_run) - x_f_ns(:,i_run);

    %% TLL
    % Pass through the TLL:
    [ x_tilde_f_ns(:,i_run), ...
        time_err(:,i_run), ...
        time_err_filtered(:,i_run), ...
        y_tilde(:, i_run)] = ...
        time_locked_loop( x_hat_f_ns(:, i_run), Kp, Ki, y0_ppb, N*T );

    %% Compute the error between the true and TLL-estimated final offsets
    tll_x_f_err(:, i_run) = x_tilde_f_ns(:, i_run) - x_f_ns(:,i_run);

    %% Expected filter steady-state value
    % The values to which the time error and the filtered time error
    % converge depend on the loop order. For a second-order loop (with a PI
    % controller), assuming an input with fixed frequency offset, the time
    % error (filter input) converges to 0. In contrast, for a first-order
    % loop (with a Proportional controller only), the time error converges
    % to y0_err_ppb*N*T/Kp. In both cases, the filter output converges to
    % y0_err_ppb*N*T, which corresponds to the amount of time offset that
    % should be provided in addition to the initial slope correction value
    % per N SYNC periods.
    expec_filt_err_ss(:,i_run) = (y0_err_ppb*N*T)*ones(n_iterations,1);

    %% Legend for this run
    % For the Hybrid Scheme, get also the Ibias from the configuration
    %     conf_label = ['$B_n T_l = ', num2str(Bn_times_Tl), '$, '];
    %     conf_label = [conf_label, '$\zeta = ', num2str(zeta), '$, '];
    %     conf_label = [conf_label, '$N = ', num2str(N), '$'];
    conf_label = ['$N = ', num2str(N), '$'];

    % Add label to plot legend cell
    legend_cell{i_run} = conf_label;

    %% Time Tracking Error Analysis
    % Note that here the error is not the same as in the TLL's error
    % detector. Instead, it is the error relative to the true value.

    % "Steady-state" values:
    tll_xf_err_ss = tll_x_f_err(end -n_iterations/2:end, i_run);
    ls_x_f_err_ss = ls_x_f_err(end -n_iterations/2:end, i_run);

    % Steady-state Mean:
    time_track_err_mean(i_run) = mean(tll_xf_err_ss);

    fprintf('Steady-state time error mean:\t%g\n', time_track_err_mean(i_run));

    % Steady-state Variance:
    time_track_err_var(i_run) = var(tll_xf_err_ss);
    time_ls_err_var(i_run) = var(ls_x_f_err_ss);

    fprintf('TLL Steady-state time error var:\t%g\n', ...
        time_track_err_var(i_run));
    fprintf('LS Steady-state time error var:\t%g\n', ...
        time_ls_err_var(i_run));

    fprintf('Noise variance:\t%g\n', gaussian_n_var);

    %% Frequency Tracking Error Analysis

    % "Steady-state" values:
    y_tilde_err_ss = y_tilde(end -n_iterations/2:end, i_run) - y_ppb;
    y_hat_err_ss   = y_hat(end -n_iterations/2:end, i_run) - y_ppb;

    % Steady-state Mean:
    freq_track_err_mean(i_run) = ...
        mean(y_tilde_err_ss);

    fprintf('Steady-state frequency error mean:\t%g\n', ...
        freq_track_err_mean(i_run));

    % Steady-state Variance:
    freq_track_err_var(i_run) = var(y_tilde_err_ss);
    freq_ls_err_var(i_run)    = var(y_hat_err_ss);

    fprintf('TLL Steady-state time error var:\t%g\n', ...
        freq_track_err_var(i_run));
    fprintf('LS Steady-state time error var:\t%g\n', ...
        freq_ls_err_var(i_run));

    fprintf('Noise variance:\t%g\n', gaussian_n_var);

    %% Cramer-Rao Lower Bound
    noise_var = var(toffset_noise(:));
    crlb_x0(i_run) = (2*(2*N - 1)*noise_var)/(N*(N+1));
    crlb_y(i_run) = (12*noise_var)/(N*(N^2  -1));

end

%% Performance

% Array of plot markers:
S={'r-','g-o','-bx','-cs','-md', 'k-v', '-y.', 'r--','g--o','--bx', ...
    '--cs','--md', 'k--v', '--y.'};

S2={'r-','g-','-b','-c','-m', 'k-', '-y',...
    'r--','g--','--b','--c','--m', 'k--', '--y'};

if (plot_en)
    % True vs. LS-estimated Time Offsets
    % 'Note the different slopes come from different block durations'
    figure
    for iPlot = 1:n_distinct_runs
        plot(x_f_ns(:, iPlot), char(S2(iPlot)), 'linewidth', 1.2)
        hold on
    end
    for iPlot = (n_distinct_runs+1):2*n_distinct_runs
        plot(x_hat_f_ns(:, iPlot-n_distinct_runs), char(S2(iPlot)), ...
            'linewidth', 1.1)
        hold on
    end
    title('True vs LS-estimated final time offsets')
    xlabel('Observation Interval', 'Interpreter', 'latex')
    ylabel('Time (ns)')
    legend([legend_cell; legend_cell], 'Interpreter', 'latex')
    grid on
    xlim([0 50])

    % True vs. TLL-processed Time Offsets
    figure
    for iPlot = 1:n_distinct_runs
        plot(x_f_ns(:, iPlot), char(S2(iPlot)), 'linewidth', 1.2)
        hold on
    end
    for iPlot = (n_distinct_runs+1):2*n_distinct_runs
        plot(x_tilde_f_ns(:, iPlot-n_distinct_runs), char(S2(iPlot)), ...
            'linewidth', 1.1)
        hold on
    end
    title('True vs TLL-processed final time offsets')
    xlabel('Observation Interval', 'Interpreter', 'latex')
    ylabel('Time (ns)')
    legend([legend_cell; legend_cell], 'Interpreter', 'latex')
    grid on
    xlim([0 50])


    % Time Error
    figure
    for iPlot = 1:n_distinct_runs
        plot(time_err(:, iPlot), char(S(iPlot)))
        hold on
    end
    legend(legend_cell, 'Interpreter', 'latex')
    xlabel('Observation Interval', 'Interpreter', 'latex')
    ylabel('Error (ns)', 'Interpreter', 'latex')
    title('Time Error')

    % Filtered Time Error
    figure
    for iPlot = 1:n_distinct_runs
        plot(time_err_filtered(:, iPlot), char(S(iPlot)))
        hold on
    end
    hold on
    plot(expec_filt_err_ss, '--', 'linewidth', 1.2)
    legend([legend_cell;legend_cell], 'Interpreter', 'latex')
    xlabel('Observation Interval', 'Interpreter', 'latex')
    ylabel('Error (ns)', 'Interpreter', 'latex')
    title('Filtered Time Error')

    % Steady-state error of the TLL output relative to the true (noiseless)
    % time offset
    figure
    for iPlot = 1:n_distinct_runs
        plot(tll_x_f_err(end -500:end, iPlot), char(S(iPlot)))
        hold on
    end
    title('TLL Steady-state Time Error')
    xlabel('Observation Interval', 'Interpreter', 'latex')
    ylabel('Time (ns)', 'Interpreter', 'latex')
    legend(legend_cell, 'Interpreter', 'latex')
    xlim([0 500])
    grid on

    % Steady-state error of the LS output relative to the true (noiseless)
    % time offset
    figure
    for iPlot = 1:n_distinct_runs
        plot(ls_x_f_err(end -500:end, iPlot), char(S(iPlot)))
        hold on
    end
    title('LS Steady-state Time Error')
    xlabel('Observation Interval', 'Interpreter', 'latex')
    ylabel('Time (ns)')
    legend(legend_cell, 'Interpreter', 'latex')
    xlim([0 500])
    grid on

    % Instantaneous TLL-processed "fine" frequency offset estimation
    figure
    for iPlot = 1:n_distinct_runs
        plot(y_tilde(:, iPlot), char(S(iPlot)))
        hold on
    end
    title('Instantaneous "fine" frequency offset estimation')
    xlabel('Observation Interval', 'Interpreter', 'latex')
    ylabel('$\tilde{y}[n]$ (ppb)', 'Interpreter', 'latex')
    legend(legend_cell, 'Interpreter', 'latex')
    grid on

    % Instantaneous LS frequency offset estimations
    figure
    for iPlot = 1:n_distinct_runs
        plot(y_hat(:, iPlot), char(S(iPlot)))
        hold on
    end
    title('LS frequency offset estimations')
    xlabel('Observation Interval', 'Interpreter', 'latex')
    ylabel('$\tilde{y}[n]$ (ppb)', 'Interpreter', 'latex')
    legend(legend_cell, 'Interpreter', 'latex')
    grid on

    % Time Tracking Error Mean
    figure
    semilogx(cat(1,sim_configs{:,4}), time_track_err_mean)
    xlabel('$N$', 'Interpreter', 'latex')
    ylabel('(ns)', 'Interpreter', 'latex')
    set(gca, 'XTick', cat(1,sim_configs{:,4}))
    grid on
    title('Mean of the Time Tracking Error')

    % Time Tracking Error Standard Deviation
    figure
    semilogx(cat(1,sim_configs{:,4}), sqrt(time_track_err_var), ...
        'linewidth', 1.1)
    hold on
    semilogx(cat(1,sim_configs{:,4}), sqrt(time_ls_err_var), '--', ...
        'linewidth', 1.1)
    hold on
    semilogx(cat(1,sim_configs{:,4}), sqrt(crlb_x0), '-d', 'linewidth', 1.1)
    xlabel('$N$', 'Interpreter', 'latex')
    ylabel('Error STD (ns)', 'Interpreter', 'latex')
    set(gca, 'XTick', cat(1,sim_configs{:,4}))
    title('Standard Deviation of the Time Tracking Error')
    h = legend('TLL', 'LS Alone', 'CRLB for $x_0$');
    set(h, 'Interpreter', 'latex')
    grid on

    % Tracking Error Standard Deviation
    figure
    semilogx(cat(1,sim_configs{:,4}), 10*log10(time_track_err_var./crlb_x0))
    hold on
    semilogx(cat(1,sim_configs{:,4}), 10*log10(time_ls_err_var./crlb_x0))
    hold off
    xlabel('$N$', 'Interpreter', 'latex')
    ylabel('(ns)', 'Interpreter', 'latex')
    set(gca, 'XTick', cat(1,sim_configs{:,4}))
    title('Tracking Error Loss from the CRLB')
    legend('TLL', 'LS Alone')
    grid on

    % Frequency Tracking Error Standard Deviation
    figure
    semilogx(cat(1,sim_configs{:,4}), sqrt(freq_track_err_var), ...
        'linewidth', 1.1)
    hold on
    semilogx(cat(1,sim_configs{:,4}), sqrt(freq_ls_err_var), '-d', ...
        'linewidth', 1.5, 'markersize', 10)
    hold on
    semilogx(cat(1,sim_configs{:,4}), sqrt(crlb_y/(T^2)), '--*', ...
        'linewidth', 1.0)
    xlabel('$N$', 'Interpreter', 'latex')
    ylabel('Error STD (ppb)', 'Interpreter', 'latex')
    set(gca, 'XTick', cat(1,sim_configs{:,4}))
    title('Standard Deviation of the Frequency Tracking Error')
    h = legend('TLL', 'LS Alone', 'CRLB for $\hat{y}$')
    set(h, 'Interpreter', 'latex')
    grid on

    %% Delay
    [n_occur, d_bin] = hist(erlang_noise, 1e3);

    figure
    plot(d_bin*1e-3, n_occur/sum(n_occur))
    xlabel('Time Offset Noise (microseconds)', 'Interpreter', 'latex')
    ylabel('Pdf', 'Interpreter', 'latex')
    title('Noise Distribution', 'Interpreter', 'latex')
    grid on

end

fprintf('Simulation ended\n');
