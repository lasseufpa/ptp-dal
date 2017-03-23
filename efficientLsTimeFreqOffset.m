function [ x_ns, x_sec, y ] = efficientLsTimeFreqOffset( x_obs_ns, ...
    x_obs_sec, i_est, w_len, sync_rate )
% Efficient LS implementation for time/frequency offset estimation
%
% Computes the LS solution using pre-computed matrices and approximating
% the time vector as composed by integer multiples of the SYNC period,
% rather than composed by the actual SYNC master timestamps (t1). This
% allows huge savings in both computational cost and memory usage.
%
% %%%%%%%%%%%%%%%%%
%
% Input:
% x_obs_ns  -> i-th ns time offset estimation
% x_obs_ns  -> i-th sec time offset estimation
% i_est     -> index i
% w_len     -> Selection window length (number of observations)
% sync_rate -> SYNC rate, used to compute the LS time vector
%
% Outputs:
% x_ns  -> Estimated time offset in nanoseconds
% x_sec -> Estimated time offset in seconds
% y     -> Estimated frequency offset
%
% %%%%%%%%%%%%%%%%%
%
% The following linear model is assumed for time offsets:
%
%   x[n] = x[0] + y*t[n]
%
% where x[0] is the initial time offset within the selection window and y
% is the frequency offset in ppb, namely the increase/decrease in the time
% offset over time given in ns per second. For a block of N samples, then,
% this model can be put in matrix form as:
%
%   x_vector = [ones(N, 1), t_vector] * [x[0]_est + y_est]';
%
%  |-------|  |----------------------|  |------------------|
%                    called "H"          vector of unknowns
%   (N x 1)          (N x 2)                  (2 x 1)
%
% The trick adopted for the efficient implementation is to assume
% "t_vector" as composed by integer multiples of the SYNC period. By doing
% so, matrix H can be computed ahead of time. Consequently, matrix
% "inv(H'*H)" that is used within the LS solution can also be pre-computed.
%
% The remainder of the "inv(H'*H)*H'*x_vector" LS solution is the product
% "H'*x_vector". Since matrix "H'" (the transpose of H) has all ones in its
% first row, the first element of "H'*x_vector" is simply the sum of
% "x_vector", namely the sum of all time offset observations. Hence, it can
% be computed iteratively by an accumulator. Finally, since the second row
% of "H'" is the transpose of "t_vector", the second element of
% "H'*x_vector" can also be computed iteratively.
%
% When the window length is achieved, the final result can be computed very
% efficiently by multiplying the pre-computed ""inv(H'*H)" (a 2x2 matrix)
% with the iteratively computed "H'*x_vector". In the end, no memory
% storage is need to retain the history of obervations in "x_vector", so
% arbitrarily large selection windows can be employed.

persistent Q_1 Q_2 x_obs_sec_start

if (i_est == 1)
    Q_1 = 0; % Reset accumulator
    Q_2 = 0; % Reset accumulator
    x_obs_sec_start = x_obs_sec; % Sample in the beginning
end

% Observed time offset in ns:
x_obs = x_obs_ns + 1e9*(x_obs_sec - x_obs_sec_start);
% Consider the fluctuations within the vector of time offsets in seconds.


sync_period = (1/sync_rate);

% Define
%   Q = H' * x_vector
%
% As explained above, the two elements of Q (Q_1 and Q_2) can be computed
% iteratively by accumulators:
Q_1 = Q_1 + x_obs;
Q_2 = Q_2 + ((i_est - 1) * sync_period * x_obs);

%% Default output (before the end of the selection window)
% Default offset output (bypassed input):
x_ns = x_obs_ns;
x_sec = x_obs_sec;

% Return no frequency offset by default
y = 0;

%% Actual estimation (at the end of the selection window)
if (i_est == w_len)

    % Load the precomputed 2x2 matrix containing the result of "inv(H'*H)":
    filename = ['data/inv_H_star_H_loglen_', num2str(log2(w_len)),...
        '_syncrate_', num2str(sync_rate), '.mat'];
    load(filename);

    % Complete the efficient least-squares estimation:
    time_freq_offset_est = P * [Q_1; Q_2];

    % Estimated initial time-offset in ns:
    x_init = time_freq_offset_est(1);

    % Estimated freq-offset in ppb:
    y = time_freq_offset_est(2);

    % Last value of a "fitted" time-offset vector in ns:
    x_fit_end = x_init + ((w_len - 1)*sync_period*y);

    % Resulting time offsets:
    x_sec = x_obs_sec_start;
    x_ns  = x_fit_end;

    % After the above step, check whether a wrap occurs within the ns
    % counter and adjust accordingly:
    while (x_ns >= 1e9)
        x_ns = x_ns - 1e9;
        x_sec = x_sec + 1;
    end

    while (x_ns < 0)
        x_ns = x_ns + 1e9;
        x_sec = x_sec - 1;
    end
end

end

