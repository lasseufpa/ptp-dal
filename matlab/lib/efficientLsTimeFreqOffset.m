function [ x_ns, x_sec, y_hat ] = efficientLsTimeFreqOffset( x_obs_ns, ...
    x_obs_sec, i_est, N, sync_rate )
% Efficient LS implementation for time/frequency offset estimation
%
% Computes the LS solution using pre-computed matrices and approximating
% the time vector as composed by integer multiples of the SYNC period,
% rather than composed by the actual SYNC arrival timestamps (t2). This
% allows huge savings in both computational cost and memory usage.
%
% %%%%%%%%%%%%%%%%%
%
% Input:
% x_obs_ns  -> i-th ns time offset estimation
% x_obs_ns  -> i-th sec time offset estimation
% i_est     -> index i
% N         -> Selection window length (number of observations)
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
%   x[n] = x_0 + y*t[n]
%
% where x_0 is the initial time offset within the selection window and y
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
% Q_1 is the accumulator for the sum of x[n]
% Q_2 is the accumulator for the sum of n*x[n]

if (i_est == 1)
    Q_1 = 0; % Reset accumulator
    Q_2 = 0; % Reset accumulator

    % Check whether the first offset is closer to the "next higher second"
    % in order to minimize the magnitude of the ns offset numbers. This
    % way, it is more likely that the observations "x_obs" will oscillate
    % over positive and negative values. This approach is safe, since in
    % the end the ns offset is processed once again to avoid negative
    % offsets (by changing the sec offset).
    if (x_obs_ns > 5e8) % 5e8 is half a second in ns
        x_obs_sec_start = x_obs_sec + 1; % Sample in the beginning
    else
        x_obs_sec_start = x_obs_sec;
    end
end

% Current observation - time offset in ns:
x_obs = x_obs_ns + 1e9*(x_obs_sec - x_obs_sec_start);
% Consider the fluctuations within the vector of time offsets in seconds.

% SYNC Period:
T = (1/sync_rate);

% Measurement index (starting from 0):
n   = i_est - 1;

% Define
%   Q = H' * x_vector
%
% As explained above, the two elements of Q (Q_1 and Q_2) can be computed
% iteratively by accumulators:
Q_1 = Q_1 + x_obs; % sum of x[n]
Q_2 = Q_2 + (n * x_obs); % sum of n*x[n]

%% Default output (before the end of the selection window)
% Default offset output (bypassed input):
x_ns = x_obs_ns;
x_sec = x_obs_sec;

% Return no frequency offset by default
y_hat = 0;

% Note that these are not used before the end of the observation interval,
% anyways.

%% Actual estimation (at the end of the selection window)
if (i_est == N)
    % Computed the 2x2 matrix that multiplies the vector containing the
    % accumulators Q_1 and Q_2
    P = (2 / (N*(N+1))) * [(2*N - 1), -3; -3, 6/(N-1)];
    % Note this matrix P depends on both N and T.

    % Complete the efficient least-squares estimation:
    Theta = P * [Q_1; Q_2];

    % Estimated initial time-offset in ns:
    x_0_hat = Theta(1);

    % Estimated freq-offset in ppb:
    y_hat_times_T = Theta(2);
    y_hat = y_hat_times_T / T;

    % Last value of a "fitted" time-offset vector in ns:
    x_fit_end = x_0_hat + (N*y_hat_times_T);

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
