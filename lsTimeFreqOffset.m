function [ x_ns, x_sec, y ] = lsTimeFreqOffset( toffset_sel_window )
% Least-squares estimator for time and frequency offsets
%
% Assumes the following linear model for the time offsets and estimates
% both the initial time offset and the frequency offset:
%
%   x[n] = x[0] + y*t[n]
%
% where x[0] is the initial time offset within the selection window and y
% is the frequency offset in ppb, namely the increase/decrease in the time
% offset over time given in ns per second.
%
% Input:
% toffset_sel_window -> Array of structs containing selection window values
%
% Outputs:
% x_ns  -> Estimated time offset in nanoseconds
% x_sec -> Estimated time offset in seconds
% y     -> Estimated frequency offsets

% Extract the content from the selection window
t_n       = cat(1,toffset_sel_window.t)/1e9; % Time vector
x_obs_ns  = cat(1, toffset_sel_window.ns);   % Observed ns time offsets
x_obs_sec = cat(1, toffset_sel_window.sec);  % Observed sec time offsets
w_len     = length(toffset_sel_window);      % Selection window length

% Observation Matrix
H = [ones(w_len, 1), t_n];
% The second column of the observation matrix is the time in seconds, so
% that the resulting slope "y" is in ns / second (namely in ppb).

% Observed time offsets in ns:
x_obs = x_obs_ns + 1e9*(x_obs_sec - x_obs_sec(1));
% Note the fluctuations within the vector of time
% offsets in seconds are considered.

% Estimate using Least-Squares:
time_freq_offset_est = ...
    H \ x_obs;
% Resulting initial time-offset in ns:
x_init = time_freq_offset_est(1);
% Resulting freq-offset in ppb:
y = time_freq_offset_est(2);

% "Fitted" time-offset vector in ns:
x_fit = x_init + t_n*y;

% Predicted time offsets:
x_sec = x_obs_sec(1);
x_ns  = x_fit(end);

% After the above step, check whether a wrap occurs within the ns counter
% and adjust accordingly:
if (x_ns >= 1e9)
    x_ns = x_ns - 1e9;
    x_sec = x_sec + 1;
elseif (x_ns < 0)
    x_ns = x_ns + 1e9;
    x_sec = x_sec - 1;
end

end

