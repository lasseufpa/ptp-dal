function [ x_ns, x_sec, y ] = sampleMeanEstimator( toffset_sel_window )
% Sample-mean estimator for time and frequency offsets
%   "Select" (compute) a ns and a sec time offset estimation.
%
% Input:
% toffset_sel_window -> Array of structs containing selection window values
%
% Outputs:
% x_ns  -> Estimated time offset in nanoseconds
% x_sec -> Estimated time offset in seconds
% y     -> Estimated frequency offsets

persistent prev_x_ns prev_x_sec;

% Extract the content from the selection window
t_n_sec   = cat(1,toffset_sel_window.t)/1e9; % Time vector
x_obs_ns  = cat(1, toffset_sel_window.ns);   % Observed ns time offsets
x_obs_sec = cat(1, toffset_sel_window.sec);  % Observed sec time offsets

%% Time offset estimation

% First compute the mean of the sec offsets:
mean_sec      = mean(x_obs_sec);

% The seconds error to be updated in the RTC is the integer part of that:
x_sec = round(mean_sec);

% The ns error, in turn, is the mean of the ns time offsets in the
% selection window + the fractional part of the mean sec time offset:
x_ns = round(mean(x_obs_ns)) + round((mean_sec - x_sec) * 1e9);

% After the above step, check whether a wrap occurs within the ns counter
% and adjust accordingly:
if (x_ns >= 1e9)
    x_ns = x_ns - 1e9;
    x_sec = x_sec + 1;
elseif (x_ns < 0)
    x_ns = x_ns + 1e9;
    x_sec = x_sec - 1;
end

%% Frequency offset

% Change in the time offset w.r.t the previous estimation:
delta_toffset_ns = (x_ns - prev_x_ns) + 1e9*(x_sec - prev_x_sec);
% Window duration:
w_duration_sec = t_n_sec(end) - t_n_sec(1);

% Estimated frequency offset in ppb
y = delta_toffset_ns / w_duration_sec;

% Save current time offset estimations for the next iteration:
prev_x_ns  = x_ns;
prev_x_sec = x_sec;

end

