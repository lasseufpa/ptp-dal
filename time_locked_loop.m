function [ x_tilde_ns, err, err_filt, y_tilde ] = time_locked_loop( x_hat_ns, Kp, Ki, y0, Tl )
% Time Locked Loop
%
% Input Parameters:
%   x_hat_ns -> Estimated time offsets at the input
%   Kp       -> Proportional constant
%   Kp       -> Integral constant
%   y0       -> Initial frequency offset (in ppb) used in the DDS
%   Tl       -> Loop "sampling period"

% Infer the number of samples (number of iterations):
n_samples = length(x_hat_ns);

% Nominal "slope correction" adopted in the DDS:
slope_corr_0 = y0*Tl

%% Loop

% Preallocate
toffset_reg = zeros(n_samples, 1); % Time Offset Register
x_tilde_ns  = zeros(n_samples, 1); % Generated time offset
err         = zeros(n_samples, 1); % Time Error
err_filt    = zeros(n_samples, 1); % Filtered Time Error
y_tilde     = zeros(n_samples, 1); % Filtered Time Error

% Initialize integral filter output
integral_out = 0;

for i = 1:n_samples

    %% Loop "DDS":
    x_tilde_ns(i) = toffset_reg(i);

    %% Error Detector

    % Error between the time offset estimation (coming from e.g. an LS
    % estimator) and the predicted time offset that the loop infers based
    % on its current knowledge of the time offset slope:
    err(i) = x_hat_ns(i) - x_tilde_ns(i);

    %% Loop Filter
    % The loop filter consists of a Proportional+Integral (PI) controller

    % Proportional term
    proportional_out = err(i)*Kp;

    % Integral term
    integral_out     = err(i)*Ki + integral_out;

    % PI Filter output:
    err_filt(i) = proportional_out + integral_out;

    %% Update the time offset register
    % The angle of the DDS complex exponential in the next clock cycle
    % results from the sum between the nominal phase increment and the
    % filtered phase error term:
    toffset_reg(i+1) = toffset_reg(i) + slope_corr_0 + err_filt(i);

    %% Instaneous "fine" frequency offset estimation
    y_tilde(i) = (slope_corr_0 + err_filt(i)) / Tl;

end

end

