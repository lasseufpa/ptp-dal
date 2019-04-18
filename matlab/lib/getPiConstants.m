function [ Kp, Ki ] = getPiConstants( Bn_times_Tl, eta )
% Computes the PI constants for the target noise bandwidth and damping
% factor

Ke = 1; % Time Detector Gain
K0 = 1; % "DDS" gain

% Theta_n (a definition in terms of T and Bn)
Theta_n = (Bn_times_Tl)/(eta + (1/(4*eta)));
% Obs: in that case the loop "sampling period" is actually N*T

% Constants obtained by analogy to the continuous time transfer function:
Kp_K0_K1 = (4 * eta * Theta_n) / (1 + 2*eta*Theta_n + Theta_n^2);
Kp_K0_K2 = (4 * Theta_n^2) / (1 + 2*eta*Theta_n + Theta_n^2);

% Kp and Ki (PI constants):
Kp = Kp_K0_K1/(Ke*K0); % Proportional Constant
Ki = Kp_K0_K2/(Ke*K0); % Integral Constant

fprintf('Kp =\t %g\n', Kp);
fprintf('Ki =\t %g\n', Ki);
end

