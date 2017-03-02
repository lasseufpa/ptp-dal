% Precision Time Protocol (PTP) simulator
%
% Terminology:
% - RTC             Real-time Clock
% - SYNC            Frame periodically sent through PTP for synchronization
% - Synchronized    Aligned in time and frequency.
% - Syntonized      Aligned in frequency only.
%

clearvars, clc

%% Debug

log_ptp_frames = 1;
print_offsets  = 1;
debug_scopes   = 1;
print_sim_time = 1;

%% Parameters and Configurations

%%%%%% Simulation %%%%%%
t_step_sim       = 1e-9;    % Simulation Time Step (resolution)

%%%%%%%%% RTC %%%%%%%%%%
nominal_rtc_clk  = 125e6;   % Nominal frequency of the RTC clk

% Master RTC
Rtc(1).freq_offset_ppb     = 0;
Rtc(1).init_time_sec       = 20;
Rtc(1).init_time_ns        = 5000;
Rtc(1).init_rising_edge_ns = 0;
% Note: use the init time to assume that the device is turned on x sec/ns
% earlier with respect to other RTCs. Meanwhile, use the
% "init_rising_edge_ns" to assume that the first clock rising edges occurs
% at y ns away from simulation time 0.

% Slave RTC
Rtc(2).freq_offset_ppb     = 50;
Rtc(2).init_time_sec       = 0;
Rtc(2).init_time_ns        = 0;
Rtc(2).init_rising_edge_ns = 3;

%%%%%%%%% PTP %%%%%%%%%%
rtc_inc_est_period = 2;     % RTC increment estimation period in frames
sync_rate          = 128;   % SYNC rate in frames per second
perfect_delay_est  = 1;     % Enable for perfect delay estimations

%%%%%%% Network %%%%%%%%
% Queueing statistics
queueing_mean      = 5e-6;  % Mean queueing delay (in seconds)
erlang_K           = 2;     % Erlang shape for delay pdf (see [5])

%% Constants
nRtcs              = length(Rtc);
default_rtc_inc_ns = (1/nominal_rtc_clk)*1e9;
sync_period        =  1 / sync_rate;

%% Derivations

% Derive the actual clock frequencies (in Hz) of the clock signals that
% drive each RTC. Note these frequencies are assumed to be fixed (not
% disciplined and not drifting) over the simulation time. This is
% reasonable given in PTP implementations the actual clock is never
% corrected (only the RTC parameters are adjusted) and since
% temperature-driven variations occurr at relatively long intervals.
for iRtc = 1:nRtcs
    % Frequency offset in Hz
    freq_offset_Hz = Rtc(iRtc).freq_offset_ppb * (nominal_rtc_clk/1e9);

    % Corresponding clock frequency (in Hz)
    Rtc(iRtc).clk_freq = nominal_rtc_clk + freq_offset_Hz;

    % Clock period
    Rtc(iRtc).clk_period = (1/Rtc(iRtc).clk_freq);
end

%% System Objects

% Time Offset Observation
hScopeTime = dsp.TimeScope(...
    'Title', 'Actual Ns Offset', ...
    'NumInputPorts', 1, ...
    'ShowGrid', 1, ...
    'ShowLegend', 1, ...
    'BufferLength', 1e5, ...
    'TimeSpanOverrunAction', 'Scroll', ...
    'TimeSpan', 1e3*sync_period, ...
    'TimeUnits', 'Metric', ...
    'SampleRate', 1/sync_period, ...
    'YLimits', 50*[-1 1]);

% Frequency Offset Observation
hScopeFreq = dsp.TimeScope(...
    'Title', 'Estimated Frequency Offset (ppb)', ...
    'NumInputPorts', 1, ...
    'ShowGrid', 1, ...
    'ShowLegend', 1, ...
    'BufferLength', 1e5, ...
    'TimeSpanOverrunAction', 'Scroll', ...
    'TimeSpan', 1e3*sync_period, ...
    'TimeUnits', 'Metric', ...
    'SampleRate', 1/sync_period, ...
    'YLimits', 4e3*[-1 1]);

%% Priority Queue

eventQueue = java.util.PriorityQueue;

%% Initialization

% Starting simulation time
t_sim      = 0;
prev_t_sim = 0;

for iRtc = 1:nRtcs
    % Initialize the Seconds RTC count
    Rtc(iRtc).sec_cnt = Rtc(iRtc).init_time_sec;

    % Initialize the Nanoseconds RTC count
    Rtc(iRtc).ns_cnt = Rtc(iRtc).init_time_ns;

    % Initialize the RTC Time Offset Register
    Rtc(iRtc).time_offset.ns = 0;
    Rtc(iRtc).time_offset.sec = 0;
    % Note: the offset register is continuously updated with the time
    % offset estimations and starts by default zeroed.

    % Initialize the increment value:
    Rtc(iRtc).inc_val_ns = default_rtc_inc_ns;
    % Note: it should start with the default increment corresponding to the
    % nominal period of the clock the feeds the RTC.
end

% Init variables
Sync.on_way     = 0;
i_sync_rx_event = 0;
next_sync_tx    = 0;
rtc_error_ns    = 0;
norm_freq_offset  = 0;
i_iteration     = 0;

%% Infinite Loop
while (1)

    i_iteration  = i_iteration + 1;

    t_sim_ns = mod(t_sim*1e9, 1e9);
    t_sim_sec = floor(t_sim);

    %% Time elapsed since the previous iteration
    elapsed_time = t_sim - prev_t_sim;

    %% RTC Increments
    for iRtc = 1:nRtcs

        % Check how many times the RTC has incremented so far:
        nIncs = floor(...
            (t_sim - Rtc(iRtc).init_rising_edge_ns*1e-9) ...
            / Rtc(iRtc).clk_period );

        % Prevent negative number of increments
        if (nIncs < 0)
            nIncs = 0;
        end

        % Update the RTC nanoseconds/seconds counts:
        Rtc(iRtc).ns_cnt = mod(nIncs * Rtc(iRtc).inc_val_ns, 1e9);
        Rtc(iRtc).sec_cnt = floor(nIncs/nominal_rtc_clk);
    end

    if (print_sim_time)
        fprintf('--- Simulation Time: ---\n');
        fprintf('t_sim\t%gs, %gns\tRTC1\t%gs, %gns\tRTC2\t%gs, %gns\n', ...
            t_sim_sec, ...
            t_sim_ns, ...
            Rtc(1).sec_cnt, ...
            Rtc(1).ns_cnt, ...
            Rtc(2).sec_cnt, ...
            Rtc(2).ns_cnt);
    end

    %% SYNC Transmissions

    % Check when it is time to transmit a SYNC frame
    if (t_sim >= next_sync_tx)

        if (log_ptp_frames)
            fprintf('--- Event: ---\n');
            fprintf('Sync transmitted at time %g\n', t_sim);
        end

        % Timestamp the departure time:
        Sync.t1.ns  = Rtc(1).ns_cnt;
        Sync.t1.sec = Rtc(1).sec_cnt;
        % Note: timestamps are always taken using the syntonized RTC
        % values, not the synchronized RTC. By doing so, the actual
        % time-offset of the slave RTC w.r.t. the master will always be
        % present and potentially could be estimated. In PTP
        % implementations, the goal is firstly to synchronize the RTC
        % increment values, such that the time offset between the two
        % (master and slave) RTCs eventually becomes constant (not varying
        % over time). After that, by estimating precisely this constant
        % time offset and updating the time offset registers, a stable
        % synchronized time scale can be derived at the slave RTC.

        % Mark the SYNC frame as "on its way" towards the slave
        Sync.on_way = 1;

        % Generate a random queueing delay
        if (perfect_delay_est)
            queuing_delay = queueing_mean;
        else
            queuing_delay = sum(...
                exprnd(queueing_mean/erlang_K, 1, erlang_K) ...
                );
        end

        % Schedule the event for sync reception
        next_sync_rx = t_sim + queuing_delay;
        eventQueue.add(next_sync_rx);

        % Schedule the next SYNC transmission
        next_sync_tx = next_sync_tx + sync_period;
        eventQueue.add(next_sync_tx);
    end

    %% SYNC Receptions

    % Process the SYNC frame at the destination
    if (t_sim >= next_sync_rx)

        % Clear "on way" status
        Sync.on_way = 0;

        % Timestamp the arrival time (t2) at the slave side:
        Sync.t2.ns = Rtc(2).ns_cnt;
        Sync.t2.sec = Rtc(2).sec_cnt;
        % Again, using syntonized-only timestamps.

        % First save the previous time offset estimation:
        prev_rtc_error_ns = rtc_error_ns;

        % Increment time offset number
        i_sync_rx_event = i_sync_rx_event + 1;

        if (log_ptp_frames)
            fprintf('--- Event: ---\n');
            fprintf('Sync[%d] received at time %g\n', ...
                i_sync_rx_event, t_sim);
        end

        % SYNC delay estimation:
        sync_route_delay_ns = queueing_mean * 1e9;
        % Note: while PTP delay is not implemented in this simulation,
        % assume the mean is known a priori and use it for time offset
        % estimations.

        % Master Timestamp corrected by the delay
        master_ns_sync_rx  = Sync.t1.ns + sync_route_delay_ns;
        master_sec_sync_rx = Sync.t1.sec;
        % Note: the correction should yield the time at the master side
        % when the SYNC frame arrives at the slave side.

        % Corresponding slave time
        slave_ns_sync_rx  = Sync.t2.ns;
        slave_sec_sync_rx = Sync.t2.sec;

        % Check whether the ns count has wrapped
        if (master_ns_sync_rx >= 1e9)
            % If it did wrap, bring the ns count back between 0 and 1e9
            master_ns_sync_rx = master_ns_sync_rx - 1e9;
            % And add one extra second:
            master_sec_sync_rx = master_sec_sync_rx + 1;
        end

        % RTC error:
        Rtc_error.ns = master_ns_sync_rx - slave_ns_sync_rx;
        % Note: it is actually computed as "(t1 + d) - t2" here.
        Rtc_error.sec = master_sec_sync_rx - slave_sec_sync_rx;

        % The nanoseconds error can not be negative. It has to be a number
        % between 0 and 1e9. Any negative ns offset can be corrected by a
        % positive ns offset plus a correction within the seconds count.
        if (Rtc_error.ns < 0)
            Rtc_error.ns = Rtc_error.ns + 1e9;
            Rtc_error.sec = Rtc_error.sec - 1;
        end

        % Update the RTC time offset registers
        Rtc(2).time_offset.ns = Rtc_error.ns;
        Rtc(2).time_offset.sec = Rtc_error.sec;
        % Note: the error is saved on the time offset registers, and never
        % corrected in the actual ns/sec count. The two informations are
        % always separately available and can be summed together to form a
        % synchronized (correct) ns/sec count.

        % Check if the desirable number of SYNC receptions was already
        % reached for estimating the frequency offset
        if (i_sync_rx_event == rtc_inc_est_period)
            % Reset SYNC event counter
            i_sync_rx_event = 0;

            % Duration in ns at the master side between the two SYNCs:
            master_sync_interval_ns = ...
                master_ns_sync_rx - prev_master_ns_sync_rx;
            % In practice, this interval could be known a priori. However,
            % since a standardized PTP master does not need to inform the
            % SYNC rate to the slave, a generic implementation would
            % measure the sync interval.

            % Check a negative duration, which can happen whenever the ns
            % counter (used for the timestampts) wraps:
            if (master_sync_interval_ns < 0)
                master_sync_interval_ns = master_sync_interval_ns + 1e9;
            end

            % Duration at the slave side between the two SYNC frames:
            slave_sync_interval_ns = ...
                slave_ns_sync_rx - prev_slave_ns_sync_rx;

            % Check a negative duration, which, again, happens whenever the
            % ns counter (used for the timestampts) wraps:
            if (slave_sync_interval_ns < 0)
                slave_sync_interval_ns = slave_sync_interval_ns + 1e9;
            end

            % Slave error in ns:
            slave_error_ns = ...
                slave_sync_interval_ns - master_sync_interval_ns;
            % Note: a positive "slave_error_ns" means the duration for the
            % master was smaller than the duration measured for the slave,
            % namely that the slave's RTC is running faster. Thus, the
            % slave's increment value has to be reduced. The sign of
            % the "norm_freq_offset" in the computation of the
            % "slave_est_clk_freq" below takes care of that.

            % Estimate the Normalized Frequency offset
            norm_freq_offset = slave_error_ns / master_sync_interval_ns;
            % Note: by definition, a normalized frequency offset
            % corresponds to the time error accumulated over 1 second. For
            % example, an offset expressed in ppb corresponds to the time
            % offset in nanoseconds accumulated at the slave RTC w.r.t the
            % master RTC after 1 full second.

            % Compute the new increment value for the slave RTC:
            slave_est_clk_freq = (1 + norm_freq_offset) * nominal_rtc_clk;
            Rtc(2).inc_val_ns = (1 / slave_est_clk_freq)*1e9;
            % Note: infinite precision for the increment value is assumed
            % here. However, in practice the RTC increment value would be
            % represented by a fixed-point number with limited number of
            % fractional (subnanoseconds bits).

            % Plot to scope
            if (debug_scopes)
                step(hScopeFreq, norm_freq_offset*1e9);
            end
        end

        % Save the sync arrival timestamps for the next iteration:
        prev_master_ns_sync_rx = master_ns_sync_rx;
        prev_slave_ns_sync_rx  = slave_ns_sync_rx;

        %% Synchronized RTC Values

        master_rtc_sync_ns  = Rtc(1).ns_cnt  + Rtc(1).time_offset.ns;
        master_rtc_sync_sec = Rtc(1).sec_cnt + Rtc(1).time_offset.sec;
        % Check for ns wrap:
        if (master_rtc_sync_ns >= 1e9)
            master_rtc_sync_ns  = master_rtc_sync_ns - 1e9;
            master_rtc_sync_sec = master_rtc_sync_sec + 1;
        end

        slave_rtc_sync_ns   = Rtc(2).ns_cnt  + Rtc(2).time_offset.ns;
        slave_rtc_sync_sec  = Rtc(2).sec_cnt + Rtc(2).time_offset.sec;
        % Check for ns wrap:
        while (slave_rtc_sync_ns >= 1e9)
            slave_rtc_sync_ns  = slave_rtc_sync_ns - 1e9;
            slave_rtc_sync_sec = slave_rtc_sync_sec + 1;
        end

        %% Actual Errors
        % In order to properly compute the RTC errors, we have to compare
        % unwrapped measures of the nanosecond count.
        rtc_1_ns_unwrapped = master_rtc_sync_ns + 1e9*master_rtc_sync_sec;
        rtc_2_ns_unwrapped = slave_rtc_sync_ns + 1e9*slave_rtc_sync_sec;

        if(isnan(rtc_2_ns_unwrapped))
            error('Nan');
        end

        % Actual ns error
        actual_ns_error = rtc_1_ns_unwrapped - rtc_2_ns_unwrapped;

        % Plot to scope
        if (debug_scopes)
            step(hScopeTime, actual_ns_error);
        end

        if (print_offsets)
            fprintf('--- Estimated Offsets: ---\n');
            fprintf('TOffset\t%g ns\tFOffset\t%g ppb\n', ...
                actual_ns_error, ...
                norm_freq_offset*1e9);
        end
    end

    %% Increase simulation step

    % Save previous t_sim
    prev_t_sim = t_sim;

    % Check the next scheduled event
    next_event = eventQueue.poll();

    if isempty(next_event)
        t_sim = t_sim + t_step_sim;
        warning('Empty next event');
    else
        % Jump simulation time to the next event
        t_sim = next_event;
    end
end


