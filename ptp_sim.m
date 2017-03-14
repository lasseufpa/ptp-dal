% Precision Time Protocol (PTP) simulator
%
% Terminology:
% - RTC             Real-time Clock
% - SYNC            Frame periodically sent through PTP for synchronization
% - Synchronized    Aligned in time and frequency.
% - Syntonized      Aligned in frequency only.
%
% In this simulator, following common practices for PTP implementations,
% the timestamps that are actually added to PTP frames are always taken
% using the syntonized RTC values, not the synchronized RTC. By doing so,
% the actual time-offset of the slave RTC w.r.t. the master will always be
% present and potentially could be estimated. In PTP implementations, the
% goal is firstly to synchronize the RTC increment values, such that the
% time offset between the two (master and slave) RTCs eventually becomes
% constant (not varying over time). After that, by estimating precisely
% this constant time offset and updating the time offset registers, a
% stable synchronized time scale can be derived at the slave RTC.

clearvars, clc

%% Debug

log_ptp_frames           = 0;
print_true_time_offsets  = 0;
print_freq_offset_est    = 1;
debug_scopes             = 1;
print_sim_time           = 0;

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
pdelay_req_rate    = 8;     % Pdelay_req rate in frames per second
perfect_delay_est  = 0;     % Enable for perfect delay estimations
foffset_thresh_ppb = 1e4;   % Maximum frequency offset correction in ppb
% RTC Increment Value
en_fp_inc_val      = 0;     % Simulate increment value as a fixed-point num
n_inc_val_int_bits = 26;    % Total number of bits in the increment value
n_inc_val_frc_bits = 20;    % Number of fractional bits in the increment
% Filtering of RTC Increment Value
filter_rtc_inc     = 1;     % Enable moving average for RTC increment
rtc_inc_filt_len   = 128;   % RTC increment filter length
% Filtering of Delay Estimations
filter_delay_est   = 1;     % Enable moving average for delay estimations
delay_est_filt_len = 128;   % Delay estimation filter length
% Packet selection
packet_selection   = 1;     % Enable packet selection
sel_window_len     = 64;    % Selection window length

%%%%%%% Network %%%%%%%%
% Queueing statistics
queueing_mean      = 5e-6;  % Mean queueing delay (in seconds)
erlang_K           = 2;     % Erlang shape for delay pdf (see [5])

%% Constants
nRtcs              = length(Rtc);
default_rtc_inc_ns = (1/nominal_rtc_clk)*1e9;
sync_period        = 1 / sync_rate;
pdelay_req_period  = 1 / pdelay_req_rate;

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

% Observations
hScope = dsp.TimeScope(...
    'NumInputPorts', 4, ...
    'ShowGrid', 1, ...
    'ShowLegend', 1, ...
    'BufferLength', 1e5, ...
    'LayoutDimensions', [3 1], ...
    'TimeSpanOverrunAction', 'Scroll', ...
    'TimeSpan', 1e3*sync_period, ...
    'TimeUnits', 'Metric', ...
    'SampleRate', 1/sync_period);

% Customize
hScope.ActiveDisplay = 1;
hScope.Title         = 'Actual Time Offset';
hScope.YLabel        = 'Nanoseconds';
hScope.YLimits       = 1e4*[-2 2];

hScope.ActiveDisplay = 2;
hScope.Title         = 'Normalized Frequency Offset';
hScope.YLabel        = 'ppb';
hScope.YLimits       = 1e7*[-1 1];

hScope.ActiveDisplay = 3;
hScope.Title         = 'Delay Estimations (Instantaneous vs Filtered)';
hScope.YLabel        = 'Nanoseconds';
hScope.YLimits       = queueing_mean*1e9*[0 3];

%% Filters

% Moving average for the RTC increment value
if (filter_rtc_inc)
    h_rtc_inc   = (1/rtc_inc_filt_len)*ones(rtc_inc_filt_len, 1);
else
    % If disabled, force the filter to a single unitary tap
    rtc_inc_filt_len = 1;
    h_rtc_inc        = 1;
end

% Moving average for the delay estimations
if (filter_delay_est)
    h_delay_est = (1/delay_est_filt_len)*ones(delay_est_filt_len, 1);
else
    % If disabled, force the filter to a single unitary tap
    delay_est_filt_len = 1;
    h_delay_est        = 1;
end

%% Priority Queue

eventQueue = java.util.PriorityQueue;

%% Initialization

% Starting simulation time
t_sim      = 0;
prev_t_sim = 0;

for iRtc = 1:nRtcs
    % Initialize the increment counter
    Rtc(iRtc).iInc = 0;

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
i_iteration         = 0;
rtc_error_ns        = 0;
delay_est_ns        = 0;
used_delay_est_ns   = 0;
norm_freq_offset    = 0;
norm_freq_offset_to_nominal = 0;
i_rtc_inc_est       = 0;
rtc_inc_filt_taps   = zeros(rtc_inc_filt_len, 1);
i_delay_est         = 0;
delay_est_filt_taps = zeros(delay_est_filt_len, 1);
i_toffset_est       = 0;
i_sel_done          = 0;
toffset_sel_window  = repmat(struct('ns',0,'sec',0), sel_window_len, 1 );
rtc_inc_est_strobe  = 0;
toffset_corr_strobe = 0;

Sync.on_way         = 0;
Pdelay_req.on_way   = 0;
Pdelay_resp.on_way  = 0;

next_sync_tx        = 0;
next_sync_rx        = inf;
i_sync_rx_event     = 0;
next_pdelay_req_tx  = 0;
next_pdelay_req_rx  = inf;
next_pdelay_resp_rx = inf;

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
        n_incs = floor(...
            (t_sim - Rtc(iRtc).init_rising_edge_ns*1e-9) ...
            / Rtc(iRtc).clk_period );

        % Prevent negative number of increments
        if (n_incs < 0)
            n_incs = 0;
        end

        % Track the number of increments that haven't been taken into
        % account yet
        new_incs = n_incs - Rtc(iRtc).iInc;

        % Elapsed time since last update:
        elapsed_ns = new_incs * Rtc(iRtc).inc_val_ns;
        % Note: the elapsed time depends on the increment value that is
        % currently configured at the RTC. The number of increments, in
        % contrast, does not depend on the current RTC configuration.

        % Update the increment counter
        Rtc(iRtc).iInc = n_incs;

        % Update the RTC seconds count:
        if (Rtc(iRtc).ns_cnt + elapsed_ns > 1e9)
             Rtc(iRtc).sec_cnt =  Rtc(iRtc).sec_cnt + ...
                 floor((Rtc(iRtc).ns_cnt + elapsed_ns)/1e9);
        end

        % Update the RTC nanoseconds count:
        Rtc(iRtc).ns_cnt = mod(Rtc(iRtc).ns_cnt + elapsed_ns, 1e9);
        % Note: the ns count can be a fractional number, since it includes
        % the sub-nanosecond bits of the RTC increment accumulator. In
        % contrast, the timestamps added to the PTP frames are always
        % integer numbers (the integer part of these counters).
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

    %% Peer Delay Request Transmission (from slave to master)

    % Check when it is time to transmit a Pdelay_req frame
    if (t_sim >= next_pdelay_req_tx && Pdelay_req.on_way == 0)
        if (log_ptp_frames)
            fprintf('--- Event: ---\n');
            fprintf('Pdelay_req transmitted at time %g\n', t_sim);
        end

        % Timestamp the departure time from the slave side:
        Pdelay.t1.ns  = floor(Rtc(2).ns_cnt);
        Pdelay.t1.sec = floor(Rtc(2).sec_cnt);
        % Note timestamps come from the syntonized (not synchronized) RTC
        % and are also integer numbers. The "floor" approximation simulates
        % the fact that the sub-nanosecond bits are "ignored" for the
        % timestamping.

        % Mark the Pdelay_req frame as "on its way" towards the master
        Pdelay_req.on_way = 1;

        % Generate a random frame delay
        frame_delay = sum(exprnd(queueing_mean/erlang_K, 1, erlang_K));

        % Schedule the event for Pdelay_req reception by the master
        next_pdelay_req_rx = t_sim + frame_delay;
        eventQueue.add(next_pdelay_req_rx);

        % Schedule the next Pdelay_req transmission
        next_pdelay_req_tx = next_pdelay_req_tx + pdelay_req_period;
        eventQueue.add(next_pdelay_req_tx);
    end

    %% Peer Delay Request Reception

    % Process the Pdelay_req frame received by the master
    if (t_sim >= next_pdelay_req_rx && Pdelay_req.on_way)

        % Clear "on way" status
        Pdelay_req.on_way = 0;

        % Timestamp the arrival time (t2) at the master side:
        Pdelay.t2.ns = floor(Rtc(1).ns_cnt);
        Pdelay.t2.sec = floor(Rtc(1).sec_cnt);
        % Note timestamps come from the syntonized (not synchronized) RTC
        % and are also integer numbers.

        if (log_ptp_frames)
            fprintf('--- Event: ---\n');
            fprintf('Pdelay_req received at time %g\n', t_sim);
        end

        % Now a Pdelay_resp must be sent back to the slave

        % Mark the Pdelay_resp frame as "on its way" towards the slave
        Pdelay_resp.on_way = 1;

        % Timestamp the response departure time (t3) at the master side:
        Pdelay.t3.ns  = floor(Rtc(1).ns_cnt);
        Pdelay.t3.sec = floor(Rtc(1).sec_cnt);
        % Note timestamps come from the syntonized (not synchronized) RTC
        % and are also integer numbers.

        % Generate a random frame delay
        frame_delay = sum(exprnd(queueing_mean/erlang_K, 1, erlang_K));

        % Schedule the event for Pdelay_resp reception
        next_pdelay_resp_rx = t_sim + frame_delay;
        eventQueue.add(next_pdelay_resp_rx);
    end

    %% Peer Delay Response Reception

    % Process the Pdelay_resp frame received by the requestor (slave)
    if (t_sim >= next_pdelay_resp_rx && Pdelay_resp.on_way)

        % Clear "on way" status
        Pdelay_resp.on_way = 0;

        % Timestamp the response arrival time (t4) at the slave side:
        Pdelay.t4.ns  = floor(Rtc(2).ns_cnt);
        Pdelay.t4.sec = floor(Rtc(2).sec_cnt);
        % Note timestamps come from the syntonized (not synchronized) RTC
        % and are also integer numbers.

        if (log_ptp_frames)
            fprintf('--- Event: ---\n');
            fprintf('Pdelay_resp received at time %g\n', t_sim);
        end

        %% Delay Estimation

        t4_minus_t1 = Pdelay.t4.ns - Pdelay.t1.ns;
        % If the ns counter wraps, this difference would become negative.
        % In this case, add one second back:
        if (t4_minus_t1 < 0)
            t4_minus_t1 = t4_minus_t1 + 1e9;
        end

        t3_minus_t2 = Pdelay.t3.ns - Pdelay.t2.ns;
        % If the ns counter wraps, this difference would become negative.
        % In this case, add one second back:
        if (t3_minus_t2 < 0)
            t3_minus_t2 = t3_minus_t2 + 1e9;
        end

        % One-way delay estimation (in ns):
        delay_est_ns = (t4_minus_t1 - t3_minus_t2) / 2;

        %% Filter the delay estimation
        delay_est_filt_taps = [delay_est_ns; delay_est_filt_taps(1:end-1)];
        filtered_delay_est  = (delay_est_filt_taps.') * h_delay_est;

        % Count how many increment value estimations so far
        i_delay_est = i_delay_est + 1;

        % Use the filtered RTC increment after the filter transitory
        if (i_delay_est >= delay_est_filt_len)
            used_delay_est_ns = filtered_delay_est;
        else
            used_delay_est_ns = delay_est_ns;
        end
    end

    %% SYNC Transmission

    % Check when it is time to transmit a SYNC frame
    if (t_sim >= next_sync_tx && Sync.on_way == 0)

        if (log_ptp_frames)
            fprintf('--- Event: ---\n');
            fprintf('Sync transmitted at time %g\n', t_sim);
        end

        % Timestamp the departure time:
        Sync.t1.ns  = floor(Rtc(1).ns_cnt);
        Sync.t1.sec = floor(Rtc(1).sec_cnt);
        % Note timestamps come from the syntonized (not synchronized) RTC
        % and are also integer numbers.

        % Mark the SYNC frame as "on its way" towards the slave
        Sync.on_way = 1;

        % Generate a random frame delay
        frame_delay = sum(exprnd(queueing_mean/erlang_K, 1, erlang_K));
        % Save the true delay within the message for use in case perfect
        % delay estimation is enabled in the simulation(for debugging):
        Sync.delay = frame_delay;

        % Schedule the event for sync reception
        next_sync_rx = t_sim + frame_delay;
        eventQueue.add(next_sync_rx);

        % Schedule the next SYNC transmission
        next_sync_tx = next_sync_tx + sync_period;
        eventQueue.add(next_sync_tx);
    end

    %% SYNC Reception

    % Process the SYNC frame at the destination
    if (t_sim >= next_sync_rx && Sync.on_way)

        % Clear "on way" status
        Sync.on_way = 0;

        %% Process SYNC timestamps

        % Timestamp the arrival time (t2) at the slave side:
        Sync.t2.ns  = floor(Rtc(2).ns_cnt);
        Sync.t2.sec = floor(Rtc(2).sec_cnt);
        % Note timestamps come from the syntonized (not synchronized) RTC
        % and are also integer numbers.

        % First save the previous time offset estimation:
        prev_rtc_error_ns = rtc_error_ns;

        % Increment time offset number
        i_sync_rx_event = i_sync_rx_event + 1;

        if (log_ptp_frames)
            fprintf('--- Event: ---\n');
            fprintf('Sync[%d] received at time %g\n', ...
                i_sync_rx_event, t_sim);
        end

        % Delay to correct the master timestamp:
        if (perfect_delay_est)
            sync_route_delay_ns = Sync.delay*1e9;
        else
            sync_route_delay_ns = used_delay_est_ns;
        end

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

        %% Time offset estimation

        % RTC error:
        Rtc_error.ns = master_ns_sync_rx - slave_ns_sync_rx;
        % Note: it is actually computed as "(t1 + d) - t2" here.
        Rtc_error.sec = master_sec_sync_rx - slave_sec_sync_rx;

        %% "Packet selection" for time offset estimation

        if (packet_selection)
            % Count the number of time offset estimations accumulated so
            % far:
            i_toffset_est = i_toffset_est + 1;

            % Add the new estimation to the selection window:
            toffset_sel_window(i_toffset_est) = Rtc_error;

            % Trigger a time offset correction when the window is full
            toffset_corr_strobe = (i_toffset_est == sel_window_len);

            % When the selection window is full, proceed with the time
            % offset selection.
            if (toffset_corr_strobe)
                % Reset count
                i_toffset_est = 0;

                % "Select" (compute) a ns and a sec time offset estimation

                % First compute the mean of the seconds
                mean_sec      = mean(cat(1,toffset_sel_window.sec));
                % The seconds error in the RTC is the integer part of that:
                Rtc_error.sec = round(mean_sec);
                % The ns error, in turn, is the mean of the ns time offsets
                % in the selection window + the fractional part of the mean
                % sec time offset:
                Rtc_error.ns  = ...
                    round(mean(cat(1,toffset_sel_window.ns))) + ...
                    round((mean_sec - round(mean_sec)) * 1e9);
                % Important to remember: the resulting RTC error depends on
                % the original time offset from when the system started and
                % the changes in time offset that are accumulated when the
                % RTC increment is changed, but are never changed by time
                % offset corrections themselves.

                % Use the selected time offset estimation to compute and
                % replace the slave-side timestamp that is used for
                % estimating the RTC increment value. By doing so, the
                % instant when the packet selection is concluded could be
                % interpreted as the actual "reception" of a SYNC (with
                % slower rate) that yields a better estimation.
                slave_ns_sync_rx = master_ns_sync_rx - Rtc_error.ns;
                slave_sec_sync_rx = master_sec_sync_rx - Rtc_error.sec;
                % Note:
                % master time - (master time + slave time) = slave time

                % Keep track of how many selections were already concludedy
                i_sel_done = i_sel_done + 1;

                % And use this count to trigger RTC increment estimations
                if (i_sel_done == rtc_inc_est_period)
                    rtc_inc_est_strobe = 1;

                    % Reset the count
                    i_sel_done = 0;
                end
            end
        else
            % When packet selection is not used, the strobe to trigger the
            % RTC increment estimation occurs after every
            % "rtc_inc_est_period".
            rtc_inc_est_strobe = (i_sync_rx_event == rtc_inc_est_period);

            % In this case, time offsets are corrected after every Rx SYNC,
            % so the strobe signal is always asserted
            toffset_corr_strobe = 1;
        end

        %% Time Offset Correction
        % Note: the error is saved on the time offset registers, and never
        % corrected in the actual ns/sec count. The two informations are
        % always separately available and can be summed together to form a
        % synchronized (time aligned) ns/sec count.

        if (toffset_corr_strobe)
            % First ensure that the nanoseconds error is not negative. It
            % has to be a number between 0 and 1e9. Any negative ns offset
            % can be corrected by a positive ns offset plus a correction
            % within the seconds count.
            if (Rtc_error.ns < 0)
                Rtc_error.ns = Rtc_error.ns + 1e9;
                Rtc_error.sec = Rtc_error.sec - 1;
            end

            % Update the RTC time offset registers
            Rtc(2).time_offset.ns = Rtc_error.ns;
            Rtc(2).time_offset.sec = Rtc_error.sec;
        end

        %% Frequency Offset Estimation and RTC increment value computation

        % Check if the desirable number of SYNC receptions was already
        % reached for estimating the frequency offset
        if (rtc_inc_est_strobe)

            % Clear the strobe signal
            rtc_inc_est_strobe = 0;

            % Reset SYNC event counter
            i_sync_rx_event = 0;

            % Duration in ns at the master side between the two SYNCs:
            master_sync_interval_ns = ...
                (master_ns_sync_rx + 1e9*master_sec_sync_rx) - ...
                (prev_master_ns_sync_rx + 1e9*prev_master_sec_sync_rx);
            % Note #1: In practice, this interval could be known a priori.
            % However, since a standardized PTP master does not need to
            % inform the SYNC rate to the slave, a generic implementation
            % would measure the sync interval.
            %
            % Note #2: normally we can measure only the nanosecond
            % difference and infer any wrapping by checking whether the
            % difference is negative. In this case, however, when packet
            % selection is adopted, it is possible to have effective SYNC
            % intervals of more than 1 second. For example, if the SYNC
            % rate is 128 packet-per-second and the packet selection length
            % is 256, there is one selected time offset for each 2 seconds
            % and one RTC increment est after 4 seconds. Thus, the
            % computation has to use the full "unwrapped" nanosecond
            % counts.

            % Check a negative duration, which can happen whenever the ns
            % counter (used for the timestampts) wraps:
            if (master_sync_interval_ns < 0)
                master_sync_interval_ns = master_sync_interval_ns + 1e9;
            end

            % Duration at the slave side between the two SYNC frames:
            slave_sync_interval_ns = ...
                (slave_ns_sync_rx + 1e9*slave_sec_sync_rx) - ...
                (prev_slave_ns_sync_rx + 1e9*prev_slave_sec_sync_rx);
            % Here, again, the unwrapped ns count is used due to the fact
            % that the intervals can be larger than one second.

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

            if (norm_freq_offset*1e9 > foffset_thresh_ppb)
                warning('Frequency offset estimation exceed the maximum');
                norm_freq_offset = 0;
            end

            % Compute the new increment value for the slave RTC:
            %
            % Note that, in contrast to time offsets, once an increment
            % value is corrected, the frequency offset that is going to be
            % "seen" next is not the true frequency offset anymore (recall
            % that since syntonized timestamps are used, the same true time
            % offset could be estimated forever if everything remained
            % stable). Instead, the frequency offset estimation is expected
            % to change from something near the true frequency offset (for
            % the first estimation/correction) to something near zero (for
            % the subsequent estimations). This is because the computation
            % is such that the estimated frequency offset reflects the
            % offset remaining from the current increment value, instead of
            % the nominal increment value.

            % First infer the current estimation for the clock frequency
            % that feeds the slave RTC, from the current increment value:
            current_slave_clk_freq_est = (1/Rtc(2).inc_val_ns)*1e9;
            % Then, update the estimation by applying the frequency offset
            % that was just estimated:
            new_slave_est_clk_freq = (1 + norm_freq_offset) * ...
                current_slave_clk_freq_est;
            % And derive the corresponding new increment value:
            new_rtc_inc        = (1 / new_slave_est_clk_freq)*1e9;

            % If simulation of increment as a fixed-point number is
            % enabled, quantize the increment value:
            if (en_fp_inc_val)
                % Use an unsigned fixed-point number for the RTC increment
                % with limited number of fractional (subnanoseconds bits).
                new_rtc_inc_fp = fi(new_rtc_inc, 0, ...
                    n_inc_val_int_bits, n_inc_val_frc_bits);

                % Convert back to double
                new_rtc_inc = double(new_rtc_inc_fp);

                % And update the resulting "new" slave clock frequency
                % considering the fixed-point precision:
                new_slave_est_clk_freq = (1/new_rtc_inc)*1e9;
            end


            % Compute the "new" frequency offset that is going to be
            % present after the RTC is updated:
            new_freq_offset = new_slave_est_clk_freq - nominal_rtc_clk;

            % And also compute a normalized frequency offset estimation
            % relative to the nominal frequency value (instead of the
            % current configuration):
            norm_freq_offset_to_nominal = new_freq_offset/nominal_rtc_clk;
            % Note: norm_freq_offset, in contrast, always estimates the
            % frequency offset with respect to the current RTC increment
            % configuration (not the nominal).

            %% Filter the increment value
            rtc_inc_filt_taps = [new_rtc_inc; rtc_inc_filt_taps(1:end-1)];
            filtered_rtc_inc  = (rtc_inc_filt_taps.') * h_rtc_inc;

            % Count how many increment value estimations so far
            i_rtc_inc_est = i_rtc_inc_est + 1;

            % Use the filtered RTC increment after the filter transitory
            if (i_rtc_inc_est >= rtc_inc_filt_len)
                Rtc(2).inc_val_ns = filtered_rtc_inc;
            else
                Rtc(2).inc_val_ns = new_rtc_inc;
            end

            %% Print Frequency Offset Estimation
            if (print_freq_offset_est)
                fprintf(...
                'Estimated FreqOffset:\t%6g ppb\t NewInc:\t%.20f ns\n', ...
                norm_freq_offset*1e9, new_rtc_inc);
            end
        end

        % Save the sync arrival timestamps for the next iteration. These
        % timestamps can be the actual timestamps carried along the SYNC
        % frames or the ones including adjustments after packet selection:
        if (toffset_corr_strobe)
            prev_master_ns_sync_rx  = master_ns_sync_rx;
            prev_master_sec_sync_rx = master_sec_sync_rx;
            prev_slave_ns_sync_rx   = slave_ns_sync_rx;
            prev_slave_sec_sync_rx  = slave_sec_sync_rx;
        end

        %% Synchronized RTC Values
        % Synchronized RTC = Syntonized RTC + Offset

        % Master
        master_rtc_sync_ns  = Rtc(1).ns_cnt  + Rtc(1).time_offset.ns;
        master_rtc_sync_sec = Rtc(1).sec_cnt + Rtc(1).time_offset.sec;
        % Check for ns wrap:
        if (master_rtc_sync_ns >= 1e9)
            master_rtc_sync_ns  = master_rtc_sync_ns - 1e9;
            master_rtc_sync_sec = master_rtc_sync_sec + 1;
        end

        % Slave
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
            step(hScope, actual_ns_error, ...
                norm_freq_offset_to_nominal*1e9, ...
                delay_est_ns, used_delay_est_ns);
        end

        if (print_true_time_offsets)
            fprintf('True TOffset\t%g ns\n', ...
                actual_ns_error);
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


