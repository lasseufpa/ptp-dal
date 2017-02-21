% RTC simulator
clearvars, clc

%% Parameters and Configurations

nominal_rtc_clk = 125e6;
t_step_sim      = 1e-9;            % Simulation Time Step (resolution)
scope_time_res  = 100*t_step_sim;

% Master RTC
rtc(1).freq_offset_ppb = 0;
rtc(1).init_time_offset_ns = 0;

% Slave RTC
rtc(2).freq_offset_ppb = 200e3;
rtc(2).init_time_offset_ns = 5000;

% PTP
sync_rate       = 128;     % Rate in frames per second
% Queueing statistics
queueingMean    = 5e-6;    % Mean queueing delay
erlangK         = 2;       % Erlang shape for delay pdf (see [5])

%% Constants
nRtcs = length(rtc);
clk8k_period = 1 / 8e3;
clk8k_period_ns = clk8k_period*1e9;
default_rtc_inc_ns = (1/nominal_rtc_clk)*1e9;
sync_period =  1 / sync_rate;

%% Derivations

% Derive the fixed clock frequency (in Hz) for each RTC. This corresponds
% to the frequency of the clock that drives the RTC, which is assumed to be
% fixed (not disciplined and not drifting) over the simulation time.
for iRtc = 1:nRtcs
    % Frequency offset in Hz
    freq_offset_Hz = rtc(iRtc).freq_offset_ppb * (nominal_rtc_clk/1e9);

    % Corresponding clock frequency (in Hz)
    rtc(iRtc).clk_freq = nominal_rtc_clk + freq_offset_Hz;

    % Clock period
    rtc(iRtc).clk_period = (1/rtc(iRtc).clk_freq);
end

%% System Objects

hScope = dsp.TimeScope(...
    'Title', 'Clk8k', ...
    'NumInputPorts', nRtcs, ...
    'ShowGrid', 1, ...
    'ShowLegend', 1, ...
    'BufferLength', 1e5, ...
    'TimeSpanOverrunAction', 'Wrap', ...
    'TimeSpan', 2*clk8k_period, ...
    'TimeUnits', 'Metric', ...
    'SampleRate', 1/scope_time_res, ...
    'YLimits', [-5 5]);

%% Priority Queue

eventQueue = java.util.PriorityQueue;

%% Initialization

% Starting simulation time
t_sim = 0;

% Scope plots
next_scope_tick = 0;

for iRtc = 1:nRtcs
    % Schedule the instant for the next RTC increment:
    rtc(iRtc).next_incr_inst =  rtc(iRtc).clk_period;

    % Schedule the next increment event:
    eventQueue.add(rtc(iRtc).next_incr_inst);

    % Initialize Seconds RTC count
    rtc(iRtc).sec_cnt = 0;

    % Initialize Nanoseconds RTC counts
    rtc(iRtc).ns_cnt = rtc(iRtc).init_time_offset_ns;

    % Initialize the RTC Time Offset Register
    rtc(iRtc).time_offset = 0;

    % Initialize the increment value:
    rtc(iRtc).inc_val_ns = default_rtc_inc_ns;

    % Value for the clk8k output:
    if (rtc(iRtc).ns_cnt == 0)
        rtc(iRtc).clk8k = 1;
    else
        rtc(iRtc).clk8k = 0;
    end

    % Keep track of the clk8k tick instants
    rtc(iRtc).prev_clk8k_tick_inst = 0;
end

sync_frame.on_way = 0;
iSync             = 0;
iTOffsetEst       = 0;
next_sync_tx      = 0;

%% Infinite Loop
while (1)

    %% RTC Increments
    for iRtc = 1:nRtcs
        if (t_sim > rtc(iRtc).next_incr_inst)
            % Increment
            rtc(iRtc).ns_cnt = rtc(iRtc).ns_cnt + rtc(iRtc).inc_val_ns;

            % Schedule the next RTC increment instant
            rtc(iRtc).next_incr_inst = rtc(iRtc).next_incr_inst + ...
                rtc(iRtc).clk_period;

            % Schedule the next increment event:
            eventQueue.add(rtc(iRtc).next_incr_inst);

            % Wrap ns counter after every full second and increment secs
            if (rtc(iRtc).ns_cnt > 1e9)
                rtc(iRtc).ns_cnt = 0;
                rtc(iRtc).sec_cnt = rtc(iRtc).sec_cnt + 1;
            end
        end
    end

    %% Clk8k Derivation

    for iRtc = 1:nRtcs
        % Check the time since the previous assertion in the clk8k
        elapsed_time_prev_tick = rtc(iRtc).ns_cnt - ...
            rtc(iRtc).prev_clk8k_tick_inst;

        % Assert the clk8k output every "clk8k_period_ns" nanoseconds and
        % clear the pulse after half period for 50% duty cycle
        if (elapsed_time_prev_tick > clk8k_period_ns)
            rtc(iRtc).clk8k = 1;
            rtc(iRtc).prev_clk8k_tick_inst = rtc(iRtc).ns_cnt;

            % Schedule the next rising edge event:

        elseif (elapsed_time_prev_tick > (clk8k_period_ns/2))
            rtc(iRtc).clk8k = 0;

            % Schedule the next falling edge event:
        end
    end

    %% SYNC Transmissions

    % Check when it is time to transmit a SYNC frame
    if (t_sim >= next_sync_tx)
        % Check
        while (sync_frame(iSync).on_way)
            iSync = iSync + 1;
        end

        % Departure time
        sync_frame(iSync).t1 = rtc(1).ns_cnt + rtc(1).time_offset;
        % Note: the synchronized time is used

        % Mark the sync frame as "on its way"
        sync_frame(iSync).on_way = 1;

        % Generate a random queueing delay
        queuingDelay = sum(exprnd(queueingMean/erlangK, 1, erlangK), 2);

        % Schedule the event for sync reception
        next_sync_rx = t_sim + queuingDelay;
        eventQueue.add(next_sync_rx);

        % Schedule the next SYNC transmission
        next_sync_tx = next_sync_tx + sync_period;
        eventQueue.add(next_sync_tx);
    end

    % Process the SYNC frame at the destination
    if (iSync > 0 && sync_frame(iSync).on_way && (t_sim >= next_sync_rx))
        % Clear "on way" status
        sync_frame(iSync).on_way = 0;

        % Timestamp the arrival time (t2) at the slave side:
        sync_frame(iSync).t2 = rtc(2).ns_cnt + rtc(2).time_offset;
        % Note: the synchronized time is used

        % First save the previous time offset estimation:
        prev_t_offset = t_offset;

        % Estimate a new Time offset
        iTOffsetEst = iTOffsetEst + 1;
        t_offset = sync_frame(iSync).t2 - (sync_frame(iSync).t1 + queueingMean);

        % Update the RTC time offset
        rtc(2).time_offset = t_offset;

        % Estimate Frequency offset after every 2 time offset estimations
        if (mod(iTOffsetEst, 2) == 0)
            freq_offset = (t_offset - prev_t_offset) / sync_period;

            % Compute the new frequency
            rtc(iRtc).inc_val_ns = 1 / (nominal_rtc_clk + freq_offset)
        end
    end


    %% Plot to scope
    if (t_sim > next_scope_tick)
        step(hScope, rtc(1).clk8k + 2, rtc(2).clk8k);

        next_scope_tick = next_scope_tick + scope_time_res;
    end

    %% Increase simulation step

    % Check the time elapsed since the previous event
%     elapsed_since_prev_event = t_sim - next_event;

    % Check the next scheduled event
    next_event = eventQueue.poll();

    if (length(next_event) == 0)
        t_sim = t_sim + t_step_sim;
        warning('Empty next event');
    else
        % Jump simulation time to the next event
        t_sim = next_event;
    end
end


