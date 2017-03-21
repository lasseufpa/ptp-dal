function [ stage, w_len, w, i ] = changeSyncStage( Stage, target_stage )
% Change the synchronization stage
%
% stage -> Sync Stage after switching
% w_len -> window length (in samples)
% w     -> preallocated window vector
% i     -> current pointer to the window

% Print the target synchronization stage:
fprintf('Entering synchronization stage #%d\n', target_stage);

% Get the selection window parameters:
switch (target_stage)
    case 1
        w_len = Stage.stage1.sel_window_len;
    case 2
        w_len = Stage.stage2.sel_window_len;
    case 3
        w_len = Stage.stage3.sel_window_len;
    case 4
        w_len = Stage.stage4.sel_window_len;
    otherwise
        error('Synchronization stage does not exist');
end

% Reset the selection window
[ w, i ] = resetSelWindow( w_len );

% Output the stage
stage = target_stage;

end

