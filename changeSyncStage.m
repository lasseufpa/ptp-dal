function [ stage, sel_s, w_len, w, i ] = ...
    changeSyncStage( Stage, target_stage )
% Change the synchronization stage
%
% stage -> Sync Stage after switching
% sel_s -> Time offset "selection" strategy
% w_len -> window length (in samples)
% w     -> preallocated window vector
% i     -> current pointer to the window

% Print the target synchronization stage:
fprintf('==============================================\n');
fprintf('Entering synchronization stage #%d\n', target_stage);

strategies_str = cell(2, 1);
strategies_str{1} = 'sample-mean';
strategies_str{2} = 'least-squares';

% Get the selection window parameters:
switch (target_stage)
    case 1
        w_len = Stage.stage1.sel_window_len;
        sel_s = Stage.stage1.sel_strategy;
    case 2
        w_len = Stage.stage2.sel_window_len;
        sel_s = Stage.stage2.sel_strategy;
    case 3
        w_len = Stage.stage3.sel_window_len;
        sel_s = Stage.stage3.sel_strategy;
    case 4
        w_len = Stage.stage4.sel_window_len;
        sel_s = Stage.stage4.sel_strategy;
    otherwise
        error('Synchronization stage does not exist');
end

% Print the configuration for this next sync stage:
fprintf('------------ Stage configuration ------------\n');
fprintf('Selection strategy:\t %s\n', strategies_str{sel_s + 1});
fprintf('Window length:\t %d\n', w_len);
fprintf('==============================================\n');

% Reset the selection window
[ w, i ] = resetSelWindow( w_len );

% Output the stage
stage = target_stage;

end

