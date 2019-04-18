function [ w, i ] = resetSelWindow( w_len )
% Reset the packet selection window
%
% w_len -> window length (in samples)
% w     -> preallocated window vector
% i     -> current pointer to the window

% Reallocate the selection window:
w  = repmat(struct('ns',0,'sec',0), w_len, 1 );

% Reset window index:
i = 0;

end

