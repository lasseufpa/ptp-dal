"""EWMA implementation
"""


class Ewma():
    def __init__(self, beta=0, bias_corr=True):
        """Exponentially weighted moving average

        Args:
            beta      : EWMA beta weight
            bias_corr : whether to use bias correction

        """

        self.avg = 0
        self.n = 0
        self.alpha = 1 - beta
        self.beta = beta
        self.bias_corr = bias_corr

    def set_equivalent_window(self, N):
        """Define alpha and beta based on the equivalent window length

        Due to exponentially decaying weights, there is an equivalent window
        for a given alpha/beta setting. A good approximation is that the window
        encompasses 1/alpha samples.

        Args:
            N : equivalent average window length

        """
        self.alpha = 1 / N
        self.beta = 1 - (1 / N)

    def reset(self):
        """Reset average and state"""
        self.avg = 0
        self.n = 0

    def step(self, obs):
        """Calculate the exponentially weighted moving average (EWMA)

        Args:
            obs : New observation

        Returns:
            The exponentially weighted moving average, bias-corrected if bias
            correction is enabled

        """
        new_avg = self.avg = (self.beta * self.avg) + self.alpha * obs

        if (not self.bias_corr):
            return new_avg

        # Apply bias correction (but don't save the bias-corrected average)
        self.n += 1
        bias_corr = 1 / (1 - (self.beta**self.n))
        corr_avg = new_avg * bias_corr
        return corr_avg
