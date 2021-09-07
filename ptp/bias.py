"""Bias Compensator
"""
import logging

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class Bias():
    """Bias Compensator

    The PTP timestamp differences can be modeled as:

        t2[n] - t1[n] = x[n] + d_ms[n]
        t3[n] - t4[n] = x[n] - d_sm[n]

    By summing the two equations and dividing by two, one can obtain the true
    time offset as:

        ((t2[n] - t1[n]) + (t3[n] - t4[n]))/2 = x[n] + (d_ms[n] - d_sm[n])/2

    This is equivalent to:

        x_est[n] = x[n] + (d_ms[n] - d_sm[n])/2,

    since the slave's time offset measurement is computed by:

        x_est[n] = ((t2[n] - t1[n]) - (t4[n] - t3[n])) / 2

    This means that the time offset measurements are noisy:

        x_est[n] = x[n] + w[n],

    where the noise term is:

        w[n] = (d_ms[n] - d_sm[n])/2,

    which is also known as the "delay asymmetry".

    The asymmetry w[n] typically has a non-zero mean, since the distributions
    of the master-to-slave (m-to-s) and slave-to-master (s-to-m) delays usually
    have distinct mean and differ also on other statistics (like minimum value,
    maximum, mode, etc).

    This class provides mechanisms for calculating and compensating w[n]. While
    doing so, it also considers the post processing that is to be applied on
    x_est[n]. For example, if sample-minimum packet selection is applied on a
    window of x_est[n], the value that compensates w[n] differs from the case
    where x_est[n] measurements are used directly.

    """
    def __init__(self, data):
        """Initialize bias compensator

        Args:
            data : Array of objects with simulation data

        """
        self.data = data

    def calc_true_asymmetry(self, target='estimates', metric='avg'):
        """Calculate the delay asymmetry of interest for bias compensation

        Processes the true m-to-s and s-to-m delays in the dataset in order to
        compute the delay asymmetry of interest. For example, the asymmetry
        between the minimum m-to-s and the minimum s-to-m delays.

        In practice, a PTP slave can only compute such values if locked to
        another accurate time source, such as GNSS. The slave needs to measure
        the true m-to-s and s-to-m delays of PTP messages to compute the true
        asymmetries that are computed in this function.

        Args:

            target   : Whether to calculate the asymmetry to compensate bias of
                       time offset estimates or to compensate t4 timestamps.
                       Valid options are ["timestamps", "estimates"]
                       (default: "estimates").

            metric   : When calculating the true bias of estimates, define the
                       asymmetry metric of interest. Select among 'avg', 'min',
                       'max', 'median', or 'mode' (default: 'avg').

        Returns:
            The true delay asymmetry of interest

        """
        d_ms = np.array([r["d"] for r in self.data])
        d_sm = np.array([r["d_bw"] for r in self.data])

        # Correction value for timestamps
        if (target == 'timestamps'):
            corr = np.mean(d_ms - d_sm)

        # Correction value for post-processed or raw time offset estimates
        elif (target == 'estimates'):
            if (metric == 'avg'):
                corr = (np.mean(d_ms) - np.mean(d_sm)) / 2
            elif (metric == 'min'):
                corr = (np.amin(d_ms) - np.amin(d_sm)) / 2
            elif (metric == 'max'):
                corr = (np.amax(d_ms) - np.amax(d_sm)) / 2
            elif (metric == 'median'):
                corr = (np.median(d_ms) - np.median(d_sm)) / 2
            elif (metric == 'mode'):
                d_ms_mode, _ = stats.mode(np.round(d_ms))
                d_sm_mode, _ = stats.mode(np.round(d_sm))
                corr = (d_ms_mode[0] - d_sm_mode[0]) / 2
            else:
                raise ValueError('Unknown metric {}'.format(metric))
        else:
            raise ValueError("Unknown target")

        return corr

    def compensate(self, corr, target='estimates', toffset_key='x_est'):
        """Compensate the bias of time offset estimations due to delay asymmetry

        In order to correct the asymmetry, a correction value can be either
        added directly to the raw time offset measurements ('x_est') or added
        to one of the four timestamps of a two-way exchange. For timestamp
        compensation, this function adds a correction to timestamp 't4'. The
        rationale is elaborated in the sequel.

        1) Correcting time offset estimates

        When compensating time offset estimates/measurements, the goal is
        either to compensate the raw w[n] (see the expression above) or the
        bias that arises by processing m-to-s and s-to-m timestamp differences
        through packet selection. For example, when raw time offset
        measurements x_est[n] are considered, the compensation value should be
        the mean of w[n], like so:

        x_est_corr[n] = x_est[n] - mean{w[n]}
                     ~= x[n] + (w[n] - mean{w[n]}).

        The resulting corrected time offset measurements will then be unbiased
        as desired.

        On the other hand, if e.g. sample minimum is used, the bias that arises
        comes from the sample-minimum output:

        samp_min[n] = (min{x[n] + d_ms[n]} + min{x[n] - d_sm[n]}) / 2
                   ~= x[n] + (min{d_ms[n]} - min{d_sm[n]})/2
                   ~= x[n] + w_sm[n],

        where

        w_sm[n] = (min{d_ms[n]} - min{d_sm[n]})/2

        Thus, the asymmetry to be compensated is "w_sm[n]" in this case.
        Similar expressions can be derived by considering other packet
        selection operators.

        2) Correcting timestamps

        The idea of correcting timestamps is that it can be done before any
        post-processing. In practice, this would be done by the PTP slave
        itself directly, in which case the slave would correct a known static
        asymmetry. Ultimately, the timestamps delivered for post-processing
        stages would already be compensated and ideally symmetric on average.

        When timestamps are compensated, it suffices to adjust t4[n]. By
        considering that:

        t4_corr[n] = t4[n] - corr[n]

        The expression for timestamp differences becomes:

        t2[n] - t1[n]             = x[n] + d_ms[n]
        t3[n] - (t4[n] - corr[n]) = x[n] - d_sm[n]

        which can be arranged to:

        t2[n] - t1[n] = x[n] + d_ms[n]
        t3[n] - t4[n] = x[n] - (d_sm[n] + corr[n])

        This means that, to ensure symmetry, we need:

        d_ms[n] = (d_sm[n] + corr[n])

        So that:

        corr[n] = d_ms[n] - d_sm[n]

        In practice, the correction value corr[n] must be a constant that
        reflects the static asymmetry. Hence, the constant correction is:

        corr = mean{d_ms[n] - d_sm[n]}

        Note, however, that this correction won't be sufficient in case packet
        selection is later applied. This is because some selection operators
        suffer from asymmetries other than the mean asymmetry. For example, it
        is perfectly possible that on average, after compensation, the m-to-s
        and s-to-m delays become symmetric, while their
        maximum/minimum/mode/median remain asymmetric. Furthermore, the
        compensation applied to t4 can both help or worsen the asymmetry of the
        other statistics. Not necessarily it will help. Nevertheless, note that
        some post-processing strategies will benefit from compensation of
        timestamps: sample-average, EWMA and LS will all benefit.

        Args:

            corr        : Correction value to be applied.

            target      : Whether to apply bias compensation to time offset
                          estimates (post-processed or not) or to timestamps,
                          specifically by adjusting 't4'. Valid options are
                          ["timestamps", "estimates"] (default: "estimates").

            toffset_key : Time offset estimation key

        """
        # Correction for timestamps
        if (target == 'timestamps'):
            logger.info("Compensating bias on timestamps")
            logger.info("Set t4 -= %f ns" % (corr))

            for d in self.data:
                d["t4"] -= corr

        # Correction for post-processed or raw time offset estimates
        elif (target == 'estimates'):
            logger.info("Compensating bias on %s" % (toffset_key))
            logger.info("Set %s -= %f ns" % (toffset_key, corr))

            for d in self.data:
                if (toffset_key in d):
                    d[toffset_key] -= corr
        else:
            raise ValueError("Unknown target")
