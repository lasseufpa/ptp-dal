"""Recursive/moving filter operations"""
import numpy as np
import ptp.ewma


def ewma(N, x_array):
    """Exponentially weighted moving average (EWMA)

    Uses the Ewma class from the ewma.py module to implement an EWMA filter. The
    coefficients are set based on a target window length.

    Args:
        N       : Window length
        x_array : (numpy.ndarray) data array

    Returns:
        (numpy.ndarray) Array with the average computed at each observation
        window over the input data array. For an input array of size S, the
        result array has size "S -N + 1".

    """
    res = np.zeros(len(x_array))

    ewma_filt = ptp.ewma.Ewma()
    ewma_filt.set_equivalent_window(N)

    res = [ewma_filt.step(x) for x in x_array]

    return res[N-1:]


def moving_average(N, x_array):
    """Moving-average

    Uses an FIR filter to implement the moving average. The operation is
    equivalent to sliding an observation window of length N over a given data
    array and computing the average for each window.

    Args:
        N       : Window length
        x_array : (numpy.ndarray) data array

    Returns:
        (numpy.ndarray) Array with the average computed at each observation
        window over the input data array. For an input array of size S, the
        result array has size "S -N + 1".

    """
    assert(isinstance(x_array, np.ndarray))
    h = np.ones(N) / N
    return np.convolve(x_array, h, mode="valid")


def moving_minimum(N, x_array):
    """Moving-minimum

    Slides a window of length N over a given data array and re-computes the
    minimum only when necessary.

    Args:
        N       : Window length
        x_array : (numpy.ndarray) data array

    Returns:
        (numpy.ndarray) Array with minimum computed at each observation
        window over the input data array. For an input array of size S, the
        result array has size "S -N + 1".

    """
    assert(isinstance(x_array, np.ndarray))
    i_head = N                  # start on the first full window
    _min = np.amin(x_array[:N]) # starting minimium

    # Preallocate result array and set the first value
    res    = np.zeros(len(x_array) -N + 1)
    res[0] = _min

    for x in x_array[N:]:
        # Window's tail pointer
        i_tail = i_head - N + 1

        # If the value that was just thrown away (at the previous tail index)
        # was the actual minimum, recompute the minimum now.
        if (_min == x_array[i_tail - 1]):
            _min = np.amin(x_array[i_tail:(i_head + 1)])
        # If the new value entering the window now is lower than the current
        # minimum, update the minimum.
        elif (x < _min):
            _min = x

        # Save minimum on results array
        res[i_tail] = _min

        # Advance the head pointer
        i_head += 1

    return res


def moving_maximum(N, x_array):
    """Moving-maximum

    Slides a window of length N over a given data array and re-computes the
    maximum only when necessary.

    Args:
        N       : Window length
        x_array : (numpy.ndarray) data array

    Returns:
        (numpy.ndarray) Array with maximum computed at each observation
        window over the input data array. For an input array of size S, the
        result array has size "S -N + 1".

    """
    assert(isinstance(x_array, np.ndarray))
    i_head = N                  # start on the first full window
    _max = np.amax(x_array[:N]) # starting maximum

    # Preallocate result array and set the first value
    res    = np.zeros(len(x_array) -N + 1)
    res[0] = _max

    for x in x_array[N:]:
        # Window's tail pointer
        i_tail = i_head - N + 1

        # If the value that was just thrown away (at the previous tail index)
        # was the actual maximum, recompute the maximum now.
        if (_max == x_array[i_tail - 1]):
            _max = np.amax(x_array[i_tail:(i_head + 1)])
        # If the new value entering the window now is higher than the current
        # maximum, update the maximum.
        elif (x > _max):
            _max = x

        # Save maximum on results array
        res[i_tail] = _max

        # Advance the head pointer
        i_head += 1

    return res

