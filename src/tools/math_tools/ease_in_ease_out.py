# coding=utf-8

import math


def easeInSine(t: float) -> float:
    """ Ramp in sine function
    Credit https://v6.robweychert.com/blog/2023/02/python-easing-functions/

    Ramp start at `t % 1.0 == 0.0`

    :param t: time
    :return: ramp at time t
    """
    return -math.cos(t * math.pi / 2) + 1


def easeOutSine(t: float) -> float:
    """ Ramp out sine function
    Credit https://v6.robweychert.com/blog/2023/02/python-easing-functions/

    Ramp end at `t % 1.0 == 0.0`

    :param t: time
    :return: ramp at time t
    """
    return math.sin(t * math.pi / 2)


def easeInOutSine(t: float) -> float:
    """ Ramp in and out sine function
    Credit https://v6.robweychert.com/blog/2023/02/python-easing-functions/

    Ramp start at `t % 1.0 == 0.0` and end at `t+1 % 1.0 == 0.0`

    :param t: time
    :return: ramp at time t
    """
    return -(math.cos(math.pi * t) - 1) / 2


def easeInCubic(t: float) -> float:
    """ Ramp in cubic function
    Credit https://v6.robweychert.com/blog/2023/02/python-easing-functions/

    Ramp start at `t % 1.0 == 0.0`

    :param t: time
    :return: ramp at time t
    """
    return t * t * t


def easeOutCubic(t: float) -> float:
    """ Ramp out cubic function
    Credit https://v6.robweychert.com/blog/2023/02/python-easing-functions/

    Ramp end at `t % 1.0 == 0.0`

    :param t: time
    :return: ramp at time t
    """
    t -= 1
    return t * t * t + 1


def easeInOutQuad(t: float) -> float:
    """ Ramp in and out quad function
    Credit https://v6.robweychert.com/blog/2023/02/python-easing-functions/

    Ramp start at `t % 1.0 == 0.0` and end at `t+1 % 1.0 == 0.0`

    :param t: time
    :return: ramp at time t
    """
    t *= 2
    if t < 1:
        return t * t / 2
    else:
        t -= 1
        return -(t * (t - 2) - 1) / 2


def easeInOutCubic(t: float) -> float:
    """ Ramp in and out cubic function
    Credit https://v6.robweychert.com/blog/2023/02/python-easing-functions/

    Ramp start at `t % 1.0 == 0.0` and end at `t+1 % 1.0 == 0.0`

    :param t: time
    :return: ramp at time t
    """
    t *= 2
    if t < 1:
        return t * t * t / 2
    else:
        t -= 2
        return (t * t * t + 2) / 2


def easeInExpo(t: float) -> float:
    """ Ramp in expo function
    Credit https://v6.robweychert.com/blog/2023/02/python-easing-functions/

    Ramp start at `t % 1.0 == 0.0`

    :param t: time
    :return: ramp at time t
    """
    import math

    return math.pow(2, 10 * (t - 1))


def easeOutExpo(t: float) -> float:
    """ Ramp out expo function
    Credit https://v6.robweychert.com/blog/2023/02/python-easing-functions/

    Ramp end at `t % 1.0 == 0.0`

    :param t: time
    :return: ramp at time t
    """
    import math

    return -math.pow(2, -10 * t) + 1


def easeInOutExpo(t: float) -> float:
    """ Ramp in and out expo function
    Credit https://v6.robweychert.com/blog/2023/02/python-easing-functions/

    Ramp start at `t % 1.0 == 0.0` and end at `t+1 % 1.0 == 0.0`

    :param t: time
    :return: ramp at time t
    """
    import math

    t *= 2
    if t < 1:
        return math.pow(2, 10 * (t - 1)) / 2
    else:
        t -= 1
        return -math.pow(2, -10 * t) - 1
