from numpy import linspace, pi, sin

from filter import ButterworthFilter


class Main:
    """
    Main for project
    """

    def __init__(self) -> None:
        n = linspace(0, 255, 255)
        dt = 0.01
        signal = sin(2 * pi * 5 * n * dt) + sin(2 * pi * 20 * n * dt)
        ButterworthFilter(signal=signal, x=n, mode="L", dt=dt, freq_1=10)
        ButterworthFilter(signal=signal, x=n, mode="H", dt=dt, freq_1=10, freq_2=10)

        signal = sin(2 * pi * 5 * n * dt) + sin(2 * pi * 20 * n * dt) + sin(
            2 * pi * 35 * n * dt)
        ButterworthFilter(signal=signal,
                          x=n,
                          mode="B",
                          dt=dt,
                          freq_1=10,
                          freq_2=30)


Main()
