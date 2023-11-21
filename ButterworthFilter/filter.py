from typing import Literal, Optional
import numpy as np
import matplotlib.pyplot as plt


class ButterworthFilter:
    """
    Python implementation of butterworth filter
    
    parameters:
    - `signal`: Numpy arraylike discrete signal samples
    - `mode`: `L` = Low-pass, `H` = High-pass, `B` = Bandpass
    - `dt`: sample interval
    - `freq_1`, *`freq_2`: frequency(Hz) parameters for filtering
    """

    def __init__(self,
                 signal,
                 x,
                 mode: Literal["L", "H", "B"],
                 dt: float,
                 freq_1: float,
                 freq_2: Optional[float] = None) -> None:
        self.y = signal
        self.x = x
        self.dt = dt
        self.f_1 = freq_1
        self.f_2 = freq_2
        if mode == 'L':
            self.low_pass()
        elif mode == 'H':
            self.high_pass()
        else:
            self.band_pass()

    def low_pass(self):
        """
        Low-pass filter
        """
        c = self.f_1 * np.pi * self.dt
        b_0 = (c**2) / (c**2 + np.sqrt(2) * c + 1)
        b_1 = 2 * b_0
        b_2 = b_0
        a_1 = 2 * (c**2 - 1) / (c**2 + np.sqrt(2) * c + 1)
        a_2 = (c**2 - np.sqrt(2) * c + 1) / (c**2 + np.sqrt(2) * c + 1)
        y_out = []
        for i, y_i in enumerate(self.y):
            if i == 0:
                y_out_i = b_0 * y_i
                y_out.append(y_out_i)
            elif i == 1:
                y_out_i = b_0 * y_i + b_1 * self.y[0] - a_1 * y_out[0]
                y_out.append(y_out_i)
            else:
                y_out_i = b_0 * y_i + b_1 * self.y[i - 1] + b_2 * self.y[
                    i - 2] - a_1 * y_out[i - 1] - a_2 * y_out[i - 2]
                y_out.append(y_out_i)
        plt.figure(figsize=(15, 8))
        plt.subplot(2, 1, 1)
        plt.title("Original Signal")
        plt.plot(self.x, self.y, c='black')
        plt.subplot(2, 1, 2)
        plt.title(f"Low-pass Filtered Signal, f={self.f_1}Hz")
        plt.plot(self.x, y_out, c='b')
        plt.show()

        return y_out

    def high_pass(self):
        """
        High-pass filter
        """
        w_1 = 10 * 2 * np.pi * self.dt
        w_2 = self.f_1 * 2 * np.pi * self.dt
        c = w_1 / 2
        b_0 = (c**2) / (c**2 + np.sqrt(2) * c + 1)
        b_1 = 2 * b_0
        b_2 = b_0
        a_1 = 2 * (c**2 - 1) / (c**2 + np.sqrt(2) * c + 1)
        a_2 = (c**2 - np.sqrt(2) * c + 1) / (c**2 + np.sqrt(2) * c + 1)
        theta_1 = (w_1 + w_2) / 2
        theta_2 = (w_1 - w_2) / 2
        alpha = -(np.cos(theta_1) / np.cos(theta_2))
        y_out = []
        for i, y_i in enumerate(self.y):
            if i == 0:
                y_out_i = (b_0 - b_1 * alpha + b_2 * alpha**2) * y_i / (
                    1 - a_1 * alpha + a_2 * alpha**2)
                y_out.append(y_out_i)
            elif i == 1:
                y_out_i = (
                    (b_0 - b_1 * alpha + b_2 * alpha**2) * y_i +
                    (2 * b_0 * alpha - b_1 - b_1 * alpha**2 + 2 * b_2 * alpha)
                    * self.y[0] -
                    (2 * alpha - a_1 - a_1 * alpha**2 + 2 * a_2 * alpha) *
                    y_out[0]) / (1 + a_2 * alpha**2 - a_1 * alpha)
                y_out.append(y_out_i)
            else:
                y_out_i = (
                    (b_0 - b_1 * alpha + b_2 * alpha**2) * y_i +
                    (2 * b_0 * alpha - b_1 - b_1 * alpha**2 + 2 * b_2 * alpha)
                    * self.y[i - 1] -
                    (2 * alpha - a_1 - a_1 * alpha**2 + 2 * a_2 * alpha) *
                    y_out[i - 1] +
                    (b_0 * alpha**2 - b_1 * alpha + b_2) * self.y[i - 2] -
                    (alpha**2 - a_1 * alpha + a_2) * y_out[i - 2]) / (
                        1 + a_2 * alpha**2 - a_1 * alpha)
                y_out.append(y_out_i)
                print(y_out_i)
        plt.figure(figsize=(15, 8))
        plt.subplot(2, 1, 1)
        plt.title("Original Signal")
        plt.plot(self.x, self.y, c='black')
        plt.subplot(2, 1, 2)
        plt.title(f"High-pass Filtered Signal, f={self.f_1}Hz")
        plt.plot(self.x, y_out, c='b')
        plt.show()

    def band_pass(self):
        """
        band-pass filter
        """
        w_1 = 10 * 2 * np.pi * self.dt
        w_2 = self.f_1 * 2 * np.pi * self.dt
        w_3 = self.f_2 * 2 * np.pi * self.dt
        c = w_1 / 2
        b_0 = (c**2) / (c**2 + np.sqrt(2) * c + 1)
        b_1 = 2 * b_0
        b_2 = b_0
        a_1 = 2 * (c**2 - 1) / (c**2 + np.sqrt(2) * c + 1)
        a_2 = (c**2 - np.sqrt(2) * c + 1) / (c**2 + np.sqrt(2) * c + 1)
        theta_1 = (w_3 + w_2) / 2
        theta_2 = (w_3 - w_2) / 2
        gamma = np.cos(theta_1) / np.cos(theta_2)
        k = np.tan(w_1 / 2) / np.tan(theta_2)
        alpha = (2 * gamma * k) / (k + 1)
        beta = (k - 1) / (k + 1)
        y_out = []
        for i, y_i in enumerate(self.y):
            if i == 0:
                y_out_i = (b_0 - b_1 * beta + b_2 * beta**2) * y_i / (
                    1 - a_1 * beta + a_2 * beta**2)
                y_out.append(y_out_i)
            elif i == 1:
                y_out_i = (
                    (b_0 - b_1 * beta + b_2 * beta**2) * y_i -
                    (a_1 * alpha + alpha * beta * a_1 - 2 * alpha -
                     2 * alpha * beta * a_2) * y_out[0] +
                    (b_1 * alpha + alpha * beta * b_1 - 2 * alpha * b_0 -
                     2 * alpha * beta * b_2) * self.y[0]) / (1 - a_1 * beta +
                                                             a_2 * beta**2)
                y_out.append(y_out_i)
            elif i == 2:
                y_out_i = (
                    (b_0 - b_1 * beta + b_2 * beta**2) * y_i -
                    (a_1 * alpha + alpha * beta * a_1 - 2 * alpha -
                     2 * alpha * beta * a_2) * y_out[1] +
                    (b_1 * alpha + alpha * beta * b_1 - 2 * alpha * b_0 -
                     2 * alpha * beta * b_2) * self.y[1] -
                    (alpha**2 + 2 * beta - a_1 - alpha**2 * a_1 - beta**2 * a_1
                     + alpha**2 * a_2 + 2 * beta * a_2) * y_out[0] +
                    (b_0 * alpha**2 + 2 * beta * b_0 - b_1 - alpha**2 * b_1 -
                     beta**2 * b_1 + alpha**2 * b_2 + 2 * beta * b_2) *
                    self.y[0]) / (1 - a_1 * beta + a_2 * beta**2)
                y_out.append(y_out_i)
            elif i == 3:
                y_out_i = (
                    (b_0 - b_1 * beta + b_2 * beta**2) * y_i -
                    (a_1 * alpha + alpha * beta * a_1 - 2 * alpha -
                     2 * alpha * beta * a_2) * y_out[2] +
                    (b_1 * alpha + alpha * beta * b_1 - 2 * alpha * b_0 -
                     2 * alpha * beta * b_2) * self.y[2] -
                    (alpha**2 + 2 * beta - a_1 - alpha**2 * a_1 - beta**2 * a_1
                     + alpha**2 * a_2 + 2 * beta * a_2) * y_out[1] +
                    (b_0 * alpha**2 + 2 * beta * b_0 - b_1 - alpha**2 * b_1 -
                     beta**2 * b_1 + alpha**2 * b_2 + 2 * beta * b_2) *
                    self.y[1] -
                    (a_1 * alpha + a_1 * alpha * beta - 2 * alpha * beta -
                     2 * alpha * a_2) * y_out[0] +
                    (b_1 * alpha + b_1 * alpha * beta -
                     2 * alpha * beta * b_0 - 2 * alpha * b_2) * self.y[0]) / (
                         1 - a_1 * beta + a_2 * beta**2)
                y_out.append(y_out_i)
            else:
                y_out_i = (
                    (b_0 - b_1 * beta + b_2 * beta**2) * y_i -
                    (a_1 * alpha + alpha * beta * a_1 - 2 * alpha -
                     2 * alpha * beta * a_2) * y_out[i - 1] +
                    (b_1 * alpha + alpha * beta * b_1 - 2 * alpha * b_0 -
                     2 * alpha * beta * b_2) * self.y[i - 1] -
                    (alpha**2 + 2 * beta - a_1 - alpha**2 * a_1 - beta**2 * a_1
                     + alpha**2 * a_2 + 2 * beta * a_2) * y_out[i - 2] +
                    (b_0 * alpha**2 + 2 * beta * b_0 - b_1 - alpha**2 * b_1 -
                     beta**2 * b_1 + alpha**2 * b_2 + 2 * beta * b_2) *
                    self.y[i - 2] -
                    (a_1 * alpha + a_1 * alpha * beta - 2 * alpha * beta -
                     2 * alpha * a_2) * y_out[i - 3] +
                    (b_1 * alpha + b_1 * alpha * beta - 2 * alpha * beta * b_0
                     - 2 * alpha * b_2) * self.y[i - 3] -
                    (beta**2 + a_2 - a_1 * beta) * y_out[i - 4] +
                    (b_0 * beta**2 + b_2 - b_1 * beta) *
                    self.y[i - 4]) / (1 - a_1 * beta + a_2 * beta**2)
                y_out.append(y_out_i)
                print(y_out_i)
        plt.figure(figsize=(15, 8))
        plt.subplot(2, 1, 1)
        plt.title("Original Signal")
        plt.plot(self.x, self.y, c='black')
        plt.subplot(2, 1, 2)
        plt.title(
            f"Band-pass Filtered Signal, f1={self.f_1}Hz, f2={self.f_2}Hz")
        plt.plot(self.x, y_out, c='b')
        plt.show()
