import numpy as np
import pandas as pd
from src.utils import utils


no_params = {
    'lin': 2,  # a, b
    'exp': 3,  # a, b, c
    'gaussian': 3,  # mean, sigma, amp
    'asymcb': 7,  # mean, sigmaL, sigmaR, alphaL, alphaR, nL, nR, amp
    'cb': 5,  # mean, sigma, alpha, n, amp
    'deltambg': 5  # m0, varA, varB, varC, amp
}

def linear_1d(x, a: int | float, b: int | float):
    return a * x + b


def exp_1d(x, a: int | float, b: int | float, c: int | float):
    return a * np.exp(b * x) + c


def gaussian_1d(x, mean: int | float, sigma: int | float, amp: int | float):
    return amp * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))


def asymcb_1d(x: int | float | np.ndarray | pd.Series,
              mean: int | float, sigmaL: int | float, sigmaR: int | float,
              alphaL: int | float, alphaR: int | float, nL: int | float, nR: int | float,
              amp: int | float) -> int | float | np.ndarray | pd.Series:
    """
    Crystal Ball function. Read https://root.cern/doc/v628/classRooCrystalBall.html for more information.
    Args:
        x: data
        mean: mean of the data. denoted m0 on the website
        sigmaL: left standard deviation. Can be smaller than 0, but will be preprocessed as sigmaL = max(0.01, sigmaL)
        sigmaR: right standard deviation. Can be smaller than 0, but will be preprocessed as sigmaR = max(0.01, sigmaR)
        alphaL: left tail parameter. Can be smaller than 0, but will be preprocessed as alphaL = max(0.01, alphaL)
        alphaR: right tail parameter. Can be smaller than 0, but will be preprocessed as alphaR = max(0.01, alphaR)
        nL: left tail power. Can be smaller than 0, but will be preprocessed as nL = max(0, nL)
        nR: right tail power. Can be smaller than 0, but will be preprocessed as nR = max(0, nR)
        amp: amplitude of the function

    Returns:
        int | float | np.ndarray | pd.Series. The result of the function
    """
    # preprocess
    if any([sigmaL < 0, sigmaR < 0, alphaL < 0, alphaR < 0, nL < 0, nR < 0]):
        utils.log('Warning: negative values detected in asymcb_1d.')

    # coefficients
    AL = (nL / alphaL) ** nL * np.exp(-0.5 * alphaL ** 2)
    AR = (nR / alphaR) ** nR * np.exp(-0.5 * alphaR ** 2)
    BL = nL / alphaL - alphaL
    BR = nR / alphaR - alphaR

    # cb's version of `z = (x - mean) / sigma`
    epsilon = 0.001
    zL = (x - mean) / sigmaL
    blzlthing = (BL - zL)
    blzlthing = np.where(blzlthing < 0, epsilon, blzlthing)
    zL = np.where(zL < -alphaL, AL * blzlthing ** (-nL), np.exp(-0.5 * zL ** 2))

    zR = (x - mean) / sigmaR
    brzrthing = (BR + zR)
    brzrthing = np.where(brzrthing < 0, epsilon, brzrthing)
    zR = np.where(zR > alphaR, AR * brzrthing ** (-nR), np.exp(-0.5 * zR ** 2))
    z = np.where(x < mean, zL, zR)
    return amp * z


def cb_1d(x: int | float | np.ndarray | pd.Series,
          mean: int | float, sigma: int | float, alpha: int | float, n: int | float,
          amp: int | float) -> int | float | np.ndarray | pd.Series:
    """
    https://en.wikipedia.org/wiki/Crystal_Ball_function
    """
    # preprocess
    if any([sigma < 0, alpha < 0, n < 0]):
        utils.log('Warning: negative values detected in cb_1d.')

    # Calculate
    coeff_a = (n / abs(alpha)) ** n * np.exp(-0.5 * alpha ** 2)
    coeff_b = n / abs(alpha) - abs(alpha)

    z = (x - mean) / sigma
    bzthing = np.where(coeff_b - z > 0, coeff_b - z, 0.001)
    z = np.where(z > -alpha, np.exp(-0.5 * z ** 2), coeff_a * bzthing ** (-n))
    return amp * z


def deltambg_1d(x: int | float | np.ndarray | pd.Series,
                m0: int | float, varA: int | float, varB: int | float, varC: int | float,
                amp: int | float) -> int | float | np.ndarray | pd.Series:
    """
    DstD0BG function. Read https://root.cern/doc/v628/classRooDstD0BG.html for more information.
    Args:
        x: data
        m0: mean of the data. denoted m0 on the website
        varA: coefficient A.
        varB: coefficient B. Proportional to the slope of the second part of the function
        varC: coefficient C. Strength of the knee.
        amp: amplitude of the function. Approximately y value at the knee.

    Returns:
        int | float | np.ndarray | pd.Series. The result of the function
    """
    # preprocess
    x = np.where(x < m0, m0, x)
    if any([varA < 0, varB < 0, varC < 0]):
        utils.log('Warning: negative values detected in deltambg_1d.')

    # Calculate
    ret = (1 - np.exp(-(x - m0) / varC)) * (x / m0) ** varA + varB * (x / m0 - 1)
    return amp * ret
