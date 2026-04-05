import numpy as np
from scipy.stats import norm

def _d1_d2(S: np.ndarray, K: np.ndarray, T: float,
           r: float, sigma: np.ndarray):
    """Vectorised d1 & d2 (all inputs may be arrays)."""
    # Clamp to avoid log(0) or division by zero
    T_safe     = np.maximum(T, 1e-8)
    sigma_safe = np.maximum(sigma, 1e-8)
    d1 = (np.log(S / K) + (r + 0.5 * sigma_safe ** 2) * T_safe) / (sigma_safe * np.sqrt(T_safe))
    d2 = d1 - sigma_safe * np.sqrt(T_safe)
    return d1, d2


def bs_price(S, K, T, r, sigma, option="call"):
    """Black-Scholes-Merton European option price."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if option == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_greeks(S, K, T, r, sigma):
    """
    Returns a dict of Greeks for both call and put.
    All inputs can be scalars or arrays (vectorised).
    """
    T_safe     = np.maximum(T, 1e-8)
    sigma_safe = np.maximum(sigma, 1e-8)
    d1, d2     = _d1_d2(S, K, T_safe, r, sigma_safe)

    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)

    # ── Delta ──────────────────────────────────
    call_delta = cdf_d1
    put_delta  = cdf_d1 - 1.0          # ≡ N(d1) - 1

    # ── Gamma (same for call & put) ────────────
    gamma = pdf_d1 / (S * sigma_safe * np.sqrt(T_safe))

    # ── Theta (per calendar day, divide by 365) ─
    call_theta = (
        -(S * pdf_d1 * sigma_safe) / (2 * np.sqrt(T_safe))
        - r * K * np.exp(-r * T_safe) * cdf_d2
    ) / 365.0
    put_theta = (
        -(S * pdf_d1 * sigma_safe) / (2 * np.sqrt(T_safe))
        + r * K * np.exp(-r * T_safe) * norm.cdf(-d2)
    ) / 365.0

    return {
        "call_delta": call_delta,
        "put_delta":  put_delta,
        "gamma":      gamma,
        "call_theta": call_theta,
        "put_theta":  put_theta,
    }

