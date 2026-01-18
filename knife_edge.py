import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import erf
import os
import sys

# Gaussian profile
def gaussian_cdf(x, x0, w, P_total, offset):
    """
    Args:
        x: position [mm]
        x0: beam center [mm]
        w: 1/e^2 radius [mm]
        P_total: total power [mW]
        offset: background noise level [mW]

    Returns:
        cumulative power at position x
    """
    return P_total / 2 * (1 + erf(np.sqrt(2) * (x - x0) / w)) + offset

# initial guess
def get_initial_guesses(x, y):
    """
    Args:
        x: position array [mm]
        y: power measurement array [mW]

    Returns:
        initial val. array [x0, w, P_total, offset]
    """
    offset_guess = np.min(y)
    P_total_guess = np.max(y) - np.min(y)
    
    # beam center @ gradient max.
    grad = np.gradient(y, x)
    x0_guess = x[np.argmax(np.abs(grad))]
    w_guess = (np.max(x) - np.min(x)) / 5
    if w_guess == 0: w_guess = 0.1
    
    return [x0_guess, w_guess, P_total_guess, offset_guess]

# load data
def load_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)
    data = df.iloc[:, :2].dropna().values
    return data[:, 0], data[:, 1]

# Knife edge analysis
def knife_edge(filename, wavelength_nm=400.0, plot=True):
    """
    Args:
        filename
        wavelength_nm: center wavelength [nm]
        plot: plot boolean

    Returns:
        1/e^2 beam radius w [mm]
    """
    
    # load data
    x_raw, y_raw = load_data(filename)
    sort_idx = np.argsort(x_raw)
    x_raw, y_raw = x_raw[sort_idx], y_raw[sort_idx]

    # fitting
    p0 = get_initial_guesses(x_raw, y_raw)
    try:
        popt, pcov = curve_fit(gaussian_cdf, x_raw, y_raw, p0=p0, maxfev=10000)
    except RuntimeError:
        print("Converge failed!")
        return None

    x0_fit, w_fit, P_fit, off_fit = popt
    w_fit = abs(w_fit)
    
    # calculate parameters
    residuals = y_raw - gaussian_cdf(x_raw, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_raw - np.mean(y_raw))**2)
    r_squared = 1 - (ss_res / ss_tot)
    wavelength_mm = wavelength_nm * 1e-6
    z_R = (np.pi * w_fit**2) / wavelength_mm

    print(f"{'='*50}")  
    print(f"{os.path.basename(filename)}")
    print(f"{'-'*50}")
    print(f"Wavelength      : {wavelength_nm} nm")
    print(f"1/e² Radius (w)   : {w_fit*1e3:.2f} μm")
    print(f"1/e² Diameter (2w): {2*w_fit*1e3:.2f} μm")
    print(f"Rayleigh Range    : {z_R:.2f} mm")
    print(f"Fit R² Score      : {r_squared:.5f}")
    print(f"{'='*50}")

    # 6. Plotting (Only CDF Fit)
    if plot:
        plt.figure(figsize=(10, 6))
        
        # raw data
        plt.plot(x_raw, y_raw, 'ro', label='Measured Data', zorder=5)
        
        # fitted curve
        x_smooth = np.linspace(min(x_raw), max(x_raw), 500)
        plt.plot(x_smooth, gaussian_cdf(x_smooth, *popt), 'k-', lw=2, label='Fitted curve')
        
        plt.axvline(x0_fit, color='gray', ls=':', label='Center')
        plt.axvline(x0_fit - w_fit, color='green', ls='--', alpha=0.8, label=f'1/e² Width')
        plt.axvline(x0_fit + w_fit, color='green', ls='--', alpha=0.8)
        plt.xlabel('Position [mm]')
        plt.ylabel('Power [mW]')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return w_fit

if __name__ == "__main__":
    files = [f for f in os.listdir('.') if f.endswith(('.xlsx', '.xls', '.csv'))]
    print(f"Files found: {files}")
    target_file = input("Enter filename: ").strip()
    wl_input = input("Enter wavelength (nm): ").strip()
    wavelength = float(wl_input) if wl_input else 400.0
    knife_edge(target_file, wavelength)