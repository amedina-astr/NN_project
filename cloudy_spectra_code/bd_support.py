# bd_support.py
# To load and apply Bobcat to Diamondback (Peter TP correction factors)

##### IMPORTS

import re
import pickle
import numpy as np

from scipy.interpolate import RegularGridInterpolator

##### CONFIGURATIONS

pickl_path  = r"C:\Users\Alex\Desktop\Picaso\NN_project\cloudy_spectra_code\bobcat_to_diamondback.pickle"
# pickl_path  = "home/al864695/pickle"

##### HELPER METHODS

# For naming convention so not cluttered
def format_kzz(k):
    """
    Return 'k2e9' style tag from a float like 2e10
    """
    s = f"{float(k):.0e}"            # e.g., '1e+09'
    s = re.sub(r"e\+?0*", "e", s)    # '1e+09'to '1e9'
    return f"k{s}"                   # 'k1e9'

# The pickle is a numpy array of shape [301, 41, 5, 91]
# From Theodora code
# 301 corresponds to teff from [900, 2400] at 5 K steps
# 41 corresponds to logg from [3.50, 5.50] at 0.05 steps
# 5 corresponds to fsed from [1, 2, 3, 4, 8]
# 91 corresponds to 91 layers
# Each entry is a temp? correction factor for that layer

# Load and define axes
corrp  = pickle.load(open(pickl_path, "rb"))           # shape (301, 41, 5, 91)
taxis  = np.arange(900.0, 2405.0, 5.0)                 # 900..2400 (step 5)
gaxis  = np.round(np.arange(3.50, 5.54, 0.05), 3)      # 3.50..5.50 (step 0.05)
faxis  = np.array([1.0, 2.0, 3.0, 4.0, 8.0])           # fsed axis

# Pickle interpolation is still discrete?
# So for random sampling, it wouldn't quite work
# Or even out of range?

# Let's see
# Only fsed is not interpolated so that can be reference axis
# Do all sonora profiles have 91 layers?

def inject_corr(bd, Teff, logg, fsed, clip_oob=True):
    """
    Apply Diamondback/Bobcat TP correction to PICASO inputs.
    Multiplies the current temperature profile by a correction factor
    interpolated over (Teff, logg) at nearest fsed.
    """
    # Choose the nearest fsed because since interpolation here
    fi = int(np.argmin(np.abs(faxis - float(fsed))))

    # Clip Teff/logg to grid so out of bound values still gets a correction
    # Extrapolation but we'll see
    Teff_in = float(np.clip(Teff, taxis[0], taxis[-1])) if clip_oob else float(Teff)
    logg_in = float(np.clip(logg, gaxis[0], gaxis[-1])) if clip_oob else float(logg)

    # For values like random sampling
    # Interpolate over Teff and log g at that fsed
    rgi = RegularGridInterpolator((taxis, gaxis),
                                  corrp[:, :, fi, :],
                                  bounds_error=not clip_oob,
                                  fill_value=None)
    
    # Interpolate
    corr = rgi((Teff_in, logg_in)).ravel()

    prof = bd.inputs["atmosphere"]["profile"]
    T    = np.asarray(prof["temperature"], float)
    prof["temperature"] = (T * corr).tolist()
