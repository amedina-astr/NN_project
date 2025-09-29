# tp_correction.py
# Utilities to load and apply Bobcat to Diamondback (Peter TP correction factors)

from __future__ import annotations
import pickle
import numpy as np
import re
from typing import Sequence, Tuple

# For naming convention so not cluttered
def format_kzz(k):
    """
    Return 'k2e9' style tag from a float like 2e10
    """
    s = f"{float(k):.0e}"                  # e.g., '1e+09'
    s = re.sub(r"e\+?0*", "e", s)          # '1e+09' -> '1e9', '2e+10' -> '2e10'
    return f"k{s}"                         # 'k1e9'

# For correct TP profile
class TPCorrection:
    """
    Load correction factors and apply them to a PICASO inputs object.

    The shape is n_Teff, n_g, n_fsed, n_layers) which describes
    corr_facs[temps, gravs, fsed, 91].
    """

    # Load pickle
    def __init__(self, pickle_path: str):
        with open(pickle_path, 'rb') as f:
            arr = pickle.load(f)
        self.corr = arr  # shape (nT, nG, nF, nL)
        self.nT, self.nG, self.nF, self.nL = arr.shape
        # Coordinate vectors must be supplied by the caller (see set_coords)
        self._Teff = None
        self._g = None
        self._fsed = None

    def set_coords(self, Teff_grid: Sequence[float], g_grid: Sequence[float], fsed_grid: Sequence[float]):
        """
        Attach the coordinate vectors that correspond to the correction array axes.

        - Teff_grid: list/array of temperatures (K), length = nT
        - g_grid: gravities in m s^-2, length = nG
        - fsed_grid: list of fsed values, length = nF
        """
        Teff = np.asarray(Teff_grid)
        g = np.asarray(g_grid)
        f = np.asarray(fsed_grid)
        if len(Teff) != self.nT or len(g) != self.nG or len(f) != self.nF:
            raise ValueError("Coordinate lengths must match correction array dimensions: "
                             f"{self.nT} (Teff), {self.nG} (g), {self.nF} (fsed)." )
        self._Teff, self._g, self._fsed = Teff, g, f

    def _find_index(self, value: float, grid: np.ndarray) -> int:
        # Bearest-neighbor index
        # But **reject** extrapolation outside min/max
        if value < grid.min() or value > grid.max():
            raise ValueError(f"Requested value {value} is outside grid range [{grid.min()}, {grid.max()}]; "
                             "extrapolation is not allowed.")
        return int(np.argmin(np.abs(grid - value)))

    def index_for(self, Teff: float, g: float, fsed: float) -> Tuple[int, int, int]:
        if self._Teff is None:
            raise RuntimeError("Call set_coords(...) first to provide axis coordinate arrays.")
        iT = self._find_index(Teff, self._Teff)
        iG = self._find_index(g, self._g)
        iF = self._find_index(fsed, self._fsed)
        return iT, iG, iF

    def vector(self, Teff: float, g_mps2: float, fsed: float) -> np.ndarray:
        iT, iG, iF = self.index_for(Teff, g_mps2, fsed)
        return self.corr[iT, iG, iF, :]  # shape (nL,)

    def apply_to_picaso_inputs(self, bd_inputs, Teff: float, g: float, fsed: float):
        """
        Apply correction to the temperature profile inside a PICASO inputs object (bd_inputs).

        Parameters
        ----------
        bd_inputs : jdi.inputs object (already has .sonora(...) called)
        Teff, g_mps2, fsed : float
            Values used to pick the correction vector via nearest-neighbor (no extrapolation).
        """
        prof = bd_inputs.inputs['atmosphere']['profile']
        T = np.asarray(prof['temperature'], float)
        corr = np.asarray(self.vector(Teff, g, fsed), float)
        if len(T) != len(corr):
            raise ValueError(f"Layer mismatch: PICASO T has length {len(T)} but correction has {len(corr)}.")
        Tcorr = T * corr

        bd_inputs.inputs['atmosphere']['profile']['temperature'] = Tcorr.tolist()
        return Tcorr
