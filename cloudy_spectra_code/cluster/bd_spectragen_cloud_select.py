# bd_spectragen_cloud_select.py
##### IMPORTS

import os
os.environ['picaso_refdata'] = r'C:\Users\Alex\Desktop\Picaso\picaso\reference' # THIS MUST GO BEFORE YOUR IMPORT STATEMENT
os.environ['PYSYN_CDBS'] = r'C:\Users\Alex\Desktop\Picaso\grp\redcat\trds' # This is for the stellar data discussed below.

# General
import numpy as np
import astropy.units as u
import bd_support as sup

from pathlib import Path
from itertools import product

# Picaso and Virga
from picaso import justdoit as jdi
from virga import justdoit as vj

# To see what clouds are availible
# vj.available()


##### CONFIGURATIONS

# Directories
sonor_path  = r'C:\Users\Alex\Desktop\Picaso\data\sonora' # Sonora db
# sonor_path  = '/groups/tkaralidi/pbraunschweig/training_set/profiles/'
virga_path  = r'C:\Users\Alex\Desktop\Picaso\data\virga'  # Virga
# virga_path  = '/home/sa221179/picaso/virga/'
opaci_path  = None # Opacity db
# opcai_path  = '/groups/tkaralidi/opacity_500k_for_R5000_egpoutput.db'
output_path = Path(r"C:\Users\Alex\Desktop\Picaso\NN_project\outputs")
# output_path = 'home/al864695/ouputs'

# Constant values
wav_range   = [0.5, 15.0] # microns
MH          = 1.0        # [M/H] metallicity factor ~ solar
MU          = 2.36       # Average MU
R           = 300        # resolution
# R           = 5000

# Dictionary for cloud naming conventions
cloud_dict  = {'Fe': '1', 'H2O': '2', 'KCl': '3', 'Mg2SiO4': '4',
               'MgSiO3': '5', 'MnS': '6', 'NH3': '7', 'Na2S': '8'}


##### METHOD BROWN DWARF SPECTRUM=

def bd_spectrum(Teff, logg, fsed, kzz):
    """
    Compute a BD emission spectrum with Virga clouds.
    """

    # Opacity & inputs
    opa    = jdi.opannection(wav_range, opaci_path)
    bd     = jdi.inputs(calculation="browndwarf")
    bd.phase_angle(0)
    # Convert log g [cgs] to grav [m s^-2]
    gravity = 10**logg * 1e-2   # 1 cm s^-2 = 0.01 m s^-2
    bd.gravity(gravity, gravity_unit=u.Unit('m/s**2'))
    bd.sonora(sonor_path, Teff)

    # Inject corrected TP
    sup.inject_corr(bd, Teff, logg, fsed)

    # Inject Kzz (match pressure grid length)
    prof   = bd.inputs['atmosphere']['profile']
    P      = np.asarray(prof["pressure"], float)
    T      = np.asarray(prof['temperature'], float)
    bd.inputs["atmosphere"]["profile"]["kz"] = [float(kzz)] * len(P)

    # Recommended gas
    rec    = vj.recommend_gas(P, T, MH, MU, plot=False)
    # Remove THESE species (first 2 not supported) (other irrelevant to BD)
    exclude = {'CaAl12O19', 'SiO2', 'CH4', 'Cr', 'TiO2', 'Al2O3', 'CaTiO3', 'ZnS'}
    # Should only have any combo of these:
    # ['Fe', 'H2O', 'KCl', 'Mg2SiO4', 'MgSiO3', 'MnS', 'NH3', 'Na2S']
    clouds = [sp for sp in rec if sp not in exclude]

    # Saving cloud names, like 123 means clouds 1,2 and 3 used
    cl_names = ''.join(sorted(cloud_dict[c] for c in clouds))  

    # Clouds
    bd.virga(clouds, virga_path, fsed, mh=MH, mmw=MU)
    out    = bd.spectrum(opa, full_output=True)
    
    wn, fl = out["wavenumber"], out["thermal"]  # cm^-1 and erg cm^-2 s^-1 cm^-1
    
    # Regrid in wavenumber space because F is defined per wavenumber
    wn, fl = jdi.mean_regrid(wn, fl, R=R)

    # Convert wavenumber [cm^-1] to wavelength [micron]
    #w_um = 1e4 / wn

    return fl, cl_names


##### GENERATE AND SAVE SPECTRUM (MARGE format, single-case files)

Teff_s = [1253.027059]      # K
grav_s = 1082.0568603568015
logg_s = [np.log10(grav_s * 100)]  # log g cgs
fsed_s = [2.0] 
kzz_s  = [1e10]

for Teff_i, logg_i, fsed_i, kzz_i in product(Teff_s, logg_s, fsed_s, kzz_s):

    # Run spectrum
    F_i, names = bd_spectrum(Teff_i, logg_i, fsed_i, kzz_i)

    # Filename encodes the parameters; ML will parse from name
    fname  = (f"t{float(Teff_i)}g{float(logg_i)}f{float(fsed_i)}"
              f"{sup.format_kzz(kzz_i)}c{names}.npy")
    fpath  = output_path / fname

    np.save(fpath, F_i)
