Brown Dwarf Cloud Spectra with PICASO

This repository contains my Python code for generating synthetic brown dwarf emission spectra using the PICASO radiative transfer framework, with cloud microphysics handled through VIRGA. The project is designed as the first stage of a broader pipeline that will eventually connect to MARGE (for surrogate-model training) and HOMER (for retrieval and inference).

Extra functions and tutorials which are modeled taken straight from PICASO github, refer to:
https://natashabatalha.github.io/picaso/index.html 

Present features:
- Brown Dwarf Modeling with PICASO
    - Create a parameter space of:
      - Effective temperatures
      - Surface gravity’s
      - Cloud list
      - Sedimentation efficiencies
      - Kzz’s
    - Compute spectra with clouds (via Virga)
    - Functions for computing mean molecular weight (MMW).
    - Sampling utilities for parameter sweeps

Data Storage
Outputs are saved in .npz format with consistent keys:
x: input parameters (Teff, g, f_sed, Kzz)
y: flux spectra
wavelength_um: wavelength grid in microns

Roadmap
- Current:
  - End-to-end PICASO brown dwarf runs with Virga clouds.
  - Dataset generation in .npz format.
- Planned:
  - Integration with MARGE to train neural network surrogate models on spectral grids.
  - Coupling with HOMER for retrievals on synthetic and observational data.
