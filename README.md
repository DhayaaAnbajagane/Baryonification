<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/DhayaaAnbajagane/Baryonification/main/docs/source/LOGO_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/DhayaaAnbajagane/Baryonification/main/docs/source/LOGO_light.png">
  <img alt="Logo" src="https://raw.githubusercontent.com/DhayaaAnbajagane/Baryonification/main/docs/source/LOGO_dark.png" title="Logo">
</picture>

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.6%2B-blue.svg)](https://www.python.org/downloads/)
[![Documentation Status](https://img.shields.io/readthedocs/baryonification?color=blue)](https://baryonification.readthedocs.io/en/latest)

## Overview

A pipeline for *Baryonifying* N-body simulations, by adding baryon-induced corrections to the density field and/or adding thermodynamic fields such as the gas pressure, temperature etc. The entire modelling pipeline is built out of the Core Cosmology Library (CCL).

## Features

- **Baryonification**: Modify density fields from N-body simulations and/or paint any field that has a halo profile associated with it


- **Maps, grids, and snapshots**: Work directly with 2D fields (eg. HealPix maps or 2D grids) but can also use 3D grids or full particle snapshots


- **Parallelized**: Painting and baryonification is parallelized under joblib (though the CCL profiles behave properly only in some stable package configs, see yml files in repo. This is still being worked on...)


A detailed documentation is available at [readthedocs](https://baryonification.readthedocs.io/en/latest).

## Installation

To install the package, run the following command:

```bash
pip install git+https://github.com/DhayaaAnbajagane/Baryonification.git
```

or alternatively you can download the repo yourself and set it up,

```bash
git clone https://github.com/DhayaaAnbajagane/Baryonification.git
cd Baryonification
pip install -e .
```

This will keep the source files in the location you git clone'd from.


## Quickstart

```python
import Baryonification as bfn
import pyccl as ccl

#Add the healpix map and the lightcone halo catalog into the respective data objects
Shell   = bfn.utils.LightconeShell(map = HealpixMap, cosmo = cosmo_dict)
Catalog = bfn.utils.HaloLightConeCatalog(ra = ra, dec = dec, M = M200c, z = z, cdelta = c200c)

#Define a cosmology object, to be used in all profile calculations
cosmo   = ccl.Cosmology(Omega_c = 0.26, h = 0.7, Omega_b = 0.04, sigma8 = 0.8, n_s = 0.96)

#Define the DMO and DMB model which are the root of the baryonification routine
#The model params can be specified during initialization of the class.
#The Baryonification 2D class generates the offsets of density field.
#We setup an interpolator to speed up the calculations.
DMO     = bfn.Profiles.DarkMatterOnly(M_c = 1e14, proj_cutoff = 100)
DMB     = bfn.Profiles.DarkMatterBaryon(M_c = 1e14, proj_cutoff = 100)
model   = bfn.Profiles.Baryonification2D(DMO, DMB, cosmo)
model.setup_interpolator(z_min = Catalog.cat['z'].min(), z_max = Catalog.cat['z'].max(), N_samples_z = 10,
                         M_min = Catalog.cat['M'].min(), M_max = Catalog.cat['M'].max(), N_samples_M = 10,
                         R_min = 1e-3, R_max = 3e2, N_samples_R = 500,)

#The halo pressure profile as well. This is convolved with a Healpix window function
#and then tabulated for speedup
PRESS   = bfn.Profiles.Pressure(theta_ej = 8, theta_co = 0.1, mu_theta_ej = 0.1)
Pixel   = bfn.utils.HealPixel(NSIDE = 1024)
PRESS   = bfn.utils.ConvolvedProfile(PRESS, Pixel)
PRESS   = bfn.utils.TabulatedProfile(PRESS, cosmo)

#Run the baryonification on this one shell
Runner  = bfn.Runners.BaryonifyShell(Catalog, Shell, model = model, epsilon_max = 20)
New_map = Runner.process()

#Run the profile painting on this one shell
Runner  = bfn.Runners.PaintProfilesShell(Catalog, Shell, model = PRESS, epsilon_max = 20)
New_map = Runner.process()
```

See the ```/examples``` folder for more notebooks demonstrating how to use the code for different applications.

## Attribution

If you use this code or derivatives of it, please cite [Anbajagane, Pandey & Chang 2024](https://arxiv.org/abs/2409.03822).

```bibtex
@ARTICLE{Anbajagane:2024:Baryonification,
       author = {{Anbajagane}, Dhayaa and {Pandey}, Shivam and {Chang}, Chihway},
        title = "{Map-level baryonification: Efficient modelling of higher-order correlations in the weak lensing and thermal Sunyaev-Zeldovich fields}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Cosmology and Nongalactic Astrophysics, Astrophysics - Astrophysics of Galaxies},
         year = 2024,
        month = sep,
          eid = {arXiv:2409.03822},
        pages = {arXiv:2409.03822},
archivePrefix = {arXiv},
       eprint = {2409.03822},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024arXiv240903822A},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
