
import numpy as np
import pyccl as ccl
import healpy as hp
import os

from scipy import interpolate
from astropy.cosmology import z_at_value, FlatLambdaCDM, FlatwCDM
from astropy import units as u
from astropy.io import fits

from ..Profiles import DarkMatterOnly, DarkMatterBaryon, Baryonification2D, Baryonification3D, Pressure
from ..utils.io import HaloNDCatalog, GriddedMap

from tqdm import tqdm

MY_FILL_VAL = np.NaN


class DefaultRunnerGrid(object):
    '''
    A class that contains relevant utils for input/output
    '''
    
    def __init__(self, HaloNDCatalog, GriddedMap, config, model = None,
                 mass_def = ccl.halos.massdef.MassDef(200, 'critical'), verbose = True):

        self.HaloNDCatalog = HaloNDCatalog
        self.GriddedMap    = GriddedMap
        self.cosmo = HaloNDCatalog.cosmology()
        self.model = model
        
        
        self.mass_def = mass_def
        self.verbose  = verbose
        
        self.config   = self.set_config(config)


    def set_config(self, config):

        #Dictionary to hold all the params
        out = {}

        out['OutPath']   = config.get('OutPath', None)
        out['Name']      = config.get('Name', '')

        out['epsilon']   = config.get('epsilon',  4.0)  #Truncation radius for DM NFW profile normalized by R200c
        out['theta_ej']  = config.get('theta_ej', 4.0)  #Radius up to which gas is ejected, normalized by R200c
        out['theta_co']  = config.get('theta_co', 0.1)  #Radius within which gas is fully collapsed/bound, normalized by R200c
        out['M_c']       = config.get('M_c',      2e14) #in Msun, normalization of the relation between gas profile slope and mass
        out['mu']        = config.get('mu',       0.4)  #slope of the relation between gas profile slope and mass
        out['eta_star']  = config.get('eta_star', 0.3)  #
        out['eta_cga']   = config.get('eta_star', 0.6)
        out['A']         = config.get('A',        0.09)
        out['M1']        = config.get('M1',       3e11) #in Msun
        out['epsilon_h'] = config.get('epsilon_h', 0.015)
        out['a']         = config.get('a', 0.3)
        out['n']         = config.get('n', 2.0)
        out['p']         = config.get('p', 0.3)
        out['q']         = config.get('q', 0.707)

        out['epsilon_max_Cutout'] = config.get('epsilon_max_Cutout', 5)
        out['epsilon_max_Offset'] = config.get('epsilon_max_Offset', 5)

        if self.verbose:
            #Print args for debugging state
            print('-------UPDATING INPUT PARAMS----------')
            for p in out.keys():
                print('%s : %s'%(p.upper(), out[p]))
            print('-----------------------------')
            print('-----------------------------')

        return out
    
    
    def output(self, X):
        
        if isinstance(self.config['OutPath'], str):

            path_ = self.config['OutPath']
            hdu   = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(X)])
            hdu.writeto(path_, overwrite = True)
                        
            if self.verbose: print("WRITING TO ", path_)
            
        else:
            
            if self.verbose: print("OutPath is not string. Map is not saved to disk")
    
    

class BaryonifyGrid(DefaultRunnerGrid):

    def process(self):

        
        cosmo = ccl.Cosmology(Omega_c = self.cosmo['Omega_m'] - self.cosmo['Omega_b'],
                              Omega_b = self.cosmo['Omega_b'], h = self.cosmo['h'],
                              sigma8  = self.cosmo['sigma8'],  n_s = self.cosmo['n_s'],
                              matter_power_spectrum = 'linear')
        cosmo.compute_sigma()

        orig_map = self.GriddedMap.map
        new_map  = orig_map.copy()
        bins     = self.GriddedMap.bins

        if self.model is None:


            DMO = DarkMatterOnly(epsilon = self.config['epsilon'],
                                 q = self.config['q'], p = self.config['p'], xi_mm = None, R_range = [1e-5, 40])

            DMB = DarkMatterBaryon(epsilon = self.config['epsilon'], a = self.config['a'], n = self.config['n'],
                                   theta_ej = self.config['theta_ej'], theta_co = self.config['theta_co'],
                                   M_c = self.config['M_c'], mu = self.config['mu'],
                                   A = self.config['A'], M1 = self.config['M1'], epsilon_h = self.config['epsilon_h'],
                                   eta_star = self.config['eta_star'], eta_cga = self.config['eta_cga'],
                                   q = self.config['q'], p = self.config['p'], xi_mm = None, R_range = [1e-5, 40])

            model   = Baryonification2D if GriddedMap.is2D else Baryonification3D
            Baryons = model(DMO = DMO, DMB = DMB, R_range = [1e-5, 50], N_samples = 500,
                            epsilon_max = self.config['epsilon_max_Offset'])

        else:

            Baryons = self.model


        for j in tqdm(range(self.HaloNDCatalog.cat.size), desc = 'Baryonifying matter', disable = not self.verbose):

            M_j = self.HaloNDCatalog.cat['M'][j]
            x_j = self.HaloNDCatalog.cat['x'][j]
            y_j = self.HaloNDCatalog.cat['y'][j]
            z_j = self.HaloNDCatalog.cat['z'][j] #THIS IS A CARTESIAN COORDINATE, NOT REDSHIFT

            a_j = 1/(1 + self.HaloNDCatalog.redshift)
            R_j = self.mass_def.get_radius(cosmo, M_j, a_j) #in physical Mpc
            
            res    = self.GriddedMap.res
            Nsize  = 2 * self.config['epsilon_max_Cutout'] * R_j / res
            Nsize  = int(Nsize // 2)*2 #Force it to be even
            
            if Nsize < 2:
                continue

            x  = np.linspace(-Nsize/2, Nsize/2, Nsize) * res
            pixel_width = Nsize//2

            if self.GriddedMap.is2D:
                x_grid, y_grid = np.meshgrid(x, x, indexing = 'xy')
                r_grid = np.sqrt(x_grid**2 + y_grid**2)

                x_hat = x_grid/r_grid
                y_hat = y_grid/r_grid

                cen          = (np.argmin(np.abs(bins - x_j)), np.argmin(np.abs(bins - y_j)))
                slices       = (slice(cen[0] - pixel_width, cen[0] + pixel_width),  
                                slice(cen[1] - pixel_width, cen[1] + pixel_width))
                map_cutout = map[slices]

                interp_map = interpolate.RegularGridInterpolator((x, x), map_cutout.T, bounds_error = False, fill_value = MY_FILL_VAL)

                #Compute the displacement needed
                offset     = Baryons.displacements(r_grid.flatten()/a_j, M_j, a_j).reshape(r_grid.shape) * a_j
            
                in_coords  = np.vstack([(x_grid + offset*x_hat).flatten(), (y_grid + offset*y_hat).flatten()]).T
                
            
            else:
                x_grid, y_grid, z_grid = np.meshgrid(x, x, x, indexing = 'xy')
                r_grid = np.sqrt(x_grid**2 + y_grid**2 + z_grid**2)

                x_hat = x_grid/r_grid
                y_hat = y_grid/r_grid
                z_hat = z_grid/r_grid

                cen          = (np.argmin(np.abs(bins - x_j)), np.argmin(np.abs(bins - y_j)), np.argmin(np.abs(bins - z_j)))
                slices       = (slice(cen[0] - pixel_width, cen[0] + pixel_width),  
                                slice(cen[1] - pixel_width, cen[1] + pixel_width),
                                slice(cen[2] - pixel_width, cen[2] + pixel_width))
                map_cutout = map[slices]

                interp_map = interpolate.RegularGridInterpolator((x, x, x), map_cutout.T, bounds_error = False, fill_value = MY_FILL_VAL)

                #Compute the displacement needed
                offset     = Baryons.displacements(r_grid.flatten()/a_j, M_j, a_j).reshape(r_grid.shape) * a_j
            
                in_coords  = np.vstack([(x_grid + offset*x_hat).flatten(), 
                                        (y_grid + offset*y_hat).flatten(),
                                        (z_grid + offset*z_hat).flatten()]).T
            
            modded_map = interp_map(in_coords)
            mask       = np.isfinite(modded_map) #Find which part of map cannot be modified due to out-of-bounds errors
        
            if mask.sum() == 0: continue
        
            mass_offsets        = np.where(mask, modded_map - map_cutout.flatten(), 0) #Set those offsets to 0
            mass_offsets[mask] -= np.mean(mass_offsets[mask]) #Enforce mass conservation by making sure total mass moving around is 0

            #Add the offsets to the new healpix map
            new_map[slices] += mass_offsets.reshape(map_cutout.shape)

        self.output(new_map)

        return new_map


class PaintThermalSZGrid(DefaultRunnerGrid):

    def process(self):

        assert self.GriddedMap.is2D == True, "Can only paint tSZ on 2D maps. You have passed a 3D Map"

        cosmo = ccl.Cosmology(Omega_c = self.cosmo['Omega_m'] - self.cosmo['Omega_b'],
                              Omega_b = self.cosmo['Omega_b'], h = self.cosmo['h'],
                              sigma8  = self.cosmo['sigma8'],  n_s = self.cosmo['n_s'],
                              matter_power_spectrum = 'linear')
        cosmo.compute_sigma()

        orig_map = self.GriddedMap.map
        new_map  = np.zeros_like(orig_map).astype(np.float64)

        if self.model is None:

            if self.verbose: print("No model provided. We are using a Pressure Model")
            Baryons = Pressure(epsilon = self.config['epsilon'], a = self.config['a'], n = self.config['n'],
                               theta_ej = self.config['theta_ej'], theta_co = self.config['theta_co'],
                               M_c = self.config['M_c'], mu = self.config['mu'],
                               A = self.config['A'], M1 = self.config['M1'], epsilon_h = self.config['epsilon_h'],
                               eta_star = self.config['eta_star'], eta_cga = self.config['eta_cga'],
                               q = self.config['q'], p = self.config['p'], xi_mm = None, R_range = [1e-5, 40])
        else:

            Baryons = self.model


        for j in tqdm(range(self.HaloNDCatalog.cat.size), desc = 'Painting SZ', disable = not self.verbose):

            M_j = self.HaloNDCatalog.cat['M'][j]
            x_j = self.HaloNDCatalog.cat['x'][j]
            y_j = self.HaloNDCatalog.cat['y'][j]

            a_j = 1/(1 + self.HaloNDCatalog.redshift)
            R_j = self.mass_def.get_radius(cosmo, M_j, a_j) #in physical Mpc
            
            res    = self.GriddedMap.res
            dA     = res**2
            Nsize  = 2 * self.config['epsilon_max_Cutout'] * R_j / res
            Nsize  = int(Nsize // 2)*2 #Force it to be even
            
            if Nsize < 2:
                continue

            x  = np.linspace(-Nsize/2, Nsize/2, Nsize) * res
            pixel_width = Nsize//2

            x_grid, y_grid = np.meshgrid(x, x, indexing = 'xy')
            r_grid = np.sqrt(x_grid**2 + y_grid**2)

            cen          = (np.argmin(np.abs(bins - x_j)), np.argmin(np.abs(bins - y_j)))
            slices       = (slice(cen[0] - pixel_width, cen[0] + pixel_width),  
                            slice(cen[1] - pixel_width, cen[1] + pixel_width))
           
            #Compute the integrated SZ effect
            tSZ = Baryons.projected(cosmo, r_grid.flatten()/a_j, M_j, a_j) * dA
            
            mask = np.isfinite(tSZ) #Find which part of map cannot be modified due to out-of-bounds errors
            if mask.sum() == 0: continue
                
            tSZ  = np.where(mask, tSZ, 0) #Set those tSZ values to 0
            
            #Add the pressure to the new grid map
            new_map[slices] += tSZ.reshape(r_grid.shape)        

        self.output(new_map)

        return new_map
