
import numpy as np
import pyccl as ccl
from tqdm import tqdm

from scipy import interpolate
from astropy.cosmology import z_at_value, FlatLambdaCDM, FlatwCDM
from astropy import units as u


class TabulatedProfile(ccl.halos.profiles.HaloProfile):

    def __init__(self, model, ccl_cosmo, R_range = [1e-5, 40], N_samples = 500, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):


        self.model = model
        self.ccl_cosmo   = ccl_cosmo #CCL cosmology instance
        self.R_range     = R_range
        self.N_samples   = N_samples
        self.mass_def    = mass_def

        #Get all the other params. Particularly those
        #needed for projecting profiles
        super().__init__()


    def setup_interpolator(self, z_min = 1e-2, z_max = 5, M_min = 1e12, M_max = 1e16, N_samples_Mass = 30, N_samples_z = 30, z_linear_sampling = False):

        M_range = np.geomspace(M_min, M_max, N_samples_Mass)
        z_range = np.linspace(z_min, z_max, N_samples_z) if z_linear_sampling else np.linspace(z_min, z_max, N_samples_z)
        r    = np.geomspace(self.R_range[0], self.R_range[1], self.N_samples)
        dlnr = np.log(r[1]) - np.log(r[0])

        interp3D = np.zeros([z_range.size, M_range.size, r.size])
        interp2D = np.zeros([z_range.size, M_range.size, r.size])
        
        with tqdm(total = z_range.size, desc = 'Building Table') as pbar:
            for j in range(z_range.size):                
                a_j = 1/(1 + z_range[j])

                #Extra factor of "a" accounts for projection in ccl being done in comoving, not physical units
                interp3D[j, :, :] = self.model.real(self.ccl_cosmo, r, M_range, a_j, mass_def = self.mass_def)
                interp2D[j, :, :] = self.model.projected(self.ccl_cosmo, r, M_range, a_j, mass_def = self.mass_def) * a_j
                pbar.update(1)

        input_grid_1 = (np.log(1 + z_range), np.log(M_range), np.log(r))

        self.raw_input_3D = interp3D
        self.raw_input_2D = interp2D
        self.raw_input_z_range = np.log(1 + z_range)
        self.raw_input_M_range = np.log(M_range)
        self.raw_input_r_range = np.log(r)
        
        self.interp3D = interpolate.RegularGridInterpolator(input_grid_1, np.log(interp3D), bounds_error = False)
        self.interp2D = interpolate.RegularGridInterpolator(input_grid_1, np.log(interp2D), bounds_error = False)

        return 0


    def _real(self, cosmo, r, M, a, mass_def = None):
        
        if not (hasattr(self, 'interp3D') & hasattr(self, 'interp2D')):
            
            raise NameError("No Table created. Run setup_interpolator() method first")

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)
        a_use = np.atleast_1d(a)
        z_use = 1/a_use - 1
        
        empty = np.ones_like(r)
        z_in  = np.log(1/a)*empty
        M_in  = np.log(M)*empty

        prof = self.interp3D((z_in, M_in, np.log(r), ))
        prof = np.exp(prof)
        
        return prof
    
    
    def _projected(self, cosmo, r, M, a, mass_def = None):
        
        if not (hasattr(self, 'interp3D') & hasattr(self, 'interp2D')):
            
            raise NameError("No Table created. Run setup_interpolator() method first")

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)
        a_use = np.atleast_1d(a)
        z_use = 1/a_use - 1
        
        empty = np.ones_like(r)
        z_in  = np.log(1/a)*empty
        M_in  = np.log(M)*empty

        prof = self.interp2D((z_in, M_in, np.log(r), ))
        prof = np.exp(prof)
        
        return prof

    
class TabulatedCorrelation3D(object):

    def __init__(self):


        raise NotImplementedError("TabulatedCorrelation3D is not yet implemented.")