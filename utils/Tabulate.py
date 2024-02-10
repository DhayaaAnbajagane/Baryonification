
import numpy as np
import pyccl as ccl
from tqdm import tqdm
from itertools import product

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


    def setup_interpolator(self, z_min = 1e-2, z_max = 5, M_min = 1e12, M_max = 1e16, N_samples_Mass = 30, N_samples_z = 30, 
                           z_linear_sampling = False, verbose = False):

        M_range  = np.geomspace(M_min, M_max, N_samples_Mass)
        z_range  = np.linspace(z_min, z_max, N_samples_z) if z_linear_sampling else np.geomspace(z_min, z_max, N_samples_z)
        r        = np.geomspace(self.R_range[0], self.R_range[1], self.N_samples)
        dlnr     = np.log(r[1]) - np.log(r[0])

        interp3D = np.zeros([z_range.size, M_range.size, r.size])
        interp2D = np.zeros([z_range.size, M_range.size, r.size])
        
        with tqdm(total = z_range.size, desc = 'Building Table', disable = not verbose) as pbar:
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


    def _readout(self, r, M, a, table):
        
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)
        a_use = np.atleast_1d(a)
        z_use = 1/a_use - 1
        
        prof  = np.zeros([M_use.size, r_use.size])
        empty = np.ones_like(r_use)
        z_in  = np.log(1/a)*empty #This is log(1 + z)
        r_in  = np.log(r_use)
        
        for i in range(M_use.size):
            M_in  = np.log(M_use[i])*empty

            prof[i] = table((z_in, M_in, r_in, ))
            prof[i] = np.exp(prof[i])
            
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
            
        return prof
            
        
    def _real(self, cosmo, r, M, a, mass_def = None):
        
        if not (hasattr(self, 'interp3D') & hasattr(self, 'interp2D')):
            raise NameError("No Table created. Run setup_interpolator() method first")

        prof = self._readout(r, M, a, self.interp3D)
        
        return prof
    
    
    def _projected(self, cosmo, r, M, a, mass_def = None):
        
        if not (hasattr(self, 'interp3D') & hasattr(self, 'interp2D')):
            raise NameError("No Table created. Run setup_interpolator() method first")

        prof = self._readout(r, M, a, self.interp2D)
        
        return prof
    

    
class ParamTabulatedProfile(object):
    
    '''
    A class that takes in a profile, and then tabulates it as a function of halo mass, redshift, and then any other
    parameters that go as inputs into the profile class (i.e. any parameter that would go into an __init__ call)
    '''
    
    def __init__(self, model, ccl_cosmo, R_range = [1e-5, 40], N_samples = 500, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):


        self.model = model
        self.ccl_cosmo   = ccl_cosmo #CCL cosmology instance
        self.R_range     = R_range
        self.N_samples   = N_samples
        self.mass_def    = mass_def
        
        assert not isinstance(model, TabulatedProfile), "Input model cannot be 'TabulatedProfile' object."


    def setup_interpolator(self, z_min = 1e-2, z_max = 5, M_min = 1e12, M_max = 1e16, N_samples_Mass = 30, N_samples_z = 30, 
                           z_linear_sampling = False, verbose = False, other_params = {}):

        M_range  = np.geomspace(M_min, M_max, N_samples_Mass)
        z_range  = np.linspace(z_min, z_max, N_samples_z) if z_linear_sampling else np.geomspace(z_min, z_max, N_samples_z)
        r        = np.geomspace(self.R_range[0], self.R_range[1], self.N_samples)
        dlnr     = np.log(r[1]) - np.log(r[0])

        p_keys   = list(other_params.keys())
        interp3D = np.zeros([z_range.size, M_range.size, r.size] + [other_params[k].size for k in p_keys]) + np.NaN
        interp2D = np.zeros([z_range.size, M_range.size, r.size] + [other_params[k].size for k in p_keys]) + np.NaN

        #If other_params is empty then iterator will be empty and the code still works fine
        iterator = [p for p in product(*[np.arange(other_params[k].size) for k in p_keys])]
        
        #Loop over params to build table
        with tqdm(total = interp3D.size//(M_range.size*r.size), desc = 'Building Table', disable = not verbose) as pbar:
            for j in range(z_range.size):                
                a_j = 1/(1 + z_range[j])
                
                for c in iterator:
                    
                    #Modify the model input params so that they contain this
                    for k_i in range(len(p_keys)):
                        setattr(self.model, p_keys[k_i], other_params[p_keys[k_i]][c[k_i]])
                        
                        #Need this so composite models (like DarkMatterOnly, CollisionlessMatter)
                        #also have the correct modifications made to the submodels that are passed into them
                        model_keys = list(self.model.__dict__.keys())
                        for d_i in model_keys:
                            if isinstance(getattr(self.model, d_i), ccl.halos.profiles.HaloProfile):
                                setattr(getattr(self.model, d_i), p_keys[k_i], other_params[p_keys[k_i]][c[k_i]])
                    
                    #Build a custom index into the array
                    index = tuple([j, slice(None), slice(None)] + list(c))
                    
                    #Extra factor of "a" accounts for projection in ccl being done in comoving, not physical units
                    interp3D[index] = self.model.real(self.ccl_cosmo, r, M_range, a_j, mass_def = self.mass_def)
                    interp2D[index] = self.model.projected(self.ccl_cosmo, r, M_range, a_j, mass_def = self.mass_def) * a_j
                    pbar.update(1)
                    

        input_grid_1 = tuple([np.log(1 + z_range), np.log(M_range), np.log(r)] + [other_params[k] for k in p_keys])

        self.raw_input_3D = interp3D
        self.raw_input_2D = interp2D
        self.raw_input_z_range = np.log(1 + z_range)
        self.raw_input_M_range = np.log(M_range)
        self.raw_input_r_range = np.log(r)
        for k in other_params.keys(): setattr(self, 'raw_input_%s_range' % k, other_params[k]) #Save other raw inputs too
        
        self.interp3D = interpolate.RegularGridInterpolator(input_grid_1, np.log(interp3D), bounds_error = False)
        self.interp2D = interpolate.RegularGridInterpolator(input_grid_1, np.log(interp2D), bounds_error = False)

        return 0


    def _readout(self, r, M, a, table, **kwargs):
        
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)
        a_use = np.atleast_1d(a)
        z_use = 1/a_use - 1
        
        prof  = np.zeros([M_use.size, r_use.size])
        empty = np.ones_like(r_use)
        z_in  = np.log(1/a)*empty #This is log(1 + z)
        r_in  = np.log(r_use)
        k_in  = [kwargs[k] * empty for k in kwargs.keys()]
        
        for i in range(M_use.size):
            M_in  = np.log(M_use[i])*empty
            p_in  = tuple([z_in, M_in, r_in, ] + k_in)
            prof[i] = table(p_in)
            prof[i] = np.exp(prof[i])
            
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
            
        return prof
    
            
    def real(self, cosmo, r, M, a, mass_def = None, **kwargs):
        
        if not (hasattr(self, 'interp3D') & hasattr(self, 'interp2D')):
            raise NameError("No Table created. Run setup_interpolator() method first")

        prof = self._readout(r, M, a, self.interp3D, **kwargs)
        
        return prof
    
    
    def projected(self, cosmo, r, M, a, mass_def = None, **kwargs):
        
        if not (hasattr(self, 'interp3D') & hasattr(self, 'interp2D')):
            raise NameError("No Table created. Run setup_interpolator() method first")

        prof = self._readout(r, M, a, self.interp2D, **kwargs)
        
        return prof


class TabulatedCorrelation3D(object):

    def __init__(self):


        raise NotImplementedError("TabulatedCorrelation3D is not yet implemented.")