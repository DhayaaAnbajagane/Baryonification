
import numpy as np
import pyccl as ccl
from tqdm import tqdm
from scipy import interpolate
import warnings
from itertools import product

from ..utils.Tabulate import _set_parameter


class BaryonificationClass(object):

    def __init__(self, DMO, DMB, cosmo, N_samples = 500, epsilon_max = 20, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):

        self.DMO = DMO
        self.DMB = DMB
        
        #Set cutoff to 1 Gpc for calculation, assuming profiles are negligible beyond that
        #Smaller cutoffs result in asymptotic value problems at large scales
        #Larger cutoffs lead to numerical divergence during FFTLogs
        #The user supplied cutoffs will be places when implementing cutoffs in data
        #NOTE: We have not altered the PROJECTED cutoff, only the real cutoff.
        #Projected cutoff must be specified to user input at all times.
        self.DMO.set_parameter('cutoff', 1000)
        self.DMB.set_parameter('cutoff', 1000)
        
        self.cosmo       = cosmo #CCL cosmology instance
        self.epsilon_max = epsilon_max
        self.N_samples   = N_samples
        self.mass_def    = mass_def


    def get_masses(self, model, r, M, a, mass_def):

        raise NotImplementedError("Implement a get_masses() method first")


    def setup_interpolator(self, 
                           z_min = 1e-2, z_max = 5, N_samples_z = 30, z_linear_sampling = False, 
                           M_min = 1e12, M_max = 1e16, N_samples_Mass = 30, 
                           R_min = 1e-3, R_max = 1e2, N_samples_R = 100, 
                           other_params = {}, verbose = True):

        M_range  = np.geomspace(M_min, M_max, N_samples_Mass)
        r        = np.geomspace(R_min, R_max, N_samples_R)
        z_range  = np.linspace(z_min, z_max, N_samples_z) if z_linear_sampling else np.geomspace(z_min, z_max, N_samples_z)
        p_keys   = list(other_params.keys()); setattr(self, 'p_keys', p_keys)
        d_interp = np.zeros([z_range.size, M_range.size, r.size] + [other_params[k].size for k in p_keys])
        dlnr     = np.log(r[1]) - np.log(r[0])
        
        #If other_params is empty then iterator will be empty and the code still works fine
        iterator = [p for p in product(*[np.arange(other_params[k].size) for k in p_keys])]
        
        with tqdm(total = d_interp.size//(M_range.size*r.size), desc = 'Building Table', disable = not verbose) as pbar:
            for j in range(z_range.size):
                
                for c in iterator:
                    
                    #Modify the model input params so that they are run with the right parameters
                    for k_i in range(len(p_keys)):
                        _set_parameter(self.DMO, p_keys[k_i], other_params[p_keys[k_i]][c[k_i]])
                        _set_parameter(self.DMB, p_keys[k_i], other_params[p_keys[k_i]][c[k_i]])
                    
                    
                    M_DMO = self.get_masses(self.DMO, r, M_range, 1/(1 + z_range[j]), mass_def = self.mass_def)
                    M_DMB = self.get_masses(self.DMB, r, M_range, 1/(1 + z_range[j]), mass_def = self.mass_def)
                    
                    for i in range(M_range.size):
                        ln_DMB    = np.log(M_DMB[i])
                        ln_DMO    = np.log(M_DMO[i])
                        
                        #Require mass to always increase w/ radius
                        #And remove pts of DMO = DMB, improves large-scale convergence
                        #And require at least 1e-6 difference else the interpolator breaks :/
                        
                        min_diff  = -np.inf
                        diff_mask = np.ones_like(ln_DMB).astype(bool)
                        iterate   = 0
                        while (min_diff < 1e-5) & (diff_mask.sum() > 5):
                            
                            new_mask  = ( (np.diff(ln_DMB[diff_mask], prepend = 0) > 1e-5) & 
                                          (np.diff(ln_DMO[diff_mask], prepend = 0) > 1e-5) & 
                                          (np.abs(ln_DMB - ln_DMO)[diff_mask] > 1e-6) 
                                        )
                            
                            diff_mask[diff_mask] = new_mask
                            diff_mask[0] = True
                            
                            iterate += 1
                            
                            if iterate > 30:
                                diff_mask  = np.zeros_like(diff_mask).astype(bool) #Set everything to False and skip the building step next
                                warn_text  = (f"Mass profile of log10(M) = {np.log10(M_range[i])} is nearly constant over radius. " 
                                              "Suggests density is negative or zero for most of the range. If using convolutions,"
                                              "consider changing the fft precision params in the CCL profile:"
                                              "padding_lo_fftlog, padding_hi_fftlog, or n_per_decade")
                                warnings.warn(warn_text, UserWarning)
                                break
                                
                            if diff_mask.sum() < 5: 
                                warn_text  = (f"Mass profile of log10(M) = {np.log10(M_range[i])} is nearly constant over radius. " 
                                              "Or it is broken. Less than 5 datapoints are usable.")
                                warnings.warn(warn_text, UserWarning)
                                break
                            
                            min_diff  = np.min([np.min(np.diff(ln_DMB[diff_mask], prepend = 0)[1:]),
                                                np.min(np.diff(ln_DMO[diff_mask], prepend = 0)[1:])
                                               ])
                            
                            
                                                                
                            
                        #If we have enough usable mass values, then proceed as usual
                        #This generally breaks for very small halos, where projection
                        #can be catastrophicall broken (eg. only negative densities)
                        if diff_mask.sum() > 5:
                                   
                            interp_DMB = interpolate.PchipInterpolator(ln_DMB[diff_mask], np.log(r)[diff_mask], extrapolate = False)
                            interp_DMO = interpolate.PchipInterpolator(np.log(r)[diff_mask], ln_DMO[diff_mask], extrapolate = False)

                            offset = np.exp(interp_DMB(interp_DMO(np.log(r)))) - r
                            offset = np.where(np.isfinite(offset), offset, 0)
                        
                        #If broken, then these halos contribute nothing to the displacement function.
                        #Just provide a warning saying this is happening
                        else:
                            offset = np.zeros_like(r)
                            warn_text = (f"Displacement function for halo with log10(M) = {np.log10(M_range[i])} failed to compute." 
                                         "Defaulting to d = 0. Consider changing the fft precision params in the CCL profile:"
                                         "padding_lo_fftlog, padding_hi_fftlog, or n_per_decade")
                            warnings.warn(warn_text, UserWarning)
                        
                        #Build a custom index into the array
                        index = tuple([j, i, slice(None)] + list(c))
                        d_interp[index] = offset
                            
                    pbar.update(1)


        input_grid = tuple([np.log(1 + z_range), np.log(M_range), np.log(r)] + [other_params[k] for k in p_keys])

        self.raw_input_d = d_interp
        self.raw_input_z_range = np.log(1 + z_range)
        self.raw_input_M_range = np.log(M_range)
        self.raw_input_r_range = np.log(r)
        for k in other_params.keys(): setattr(self, 'raw_input_%s_range' % k, other_params[k]) #Save other raw inputs too
            
        self.interp_d = interpolate.RegularGridInterpolator(input_grid, d_interp, bounds_error = False, fill_value = np.NaN)        

        return 0

    
    def _readout(self, r, M, a, **kwargs):
        
        table = self.interp_d #The interpolation table to use
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)
        a_use = np.atleast_1d(a)
        z_use = 1/a_use - 1
        
        displ = np.zeros([M_use.size, r_use.size])
        empty = np.ones_like(r_use)
        z_in  = np.log(1/a)*empty #This is log(1 + z)
        r_in  = np.log(r_use)
        k_in  = [kwargs[k] * empty for k in kwargs.keys()]
        
        for i in range(M_use.size):
            M_in  = np.log(M_use[i])*empty
            p_in  = tuple([z_in, M_in, r_in] + k_in)
            displ[i] = table(p_in)
            
            R        = self.mass_def.get_radius(self.cosmo, np.atleast_1d(M), a)/a #in comoving Mpc
            inside   = (r < self.epsilon_max*R)
            displ[i] = np.where(inside, displ, 0) #Set large-scale displacements to 0
            
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            displ = np.squeeze(displ, axis=-1)
        if np.ndim(M) == 0:
            displ = np.squeeze(displ, axis=0)
            
        return displ

    
    def displacement(self, r, M, a, **kwargs):
        
        if not hasattr(self, 'interp_d'):
            raise NameError("No Table created. Run setup_interpolator() method first")
            
        for k in self.p_keys:
            assert k in kwargs.keys(), "Need to provide %s as input into `displacement'. Table was built with this." % k
        
        return self._readout(r, M, a, **kwargs)



class Baryonification3D(BaryonificationClass):

    def get_masses(self, model, r, M, a, mass_def):
        
        #Make sure the min/max does not mess up the integral
        #Adding some 20% buffer just in case
        r_min = np.min([np.min(r), 1e-6])
        r_max = np.max([np.max(r), 1000])
        r_int = np.geomspace(r_min/1.2, r_max*1.2, 500)
        
        dlnr  = np.log(r_int[1]/r_int[0])
        rho   = model.real(self.cosmo, r_int, M, a, mass_def = mass_def)
        rho   = np.where(rho < 0, 0, rho) #Enforce non-zero densities
        
        if isinstance(M, (float, int) ): rho = rho[None, :]
            
        M_enc = np.cumsum(4*np.pi*r_int**3 * rho * dlnr, axis = -1)
        lnr   = np.log(r)
        
        M_f   = np.zeros([M_enc.shape[0], r.size])
        
        #Remove datapoints in profile where rho == 0 and then just interpolate
        #across them. This helps deal with ringing profiles due to 
        #fourier space issues, where profile could go negative sometimes
        for M_i in range(M_enc.shape[0]):
            Mask     = (rho[M_i] > 0) & (np.isfinite(M[M_i])) #Keep only finite points, and ones with increasing density
            M_f[M_i] = np.exp( interpolate.PchipInterpolator(np.log(r_int)[Mask], np.log(M_enc[M_i])[Mask], extrapolate = False)(lnr) )
        
        if isinstance(M, (float, int) ): M_f = np.squeeze(M_f, axis = 0)
            
        return M_f


class Baryonification2D(BaryonificationClass):

    def get_masses(self, model, r, M, a, mass_def):
        
        #Make sure the min/max does not mess up the integral
        #Adding some 20% buffer just in case
        r_min = np.min([np.min(r), 1e-6])
        r_max = np.max([np.max(r), 1000])
        r_int = np.geomspace(r_min/1.5, r_max*1.5, 500)
        
        #The scale fac. is used in Sigma cause the projection in ccl is
        #done in comoving coords not physical coords
        dlnr  = np.log(r_int[1]/r_int[0])
        Sigma = model.projected(self.cosmo, r_int, M, a, mass_def = mass_def) * a 
        Sigma = np.where(Sigma < 0, 0, Sigma) #Enforce non-zero densities
        
        if isinstance(M, (float, int) ): Sigma = Sigma[None, :]
        
        M_enc = np.cumsum(2*np.pi*r_int**2 * Sigma * dlnr, axis = -1)
        lnr   = np.log(r)
        
        
        M_f  = np.zeros([M_enc.shape[0], r.size])
        #Remove datapoints in profile where Sigma == 0 and then just interpolate
        #across them. This helps deal with ringing profiles due to 
        #fourier space issues, where profile could go negative sometimes
        for M_i in range(M_enc.shape[0]):
            Mask     = (Sigma[M_i] > 0) & (np.isfinite(M_enc[M_i])) #Keep only finite points, and ones with increasing density
            M_f[M_i] = np.exp( interpolate.PchipInterpolator(np.log(r_int)[Mask], np.log(M_enc[M_i])[Mask], extrapolate = False)(lnr) )
        
        if isinstance(M, (float, int) ): M_f = np.squeeze(M_f, axis = 0)
            
        return M_f
