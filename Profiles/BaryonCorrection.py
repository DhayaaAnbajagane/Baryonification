
import numpy as np
import pyccl as ccl
from tqdm import tqdm
from scipy import interpolate


class BaryonificationClass(object):

    def __init__(self, DMO, DMB, ccl_cosmo, N_samples = 500, epsilon_max = 20, use_concentration = False,
                 mass_def = ccl.halos.massdef.MassDef(200, 'critical')):

        self.DMO = DMO
        self.DMB = DMB
        
        #Set cutoff to 1 Gpc for calculation, assuming profiles are negligible beyond that
        #Smaller cutoffs result in asymptotic value problems at large scales
        #Larger cutoffs lead to numerical divergence during FFTLogs
        #The user supplied cutoffs will be places when implementing cutoffs in data
        self.DMO.set_parameter('cutoff', 1000)
        self.DMB.set_parameter('cutoff', 1000)
        
        self.ccl_cosmo   = ccl_cosmo #CCL cosmology instance
        self.epsilon_max = epsilon_max
        self.N_samples   = N_samples
        self.mass_def    = mass_def
        self.use_concentration = use_concentration


    def get_masses(self, model, r, M, a, mass_def):

        raise NotImplementedError("Implement a get_masses() method first")


    def setup_interpolator(self, 
                           z_min = 1e-2, z_max = 5, N_samples_z = 30, 
                           M_min = 1e12, M_max = 1e16, N_samples_Mass = 30, 
                           R_min = 1e-3, R_max = 1e2, N_samples_R = 100, 
                           c_min = 1, c_max = 20, N_samples_c = 30, 
                           z_linear_sampling = False, verbose = True):

        M_range = np.geomspace(M_min, M_max, N_samples_Mass)
        r       = np.geomspace(R_min, R_max, N_samples_R)
        z_range = np.linspace(z_min, z_max, N_samples_z) if z_linear_sampling else np.geomspace(z_min, z_max, N_samples_z)
        c_range = np.linspace(c_min, c_max, N_samples_c)
        dlnr    = np.log(r[1]) - np.log(r[0])

        if not self.use_concentration: c_range = np.zeros(1)
        d_interp = np.zeros([z_range.size, M_range.size, c_range.size, r.size])
        
        with tqdm(total = z_range.size * c_range.size, desc = 'Building Table', disable = not verbose) as pbar:
            for j in range(z_range.size):
                
                for k in range(c_range.size):
                    
                    if self.use_concentration:
                        self.DMO.cdelta, self.DMB.cdelta = c_range[k], c_range[k]
                    else:
                        assert self.DMO.cdelta is None, "use_concentration = False, so set DMO model to have cdelta = None"
                        assert self.DMB.cdelta is None, "use_concentration = False, so set DMB model to have cdelta = None"
                    
                    M_DMO = self.get_masses(self.DMO, r, M_range, 1/(1 + z_range[j]), mass_def = self.mass_def)
                    M_DMB = self.get_masses(self.DMB, r, M_range, 1/(1 + z_range[j]), mass_def = self.mass_def)

                    for i in range(M_range.size):
                        ln_DMB    = np.log(M_DMB[i])
                        ln_DMO    = np.log(M_DMO[i])
                        
                        #Require mass to always increase w/ radius
                        #And remove pts of DMO = DMB, improves large-scale convergence
                        #And require at least 1e-6 difference else the interpolator breaks :/
                        diff_mask = (np.diff(ln_DMB, prepend = 0) > 1e-5) & (np.diff(ln_DMO, prepend = 0) > 1e-5) 
                        diff_mask = diff_mask & (np.abs(ln_DMB - ln_DMO) > 1e-6) 
                                   
                        interp_DMB = interpolate.CubicSpline(ln_DMB[diff_mask], np.log(r)[diff_mask], extrapolate = False)
                        interp_DMO = interpolate.CubicSpline(np.log(r)[diff_mask], ln_DMO[diff_mask], extrapolate = False)
                        
                        offset = np.exp(interp_DMB(interp_DMO(np.log(r)))) - r
                        offset = np.where(np.isfinite(offset), offset, 0)
                        
                        d_interp[j, i, k, :] = offset
                            
                    pbar.update(1)


        if self.use_concentration:
            input_grid_1 = (np.log(1 + z_range), np.log(M_range), np.log(c_range), np.log(r))
        else:
            input_grid_1 = (np.log(1 + z_range), np.log(M_range), np.log(r))
            
            #Also squeeze the output to remove the redundant axis, since we don't extend in 
            #the concentration direction anymore
            d_interp = np.squeeze(d_interp, axis = 2)
        
        self.interp_d = interpolate.RegularGridInterpolator(input_grid_1, d_interp, bounds_error = False, fill_value = np.NaN)        

        return 0


    def displacements(self, r, M, a, c = None):
        
        if not hasattr(self, 'interp_d'):
            raise NameError("No Table created. Run setup_interpolator() method first")

        #Commenting out for now
        #bounds = np.all((self.R_range[0] < r) & (self.R_range[1] > r))
        #assert bounds, "Input x has limits (%0.2e, %0.2e). Rerun setup_interpolatr() with R_range = (r_min, r_max)" % (np.min(r), np.max(r)) 
        
        if self.use_concentration:
            assert c is not None, f"You asked for model to be built with concentration. But you set c = {c}"
            c_use = np.atleast_1d(c)
            
        offset = np.zeros_like(r)
        R      = self.mass_def.get_radius(self.ccl_cosmo, np.atleast_1d(M), a)/a #in comoving Mpc
        inside = (r < self.epsilon_max*R)

        r = r[inside]
        
        ones   = np.ones_like(r)
        z      = 1/a - 1
        z_in   = np.log(1 + z)*ones
        M_in   = np.log(M)*ones
        

        if self.use_concentration:
            c_in = np.log(c)*ones
            offset[inside] = self.interp_d((z_in, M_in, c_in, np.log(r), ))

        else:
            offset[inside] = self.interp_d((z_in, M_in, np.log(r), ))
            
        return offset


class Baryonification3D(BaryonificationClass):

    def get_masses(self, model, r, M, a, mass_def):
        
        #Make sure the min/max does not mess up the integral
        #Adding some 20% buffer just in case
        r_min = np.min([np.min(r), 1e-6])
        r_max = np.max([np.max(r), 1000])
        r_int = np.geomspace(r_min/1.2, r_max*1.2, 500)
        
        dlnr  = np.log(r_int[1]/r_int[0])
        rho   = model.real(self.ccl_cosmo, r_int, M, a, mass_def = mass_def)
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
            M_f[M_i] = np.exp( interpolate.CubicSpline(np.log(r_int)[Mask], np.log(M_enc[M_i])[Mask], extrapolate = False)(lnr) )
        
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
        Sigma = model.projected(self.ccl_cosmo, r_int, M, a, mass_def = mass_def) * a 
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
            M_f[M_i] = np.exp( interpolate.CubicSpline(np.log(r_int)[Mask], np.log(M_enc[M_i])[Mask], extrapolate = False)(lnr) )
        
        if isinstance(M, (float, int) ): M_f = np.squeeze(M_f, axis = 0)
            
        return M_f
