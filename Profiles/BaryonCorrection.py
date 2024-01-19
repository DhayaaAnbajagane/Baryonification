
import numpy as np
import pyccl as ccl
from tqdm import tqdm
from scipy import interpolate


class BaryonificationClass(object):

    def __init__(self, DMO, DMB, ccl_cosmo, R_range = [1e-5, 200], N_samples = 500, epsilon_max = 20, use_concentration = False,
                 mass_def = ccl.halos.massdef.MassDef(200, 'critical')):

        self.DMO = DMO
        self.DMB = DMB
        self.ccl_cosmo   = ccl_cosmo #CCL cosmology instance
        self.R_range     = R_range
        self.epsilon_max = epsilon_max
        self.N_samples   = N_samples
        self.mass_def    = mass_def
        self.use_concentration = use_concentration


    def get_masses(self, model, r, M, a, mass_def):

        raise NotImplementedError("Implement a get_masses() method first")


    def setup_interpolator(self, z_min = 1e-2, z_max = 5, M_min = 1e12, M_max = 1e16, c_min = 1, c_max = 20,
                           N_samples_Mass = 30, N_samples_z = 30, N_samples_c = 30, 
                           z_linear_sampling = False, verbose = True):

        M_range = np.geomspace(M_min, M_max, N_samples_Mass)
        z_range = np.linspace(z_min, z_max, N_samples_z) if z_linear_sampling else np.geomspace(z_min, z_max, N_samples_z)
        c_range = np.linspace(c_min, c_max, N_samples_c)
        r       = np.geomspace(self.R_range[0], self.R_range[1], self.N_samples)
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
                    
                    #Extra factor of "a" accounts for projection in ccl being done in comoving, not physical units
                    M_DMO = self.get_masses(self.DMO, r, M_range, 1/(1 + z_range[j]), mass_def = self.mass_def)
                    M_DMB = self.get_masses(self.DMB, r, M_range, 1/(1 + z_range[j]), mass_def = self.mass_def)

                    for i in range(M_range.size):
                        ln_DMB    = np.log(M_DMB[i])
                        ln_DMO    = np.log(M_DMO[i])
                        diff_mask = np.ones(len(M_DMB[i]), dtype = bool)
                        diff_mask[1:] = np.invert(np.isclose(np.diff(ln_DMB), 0, atol = 1e-10))
                        
                        interp_DMB = interpolate.CubicSpline(ln_DMB[diff_mask], np.log(r)[diff_mask], extrapolate = False)
                        interp_DMO = interpolate.CubicSpline(np.log(r), ln_DMO, extrapolate = False)
                        
                        d_interp[j, i, k, :] = np.exp(interp_DMB(interp_DMO(np.log(r)))) - r
                            
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


    def displacements(self, x, M, a, c = None):
        
        if not hasattr(self, 'interp_d'):
            raise NameError("No Table created. Run setup_interpolator() method first")

#         bounds = np.all((self.R_range[0] < x) & (self.R_range[1] > x))
#         assert bounds, "Input x has limits (%0.2e, %0.2e). Rerun setup_interpolatr() with R_range = (x_min, x_max)" % (np.min(x), np.max(x)) 
        
        if self.use_concentration:
            assert c is not None, f"You asked for model to be built with concentration. But you set c = {c}"
            c_use = np.atleast_1d(c)
            
        offset = np.zeros_like(x)
        R      = self.mass_def.get_radius(self.ccl_cosmo, np.atleast_1d(M), a)/a #in comoving Mpc
        inside = (x > self.R_range[0]) & (x < self.epsilon_max*R)

        x = x[inside]
        
        ones   = np.ones_like(x)
        z      = 1/a - 1
        z_in   = np.log(1 + z)*ones
        M_in   = np.log(M)*ones
        

        if self.use_concentration:
            c_in = np.log(c)*ones
            offset[inside] = self.interp_d((z_in, M_in, c_in, np.log(x), ))

        else:
            offset[inside] = self.interp_d((z_in, M_in, np.log(x), ))
            
        return offset


class Baryonification3D(BaryonificationClass):

    def get_masses(self, model, r, M, a, mass_def):

        dlnr = np.log(r[1]/r[0])
        rho  = model.real(self.ccl_cosmo, r, M, a, mass_def = mass_def)
        M    = np.cumsum(4*np.pi*r**3 * rho * dlnr, axis = -1)

        return M


class Baryonification2D(BaryonificationClass):

    def get_masses(self, model, r, M, a, mass_def):

        dlnr  = np.log(r[1]/r[0])
        Sigma = model.projected(self.ccl_cosmo, r, M, a, mass_def = mass_def) * a #scale fac. cause proj. was done in comoving not phys.
        M     = np.cumsum(2*np.pi*r**2 * Sigma * dlnr, axis = -1)

        return M
