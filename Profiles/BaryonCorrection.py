
import numpy as np
import pyccl as ccl
from tqdm import tqdm
from scipy import interpolate


class BaryonificationClass(object):

    def __init__(self, DMO, DMB, ccl_cosmo, R_range = [1e-5, 40], N_samples = 500, epsilon_max = 4, use_concentration = False,
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
        M_DMO_interp = np.zeros([z_range.size, M_range.size, c_range.size, r.size])
        M_DMB_interp = np.zeros([z_range.size, M_range.size, c_range.size, r.size])

        M_DMB_range_interp = np.geomspace(1e5, 1e18, self.N_samples)
        log_r_new_interp   = np.zeros([z_range.size, M_range.size, c_range.size, M_DMB_range_interp.size])

        with tqdm(total = z_range.size * c_range.size, desc = 'Building Table', disable = not verbose) as pbar:
            for j in range(z_range.size):
                
                for k in range(c_range.size):
                    
                    if self.use_concentration:
                        self.DMO.set_parameter('cdelta', c_range[k])
                        self.DMB.set_parameter('cdelta', c_range[k])
                    else:
                        assert self.DMO.cdelta is None, "use_concentration = False, so set DMO model to have cdelta = None"
                        assert self.DMB.cdelta is None, "use_concentration = False, so set DMB model to have cdelta = None"
                    
                    #Extra factor of "a" accounts for projection in ccl being done in comoving, not physical units
                    M_DMO_interp[j, :, k, :] = self.get_masses(self.DMO, r, M_range, 1/(1 + z_range[j]), mass_def = self.mass_def)
                    M_DMB_interp[j, :, k, :] = self.get_masses(self.DMB, r, M_range, 1/(1 + z_range[j]), mass_def = self.mass_def)

                    for i in range(M_range.size):
                        log_r_new_interp[j, i, k, :] = np.interp(np.log(M_DMB_range_interp), np.log(M_DMB_interp[j, i, k]), np.log(r))

                    pbar.update(1)

                        

        if self.use_concentration:
            input_grid_1 = (np.log(1 + z_range), np.log(M_range), np.log(c_range), np.log(r))
            input_grid_2 = (np.log(1 + z_range), np.log(M_range), np.log(c_range), np.log(M_DMB_range_interp))
        else:
            input_grid_1 = (np.log(1 + z_range), np.log(M_range), np.log(r))
            input_grid_2 = (np.log(1 + z_range), np.log(M_range), np.log(M_DMB_range_interp))
            
            #Also squeeze the output to remove the redundant axis, since we don't extend in 
            #the concentration direction anymore
            M_DMO_interp = np.squeeze(M_DMO_interp, axis = 2)
            M_DMB_interp = np.squeeze(M_DMB_interp, axis = 2)
            log_r_new_interp = np.squeeze(log_r_new_interp, axis = 2)

        
        self.interp_DMO = interpolate.RegularGridInterpolator(input_grid_1, np.log(M_DMO_interp), bounds_error = False)
        self.interp_DMB = interpolate.RegularGridInterpolator(input_grid_2, log_r_new_interp, bounds_error = False) #Reverse needed for practical application
        

        return 0


    def displacements(self, x, M, a, c = None):
        
        if not (hasattr(self, 'interp_DMO') & hasattr(self, 'interp_DMB')):
            
            raise NameError("No Table created. Run setup_interpolator() method first")

        if self.use_concentration:
            assert c is not None, f"You asked for model to be built with concentration. But you set c = {c}"
            c_use = np.atleast_1d(c)
            
        r    = np.geomspace(self.R_range[0], self.R_range[1], self.N_samples)
        dlnr = np.log(r[1]) - np.log(r[0])

        z = 1/a - 1

        M_use = np.atleast_1d(M)
        R     = self.mass_def.get_radius(self.ccl_cosmo, M_use, a)/a #in comoving Mpc

        offset = np.zeros_like(x)
        inside = (x > self.R_range[0]) & (x < self.epsilon_max*R)

        x = x[inside]

        empty = np.ones_like(x)
        z_in  = np.log(1 + z)*empty
        M_in  = np.log(M)*empty

        
        if self.use_concentration:
            c_in  = np.log(c)*empty
            
            one = self.interp_DMO((z_in, M_in, c_in, np.log(x), ))
            two = self.interp_DMB((z_in, M_in, c_in, one, ))

        else:
            one = self.interp_DMO((z_in, M_in, np.log(x), ))
            two = self.interp_DMB((z_in, M_in, one, ))

        
        offset[inside] = np.exp(two) - x

        
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
        Sigma = model.projected(self.ccl_cosmo, r, M, a, mass_def = mass_def) * a #scale fac. needed because ccl projection done in comoving, not physical, units
        M     = np.cumsum(2*np.pi*r**2 * Sigma * dlnr, axis = -1)

        return M
