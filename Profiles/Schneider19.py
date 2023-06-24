
import numpy as np
import pyccl as ccl

from scipy import interpolate
from astropy.cosmology import z_at_value, FlatLambdaCDM, FlatwCDM
from astropy import units as u

class SchneiderProfiles(ccl.halos.profiles.HaloProfile):

    def __init__(self,
                 epsilon = None, a = None, n = None,
                 theta_ej = None, theta_co = None, M_c = None, mu = None,
                 A = None, M1 = None, eta_star = None, eta_cga = None, epsilon_h = None,
                 q = None, p = None, xi_mm = None, R_range = [1e-10, 1e10]):


        self.epsilon   = epsilon
        self.a         = a
        self.n         = n
        self.theta_ej  = theta_ej
        self.theta_co  = theta_co
        self.M_c       = M_c
        self.mu        = mu
        self.A         = A
        self.M1        = M1
        self.eta_star  = eta_star
        self.eta_cga   = eta_cga
        self.epsilon_h = epsilon_h
        self.q         = q
        self.p         = p

        #Import all other parameters from the base CCL Profile class
        super(SchneiderProfiles, self).__init__()

        #Function that returns correlation func at different radii
        self.xi_mm     = xi_mm

        #Sets the range that we compute profiles too (if we need to do any numerical stuff)
        self.R_range = R_range

        #Constant that helps with the fourier transform convolution integral.
        #This value minimized the ringing due to the transforms
        self.precision_fftlog['plaw_fourier'] = -2

        #Need this to prevent projected profile from artificially cutting off
        self.precision_fftlog['padding_lo_fftlog'] = 1e-2
        self.precision_fftlog['padding_hi_fftlog'] = 1e2

        self.precision_fftlog['padding_lo_extra'] = 1e-2
        self.precision_fftlog['padding_hi_extra'] = 1e2

    def _projected_realspace(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):
        '''
        Custom method for projection where we do it all in real-space. Not that slow and
        can avoid any hankel transform features.
        '''

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        #Integral limits
        int_min = self.precision_fftlog['padding_lo_fftlog']*np.min(r_use)
        int_max = self.precision_fftlog['padding_hi_fftlog']*np.max(r_use)
        int_N   = self.precision_fftlog['n_per_decade'] * np.int32(np.log10(int_max/int_min))

        r_integral = np.geomspace(int_min, int_max, int_N)

        prof = self._real(cosmo, r_integral, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical'))

        #The prof object is already "squeezed" in some way.
        #Code below removes that squeezing so rest of code can handle
        #passing multiple radii and masses.
        if np.ndim(r) == 0:
            prof = prof[:, None]
        if np.ndim(M) == 0:
            prof = prof[None, :]

        proj_prof = np.zeros([M_use.size, r_use.size])

        for i in range(M_use.size):
            for j in range(r_use.size):

                proj_prof[i, j] = 2*np.trapz(np.interp(np.sqrt(r_integral**2 + r_use[j]**2), r_integral, prof[i]), r_integral)

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            proj_prof = np.squeeze(proj_prof, axis=-1)
        if np.ndim(M) == 0:
            proj_prof = np.squeeze(proj_prof, axis=0)

        assert np.all(proj_prof >= 0), "Something went wrong. Profile is negative in some places"

        return proj_prof



class DarkMatter(SchneiderProfiles):
    '''
    Total DM profile, which is just NFW
    '''

    def _real(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        c_M_relation = ccl.halos.concentration.ConcentrationDiemer15(mdef = mass_def)
#         c_M_relation = ccl.halos.concentration.ConcentrationConstant(7, mdef = mass_def) #needed to get Schneider result

        c   = c_M_relation.get_concentration(cosmo, M_use, a)
        R   = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc
        r_s = R/c
        r_t = R*self.epsilon

        rho_c = M_use/(4*np.pi*r_s**3*(np.log(1 + c) - c/(1 + c)))

        r_s, rho_c, r_t = r_s[:, None], rho_c[:, None], r_t[:, None]
        
        prof = rho_c/(r_use/r_s * (1 + r_use/r_s)**2) * 1/(1 + (r_use/r_t)**2)**2

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)


#         assert np.all(prof >= 0), "Something went wrong. Profile is negative in some places"

        return prof

class TwoHalo(SchneiderProfiles):
    '''
    Simple two halo term (uses 2pt corr func, not halo model)
    '''

    def _real(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):

        #Need it to be linear if we're doing two halo term
        assert cosmo._config_init_kwargs['matter_power_spectrum'] == 'linear', "Must use matter_power_spectrum = linear for 2-halo term"

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        R   = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        z = 1/a - 1

        if self.xi_mm is None:
            xi_mm   = ccl.correlation_3d(cosmo, a, r_use)
        else:
            xi_mm   = self.xi_mm(r_use,)

        delta_c = 1.686/ccl.growth_factor(cosmo, a)
        nu_M    = delta_c / ccl.sigmaM(cosmo, M_use, a)
        bias_M  = 1 + (self.q*nu_M**2 - 1)/delta_c + 2*self.p/delta_c/(1 + (self.q*nu_M**2)**self.p)

        #Schneider uses (bias * corr + 1), but this includes the background mean matter density
        #so we instead use just (bias * corr)
        bias_M  = bias_M[:, None]
        prof    = (1 + bias_M * xi_mm)*ccl.rho_x(cosmo, a, species = 'matter', is_comoving = True)

        #Need this zeroing out to do projection in fourier space
        prof[:, (r_use > 50)] = 0

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)

        #print(prof)
        #assert np.all(prof >= 0), "Something went wrong. TwoHalo profile is negative in some places"

        return prof

class Stars(SchneiderProfiles):
    '''
    Exponential stellar mass profile
    '''
    
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        
        #For some reason, we need to make this extreme in order
        #to prevent ringing in the profiles. Haven't figured out
        #why this is the case
        self.precision_fftlog['padding_lo_fftlog'] = 1e-5
        self.precision_fftlog['padding_hi_fftlog'] = 1e5

    def _real(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):


        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R   = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        f_cga = self.A * (self.M1/M_use)**self.eta_cga
        R_h   = self.epsilon_h * R

        f_cga, R_h = f_cga[:, None], R_h[:, None]

        r_integral = np.geomspace(1e-3, 100, 500)
        rho   = DarkMatter(epsilon = self.epsilon).real(cosmo, r_integral, M_use, a, mass_def)
        M_tot = np.trapz(4*np.pi*r_integral**2 * rho, r_integral, axis = -1)
        M_tot = np.atleast_1d(M_tot)[:, None]
        
        prof = f_cga*M_tot / (4*np.pi**(3/2)*R_h) * 1/r_use**2 * np.exp(-(r_use/2/R_h)**2)
                
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)

#         assert np.all(prof >= 0), "Something went wrong. Profile is negative in some places"

        return prof


class Gas(SchneiderProfiles):

    def _real(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):


        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        u = r_use/(self.theta_co*R)[:, None]
        v = r_use/(self.theta_ej*R)[:, None]
        # w = r_use/(50*R)[:, None] #We hardcode 50*R200c as a choice for radial cutoff of profile

        f_star = self.A * (self.M1/M_use)**self.eta_star
        f_bar  = cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m
        f_gas  = f_bar - f_star

        beta   = 3 - (self.M_c/M_use)**self.mu

        f_gas, beta = f_gas[:, None], beta[:, None]

        #Integrate over wider region in radii to get normalization of gas profile

        r_integral = np.geomspace(1e-3, 100, 500)

        u_integral = r_integral/(self.theta_co*R)[:, None]
        v_integral = r_integral/(self.theta_ej*R)[:, None]
        # w_integral = r_integral/(50*R)[:, None]

        #We modify the profile slightly. We use (1 + v) instead of (1 + v^2) so that we match Battaglia (and also)
        #match the GNFW form. We include a second truncation radius beyond that for numerical reasons so 
        #that M(< r_infty) is a finite number.
        prof_integral  = 1/(1 + u_integral)**beta / (1 + v_integral**2)**((7 - beta)/2) #/ (1 + w_integral**2)**2

        Normalization  = interpolate.CubicSpline(np.log(r_integral), 4 * np.pi * r_integral**3 * prof_integral, axis = -1)
        Normalization  = Normalization.integrate(np.log(r_integral[0]), np.log(r_integral[-1]))
        Normalization  = Normalization[:, None]

        del u_integral, v_integral, prof_integral

        rho   = DarkMatter(epsilon = self.epsilon).real(cosmo, r_integral, M, a, mass_def)
        M_tot = np.trapz(4*np.pi*r_integral**2 * rho, r_integral, axis = -1)
        M_tot = np.atleast_1d(M_tot)[:, None]

        prof  = 1/(1 + u)**beta / (1 + v**2)**((7 - beta)/2) #/ (1 + w**2)**2
        prof *= f_gas*M_tot/Normalization

#         prof[r_use > 50*R[:, None]] = 0

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)


#         assert np.all(prof >= 0), "Something went wrong. Profile is negative in some places"

        return prof

class CollisionlessMatter(SchneiderProfiles):

    def _real(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        r_integral = np.geomspace(1e-5, 200, 500)

        z = 1/a - 1

        R = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        f_star = self.A * (self.M1/M_use)**self.eta_star
        f_cga  = self.A * (self.M1/M_use)**self.eta_cga
        f_star = f_star[:, None]
        f_cga  = f_cga[:, None]
        f_sga  = f_star - f_cga
        f_clm  = 1 - cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m + f_sga
        
        
        NFW_DMO    = DarkMatter(epsilon = self.epsilon)
        Stars_prof = Stars(A = self.A, M1 = self.M1, eta_star = self.eta_star, eta_cga = self.eta_cga, epsilon_h = self.epsilon_h, epsilon = self.epsilon)
        Gas_prof   = Gas(theta_ej = self.theta_ej, theta_co = self.theta_co, M_c = self.M_c, mu = self.mu, A = self.A, M1 = self.M1, eta_star = self.eta_star, epsilon = self.epsilon)

        rho_i      = NFW_DMO.real(cosmo, r_integral, M, a, mass_def)
        rho_cga    = Stars_prof.real(cosmo, r_integral, M, a, mass_def)
        rho_gas    = Gas_prof.real(cosmo, r_integral, M, a, mass_def)

        #The ccl profile class removes the dimension of size 1
        #we're adding it back in here in order to keep code general
        if M_use.size == 1:
            rho_i   = rho_i[None, :]
            rho_cga = rho_cga[None, :]
            rho_gas = rho_gas[None, :]
            
        dlnr  = np.log(r_integral[1]) - np.log(r_integral[0])
        M_i   = 4 * np.pi * np.cumsum(r_integral**3 * rho_i   * dlnr, axis = -1)
        M_cga = 4 * np.pi * np.cumsum(r_integral**3 * rho_cga * dlnr, axis = -1)
        M_gas = 4 * np.pi * np.cumsum(r_integral**3 * rho_gas * dlnr, axis = -1)
        
        ln_M_NFW = [interpolate.CubicSpline(np.log(r_integral), np.log(M_i[m_i]), axis = -1) for m_i in range(M_i.shape[0])]
        ln_M_cga = [interpolate.CubicSpline(np.log(r_integral), np.log(M_cga[m_i]), axis = -1) for m_i in range(M_i.shape[0])]
        ln_M_gas = [interpolate.CubicSpline(np.log(r_integral), np.log(M_gas[m_i]), axis = -1) for m_i in range(M_i.shape[0])]

        del M_cga, M_gas, rho_i, rho_cga, rho_gas

        relaxation_fraction = np.ones_like(M_i)

        for m_i in range(M_i.shape[0]):
            
            counter = 0
            diff = np.inf #Initializing variable at infinity
            
            while np.any(np.abs(diff)) > 1e-2:

                r_f  = r_integral*relaxation_fraction[m_i]
                M_f  = f_clm[m_i]*M_i[m_i] + np.exp(ln_M_cga[m_i](np.log(r_f))) + np.exp(ln_M_gas[m_i](np.log(r_f)))

                relaxation_fraction_new = self.a*((M_i[m_i]/M_f)**self.n - 1) + 1

                diff = relaxation_fraction_new/relaxation_fraction[m_i] - 1

                relaxation_fraction[m_i] = relaxation_fraction_new

                counter += 1

                #Though we do a while loop, we break it off after 10 tries
                #this seems to work well enough. The loop converges
                #after two or three iterations.
                if counter >= 10: break

        ln_M_clm = np.vstack([np.log(f_clm[m_i]) + 
                              ln_M_NFW[m_i](np.log(r_integral/relaxation_fraction[m_i])) for m_i in range(M_i.shape[0])])
        ln_M_clm = interpolate.CubicSpline(np.log(r_integral), ln_M_clm, axis = -1, extrapolate = False)
        log_der  = ln_M_clm.derivative(nu = 1)(np.log(r_use))
        lin_der  = log_der * np.exp(ln_M_clm(np.log(r_use))) / r_use
        prof     = 1/(4*np.pi*r_use**2) * lin_der
        prof = np.where(np.isnan(prof), 0, prof)

#         prof[(r_use[None, :] < r_integral[0]) | (r_use > 50*R[:, None]) | (r_use[None, :] > r_integral[-1])] = 0

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)

#         assert np.all(prof >= 0), "Something went wrong. Profile is negative in some places"

        return prof

class DarkMatterOnly(SchneiderProfiles):

    def _real(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        DarkMatter_prof = DarkMatter(epsilon = self.epsilon)
        TwoHalo_prof    = TwoHalo(p = self.p, q = self.q, xi_mm = self.xi_mm)

        prof = (DarkMatter_prof.real(cosmo, r, M, a, mass_def) +
                TwoHalo_prof.real(cosmo, r, M, a, mass_def))

        return prof

class DarkMatterBaryon(SchneiderProfiles):

    def _real(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        Collisionless_prof = CollisionlessMatter(epsilon = self.epsilon, a = self.a, n = self.n,
                                                 theta_ej = self.theta_ej, theta_co = self.theta_co, M_c = self.M_c, mu = self.mu,
                                                 A = self.A, M1 = self.M1, eta_star = self.eta_star, eta_cga = self.eta_cga, epsilon_h = self.epsilon_h)
        Stars_prof         = Stars(A = self.A, M1 = self.M1, eta_star = self.eta_star, eta_cga = self.eta_cga, epsilon_h = self.epsilon_h, epsilon = self.epsilon)
        Gas_prof           = Gas(theta_ej = self.theta_ej, theta_co = self.theta_co, M_c = self.M_c, mu = self.mu, A = self.A, M1 = self.M1, eta_star = self.eta_star, epsilon = self.epsilon)
        TwoHalo_prof       = TwoHalo(p = self.p, q = self.q, xi_mm = self.xi_mm)


        #Need DMO for normalization
        #Makes sure that M_DMO(<r) = M_DMB(<r) for the limit r --> infinity
        #This is just for the onehalo term
        r_integral = np.geomspace(1e-3, 100, 500)

        rho   = DarkMatter(epsilon = self.epsilon).real(cosmo, r_integral, M, a, mass_def)
        M_tot = np.trapz(4*np.pi*r_integral**2 * rho, r_integral)

        rho   = (Collisionless_prof.real(cosmo, r_integral, M, a, mass_def) +
                 Stars_prof.real(cosmo, r_integral, M, a, mass_def) +
                 Gas_prof.real(cosmo, r_integral, M, a, mass_def))

        M_tot_dmb = np.trapz(4*np.pi*r_integral**2 * rho, r_integral, axis = -1)

        Factor = M_tot/M_tot_dmb
        
        if np.ndim(Factor) == 1:
            Factor = Factor[:, None]

        prof = (Collisionless_prof.real(cosmo, r, M, a, mass_def) * Factor +
                Stars_prof.real(cosmo, r, M, a, mass_def) * Factor +
                Gas_prof.real(cosmo, r, M, a, mass_def) * Factor +
                TwoHalo_prof.real(cosmo, r, M, a, mass_def))

        return prof
