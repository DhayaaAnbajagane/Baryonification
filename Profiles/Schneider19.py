
import numpy as np
import pyccl as ccl
from operator import add, mul, sub, truediv, pow, neg, pos, abs
import warnings

from scipy import interpolate
from ..utils.Tabulate import _set_parameter


model_params = ['cdelta', 'epsilon', 'a', 'n', #DM profle params
                'q', 'p', #Relaxation params
                'cutoff', 'proj_cutoff', #Cutoff parameters (numerical)
                
                'theta_ej', 'theta_co', 'M_c', 'gamma', 'delta', #Default gas profile param
                'mu_theta_ej', 'mu_theta_co', 'mu_beta', 'mu_gamma', 'mu_delta', #Mass dep
                'M_theta_ej',  'M_theta_co', 'M_gamma', 'M_delta', #Mass dep norm
                'nu_theta_ej', 'nu_theta_co', 'nu_M_c',  'nu_gamma', 'nu_delta', #Redshift  dep
                'zeta_theta_ej', 'zeta_theta_co', 'zeta_M_c', 'zeta_gamma', 'zeta_delta', #Concentration dep
                
                'A', 'M1', 'eta', 'eta_delta', 'tau', 'tau_delta', 'epsilon_h', #Star params
                
                'alpha_nt', 'nu_nt', 'gamma_nt', 'mean_molecular_weight' #Non-thermal pressure and gas density
               
               ]

projection_params = ['padding_lo_proj', 'padding_hi_proj', 'n_per_decade_proj'] #Projection params

class SchneiderProfiles(ccl.halos.profiles.HaloProfile):

    def __init__(self, xi_mm = None, use_fftlog_projection = False, 
                 padding_lo_proj = 0.1, padding_hi_proj = 10, n_per_decade_proj = 10, **kwargs):
        
        #Go through all input params, and assign Nones to ones that don't exist.
        #If mass/redshift/conc-dependence, then set to 1 if don't exist
        for m in model_params + projection_params:
            if m in kwargs.keys():
                setattr(self, m, kwargs[m])
            elif ('mu_' in m) or ('nu_' in m) or ('zeta_' in m): #Set mass/red/conc dependence
                setattr(self, m, 0)
            elif ('M_' in m): #Set mass normalization
                setattr(self, m, 1e14)
            else:
                setattr(self, m, None)
                    
        #Some params for handling the realspace projection
        self.padding_lo_proj   = padding_lo_proj
        self.padding_hi_proj   = padding_hi_proj
        self.n_per_decade_proj = n_per_decade_proj 
        
        #Import all other parameters from the base CCL Profile class
        super(SchneiderProfiles, self).__init__()

        #Function that returns correlation func at different radii
        self.xi_mm = xi_mm

        #Sets the cutoff scale of all profiles, in comoving Mpc. Prevents divergence in FFTLog
        #Also set cutoff of projection integral. Should be the box side length
        self.cutoff      = kwargs['cutoff'] if 'cutoff' in kwargs.keys() else 1e3 #1Gpc is a safe default choice
        self.proj_cutoff = kwargs['proj_cutoff'] if 'proj_cutoff' in kwargs.keys() else self.cutoff
        
        
        #This allows user to force usage of the default FFTlog projection, if needed.
        #Otherwise, we use the realspace integration, since that allows for specification
        #of a hard boundary on radius
        if not use_fftlog_projection:
            self._projected = self._projected_realspace
        else:
            text = ("You must set the same cutoff for 3D profile and projection profile if you want to use fftlog projection. "
                    f"You have cutoff = {self.cutoff} and proj_cutoff = {self.proj_cutoff}")
            assert self.cutoff == self.proj_cutoff, text


        #Constant that helps with the fourier transform convolution integral.
        #This value minimized the ringing due to the transforms
        self.precision_fftlog['plaw_fourier'] = -2

        #Need this to prevent projected profile from artificially cutting off
        self.precision_fftlog['padding_lo_fftlog'] = 1e-2
        self.precision_fftlog['padding_hi_fftlog'] = 1e2

        self.precision_fftlog['padding_lo_extra'] = 1e-4
        self.precision_fftlog['padding_hi_extra'] = 1e4
        
    
    @property
    def model_params(self):
        
        params = {k:v for k,v in vars(self).items() if k in model_params}
                  
        return params
        
        
    
    def _get_gas_params(self, M, z):
        
        cdelta   = 1 if self.cdelta is None else self.cdelta
        
        M_c      = self.M_c * (1 + z)**self.nu_M_c * cdelta**self.zeta_M_c
        beta     = 3*(M/M_c)**self.mu_beta / (1 + (M/M_c)**self.mu_beta)
        
        #Use M_c as the mass-normalization for simplicity sake
        theta_ej = self.theta_ej * (M/self.M_theta_ej)**self.mu_theta_ej * (1 + z)**self.nu_theta_ej * cdelta**self.zeta_theta_ej
        theta_co = self.theta_co * (M/self.M_theta_co)**self.mu_theta_co * (1 + z)**self.nu_theta_co * cdelta**self.zeta_theta_co
        delta    = self.delta    * (M/self.M_delta)**self.mu_delta       * (1 + z)**self.nu_delta    * cdelta**self.zeta_delta
        gamma    = self.gamma    * (M/self.M_gamma)**self.mu_gamma       * (1 + z)**self.nu_gamma    * cdelta**self.zeta_gamma
        
        beta     = beta[:, None]
        theta_ej = theta_ej[:, None]
        theta_co = theta_co[:, None]
        delta    = delta[:, None]
        gamma    = gamma[:, None]
        
        return beta, theta_ej, theta_co, delta, gamma
        
        
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
        int_min = self.padding_lo_proj   * np.min(r_use)
        int_max = self.padding_hi_proj   * np.max(r_use)
        int_N   = self.n_per_decade_proj * np.int32(np.log10(int_max/int_min))
        
        #If proj_cutoff was passed, then rewrite the integral max limit
        if self.proj_cutoff is not None:
            int_max = self.proj_cutoff

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

        #This nested loop saves on memory, and vectorizing the calculation doesn't really
        #speed things up, so better to keep the loop this way.
        for i in range(M_use.size):
            for j in range(r_use.size):

                proj_prof[i, j] = 2*np.trapz(np.interp(np.sqrt(r_integral**2 + r_use[j]**2), r_integral, prof[i]), r_integral)

        
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            proj_prof = np.squeeze(proj_prof, axis=-1)
        if np.ndim(M) == 0:
            proj_prof = np.squeeze(proj_prof, axis=0)

        if np.any(proj_prof <= 0):
            warnings.warn("WARNING: Profile is zero/negative in some places."
                          "Likely a convolution artifact for objects smaller than the pixel scale")

        return proj_prof
    
    
    def __str_par__(self):
        '''
        String with all input params and their values
        '''
        
        string = f"("
        for m in model_params:
            string += f"{m} = {self.__dict__[m]}, "
        string += f"xi_mm = {self.xi_mm})"
        return string
        
    def __str_prf__(self):
        '''
        String with the class/profile name
        '''
        
        string = f"{self.__class__.__name__}"
        return string
        
    
    def __str__(self):
        
        string = self.__str_prf__() + self.__str_par__()
        return string 
    
    
    def __repr__(self):
        
        return self.__str__()
    
    
    #Add routines for consistently changing input params across all profiles
    def set_parameter(self, key, value): 
        _set_parameter(self, key, value)
    
    
    #Add routines for doing simple arithmetic operations with the classes
    from ..utils.misc import generate_operator_method
    
    __add__      = generate_operator_method(add)
    __mul__      = generate_operator_method(mul)
    __sub__      = generate_operator_method(sub)
    __truediv__  = generate_operator_method(truediv)
    __pow__      = generate_operator_method(pow)
    
    __radd__     = generate_operator_method(add, reflect = True)
    __rmul__     = generate_operator_method(mul, reflect = True)
    __rsub__     = generate_operator_method(sub, reflect = True)
    __rtruediv__ = generate_operator_method(truediv, reflect = True)
    
    __abs__      = generate_operator_method(abs)
    __pos__      = generate_operator_method(pos)
    __neg__      = generate_operator_method(neg)    



class DarkMatter(SchneiderProfiles):
    '''
    Total DM profile, which is just NFW
    '''

    def _real(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        if self.cdelta is None:
            c_M_relation = ccl.halos.concentration.ConcentrationDiemer15(mdef = mass_def) #Use the diemer calibration
            
        else:
            c_M_relation = ccl.halos.concentration.ConcentrationConstant(self.cdelta, mdef = mass_def)
            #c_M_relation = ccl.halos.concentration.ConcentrationConstant(7, mdef = mass_def) #needed to get Schneider result
            
        c   = c_M_relation.get_concentration(cosmo, M_use, a)
        R   = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc
        r_s = R/c
        r_t = R*self.epsilon
        
        r_s, r_t = r_s[:, None], r_t[:, None]

        
        #Get the normalization (rho_c) numerically
        #The analytic integral doesn't work since we have a truncation radii now.
        r_integral = np.geomspace(1e-6, 1000, 500)

        prof_integral  = 1/(r_integral/r_s * (1 + r_integral/r_s)**2) * 1/(1 + (r_integral/r_t)**2)**2
        
        Normalization  = [interpolate.PchipInterpolator(np.log(r_integral), 4 * np.pi * r_integral**3 * p) for p in prof_integral]
        Normalization  = np.array([N_i.antiderivative(nu = 1)(np.log(R_i)) for N_i, R_i in zip(Normalization, R)])
        
        rho_c = M_use/Normalization
        rho_c = rho_c[:, None]

        arg  = (r_use[None, :] - self.cutoff)
        arg  = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        prof = rho_c/(r_use/r_s * (1 + r_use/r_s)**2) * 1/(1 + (r_use/r_t)**2)**2 * kfac
        
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)


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
            xi_mm   = self.xi_mm(r_use, a)

        delta_c = 1.686/ccl.growth_factor(cosmo, a)
        nu_M    = delta_c / ccl.sigmaM(cosmo, M_use, a)
        bias_M  = 1 + (self.q*nu_M**2 - 1)/delta_c + 2*self.p/delta_c/(1 + (self.q*nu_M**2)**self.p)

        bias_M  = bias_M[:, None]
        prof    = (1 + bias_M * xi_mm)*ccl.rho_x(cosmo, a, species = 'matter', is_comoving = True)

        #Need this truncation so the fourier space integral isnt infinity
        arg  = (r_use[None, :] - self.cutoff)
        arg  = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        prof = prof * kfac

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)

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

        eta_cga = self.eta + self.eta_delta
        tau_cga = self.tau + self.tau_delta
        
        f_cga  = 2 * self.A * ((M_use/self.M1)**tau_cga  + (M_use/self.M1)**eta_cga)**-1

        R_h   = self.epsilon_h * R

        f_cga, R_h = f_cga[:, None], R_h[:, None]

        r_integral = np.geomspace(1e-6, 1000, 500)
        DM    = DarkMatter(**self.model_params); setattr(DM, 'cutoff', 1e3) #Set large cutoff just for normalization calculation
        rho   = DM.real(cosmo, r_integral, M_use, a, mass_def)
        M_tot = np.trapz(4*np.pi*r_integral**2 * rho, r_integral, axis = -1)
        M_tot = np.atleast_1d(M_tot)[:, None]
        
        arg  = (r_use[None, :] - self.cutoff)
        arg  = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        prof = f_cga*M_tot / (4*np.pi**(3/2)*R_h) * 1/r_use**2 * np.exp(-(r_use/2/R_h)**2) * kfac
                
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)

        return prof


class Gas(SchneiderProfiles):

    def _real(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):


        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        f_star = 2 * self.A * ((M_use/self.M1)**self.tau + (M_use/self.M1)**self.eta)**-1
        f_bar  = cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m
        f_gas  = f_bar - f_star
        f_gas  = f_gas[:, None]
        
        #Get gas params
        beta, theta_ej, theta_co, delta, gamma = self._get_gas_params(M_use, z)
        R_co = theta_co*R[:, None]
        R_ej = theta_ej*R[:, None]
        
        u = r_use/R_co
        v = r_use/R_ej
        
        
        #Integrate over wider region in radii to get normalization of gas profile
        r_integral = np.geomspace(1e-6, 1000, 500)

        u_integral = r_integral/R_co
        v_integral = r_integral/R_ej
        

        prof_integral = 1/(1 + u_integral)**beta / (1 + v_integral**gamma)**( (delta - beta)/gamma )
        Normalization = np.trapz(4 * np.pi * r_integral**2 * prof_integral, r_integral, axis = -1)[:, None]

        del u_integral, v_integral, prof_integral

        DM    = DarkMatter(**self.model_params); setattr(DM, 'cutoff', 1e3) #Set large cutoff just for normalization calculation
        rho   = DM.real(cosmo, r_integral, M_use, a, mass_def)
        M_tot = np.trapz(4*np.pi*r_integral**2 * rho, r_integral, axis = -1)
        M_tot = np.atleast_1d(M_tot)[:, None]
        
        arg   = (r_use[None, :] - self.cutoff)
        arg   = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac  = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        prof  = 1/(1 + u)**beta / (1 + v**gamma)**( (delta - beta)/gamma ) * kfac
        prof *= f_gas*M_tot/Normalization
        

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)


        return prof
    
    
class ShockedGas(Gas):
    '''
    Implements shocked gas profile, assuming a Rankine-Hugonoit conditions.
    To simplify, we assume a high mach-number shock, and so the 
    density is suppressed by a factor of 4.
    '''
    
    def __init__(self, epsilon_shock, width_shock, **kwargs):
        
        self.epsilon_shock = epsilon_shock
        self.width_shock   = width_shock
        
        super().__init__(**kwargs)

    def _real(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        #Minimum is 0.25 since a factor of 4x drop is the maximum possible for a shock
        rho_gas = super()._real(cosmo, r, M, a, mass_def)
        g_arg   = 1/self.width_shock*(np.log(r_use) - np.log(self.epsilon_shock*R)[:, None])
        g_arg   = np.where(g_arg > 1e2, np.inf, g_arg) #To prevent overflows when doing exp
        factor  = (1 - 0.25)/(1 + np.exp(g_arg)) + 0.25
        
        #Get the right size for rho_gas
        if M_use.size == 1: rho_gas = rho_gas[None, :]
            
        prof = rho_gas * factor
        
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        
        return prof


class CollisionlessMatter(SchneiderProfiles):
    
    def __init__(self, gas = None, stars = None, darkmatter = None, max_iter = 10, reltol = 1e-2, r_min_int = 1e-8, r_max_int = 1e5, r_steps = 5000, **kwargs):
        
        self.Gas   = gas
        self.Stars = stars
        self.DarkMatter = darkmatter
        
        if self.Gas is None: self.Gas = Gas(**kwargs)          
        if self.Stars is None: self.Stars = Stars(**kwargs)
        if self.DarkMatter is None: self.DarkMatter = DarkMatter(**kwargs)
            
        #Stop any artificially cutoffs when doing the relaxation.
        #The profile will be cutoff at the very last step instead
        self.Gas.set_parameter('cutoff', 1000)
        self.Stars.set_parameter('cutoff', 1000)
        self.DarkMatter.set_parameter('cutoff', 1000)
            
        self.max_iter   = max_iter
        self.reltol     = reltol

        self.r_min_int  = r_min_int
        self.r_max_int  = r_max_int
        self.r_steps    = r_steps
        
        super().__init__(**kwargs)
        

    def _real(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        if np.min(r) < self.r_min_int: 
            warnings.warn(f"Decrease integral lower limit, r_min_int ({self.r_min_int}) < minimum radius ({np.min(r)})", UserWarning)
        if np.max(r) > self.r_max_int: 
            warnings.warn(f"Increase integral lower limit, r_min_int ({self.r_max_int}) < minimum radius ({np.max(r)})", UserWarning)

        #Def radius sampling for doing iteration.
        #And don't check iteration near the boundaries, since we can have numerical errors
        #due to the finite width oof the profile during iteration.
        #Radius boundary is very large, I found that worked best without throwing edgecases
        #especially when doing FFTlog transforms
        r_integral = np.geomspace(self.r_min_int, self.r_max_int, self.r_steps)
        safe_range = (r_integral > 2 * np.min(r_integral) ) & (r_integral < 1/2 * np.max(r_integral) )
        
        z = 1/a - 1

        R = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        eta_cga = self.eta + self.eta_delta
        tau_cga = self.tau + self.tau_delta
        
        f_star = 2 * self.A * ((M_use/self.M1)**self.tau + (M_use/self.M1)**self.eta)**-1
        f_cga  = 2 * self.A * ((M_use/self.M1)**tau_cga  + (M_use/self.M1)**eta_cga)**-1
        f_star = f_star[:, None]
        f_cga  = f_cga[:, None]
        f_sga  = f_star - f_cga
        f_clm  = 1 - cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m + f_sga
        
        
        rho_i      = self.DarkMatter.real(cosmo, r_integral, M_use, a, mass_def)
        rho_cga    = self.Stars.real(cosmo, r_integral, M_use, a, mass_def)
        rho_gas    = self.Gas.real(cosmo, r_integral, M_use, a, mass_def)

        dlnr  = np.log(r_integral[1]) - np.log(r_integral[0])
        M_i   = 4 * np.pi * np.cumsum(r_integral**3 * rho_i   * dlnr, axis = -1)
        M_cga = 4 * np.pi * np.cumsum(r_integral**3 * rho_cga * dlnr, axis = -1)
        M_gas = 4 * np.pi * np.cumsum(r_integral**3 * rho_gas * dlnr, axis = -1)
        
        #We intentionally set Extrapolate = True. This is to handle behavior at extreme small-scales (due to stellar profile)
        #and radius limits at largest scales. Using extrapolate=True does not introduce numerical artifacts into predictions
        ln_M_NFW = [interpolate.PchipInterpolator(np.log(r_integral), np.log(M_i[m_i]),   extrapolate = True) for m_i in range(M_i.shape[0])]
        ln_M_cga = [interpolate.PchipInterpolator(np.log(r_integral), np.log(M_cga[m_i]), extrapolate = True) for m_i in range(M_i.shape[0])]
        ln_M_gas = [interpolate.PchipInterpolator(np.log(r_integral), np.log(M_gas[m_i]), extrapolate = True) for m_i in range(M_i.shape[0])]

        del M_cga, M_gas, rho_i, rho_cga, rho_gas

        relaxation_fraction = np.ones_like(M_i)

        for m_i in range(M_i.shape[0]):
            
            counter  = 0
            max_rel_diff = np.inf #Initializing variable at infinity
            
            while max_rel_diff > self.reltol:

                with np.errstate(over = 'ignore'):
                    r_f  = r_integral*relaxation_fraction[m_i]
                    M_f  = f_clm[m_i]*M_i[m_i] + np.exp(ln_M_cga[m_i](np.log(r_f))) + np.exp(ln_M_gas[m_i](np.log(r_f)))

                relaxation_fraction_new = self.a*( (M_i[m_i]/M_f)**self.n - 1 ) + 1

                diff     = relaxation_fraction_new/relaxation_fraction[m_i] - 1
                abs_diff = np.abs(diff)
                
                max_rel_diff = np.max(abs_diff[safe_range])
                
                relaxation_fraction[m_i] = relaxation_fraction_new

                counter += 1

                #Though we do a while loop, we break it off after 10 tries
                #this seems to work well enough. The loop converges
                #after two or three iterations.
                if (counter >= self.max_iter) & (max_rel_diff > self.reltol): 
                    
                    med_rel_diff = np.max(abs_diff[safe_range])
                    warn_text = ("Profile of halo index %d did not converge after %d tries." % (m_i, counter) +
                                 "Max_diff = %0.5f, Median_diff = %0.5f. Try increasing max_iter." % (max_rel_diff, med_rel_diff)
                                )
                    
                    warnings.warn(warn_text, UserWarning)
                    break

        ln_M_clm = np.vstack([np.log(f_clm[m_i]) + ln_M_NFW[m_i](np.log(r_integral/relaxation_fraction[m_i])) for m_i in range(M_i.shape[0])])
        ln_M_clm = interpolate.CubicSpline(np.log(r_integral), ln_M_clm, axis = -1, extrapolate = False)
        log_der  = ln_M_clm.derivative(nu = 1)(np.log(r_use))
        lin_der  = log_der * np.exp(ln_M_clm(np.log(r_use))) / r_use
        prof     = 1/(4*np.pi*r_use**2) * lin_der
        
        arg  = (r_use[None, :] - self.cutoff)
        arg  = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        prof = np.where(np.isnan(prof), 0, prof) * kfac

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)

        return prof


class DarkMatterOnly(SchneiderProfiles):

    def __init__(self, darkmatter = None, twohalo = None, **kwargs):
        
        self.DarkMatter = darkmatter
        self.TwoHalo    = twohalo
        
        if self.TwoHalo is None: self.TwoHalo = TwoHalo(**kwargs)
        if self.DarkMatter is None: self.DarkMatter = DarkMatter(**kwargs)
            
        super().__init__(**kwargs)
        
    def _real(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        prof = (self.DarkMatter.real(cosmo, r, M, a, mass_def) +
                self.TwoHalo.real(cosmo, r, M, a, mass_def)
               )

        return prof


class DarkMatterBaryon(SchneiderProfiles):

    def __init__(self, gas = None, stars = None, collisionlessmatter = None, darkmatter = None, twohalo = None, **kwargs):
        
        self.Gas   = gas
        self.Stars = stars
        self.TwoHalo    = twohalo
        self.DarkMatter = darkmatter
        self.CollisionlessMatter = collisionlessmatter
        
        if self.Gas is None: self.Gas = Gas(**kwargs)          
        if self.Stars is None: self.Stars = Stars(**kwargs)
        if self.TwoHalo is None: self.TwoHalo = TwoHalo(**kwargs)
        if self.DarkMatter is None: self.DarkMatter = DarkMatter(**kwargs)
        if self.CollisionlessMatter is None: self.CollisionlessMatter = CollisionlessMatter(**kwargs)
            
        super().__init__(**kwargs)
        
    def _real(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        #Need DMO for normalization
        #Makes sure that M_DMO(<r) = M_DMB(<r) for the limit r --> infinity
        #This is just for the onehalo term
        r_integral = np.geomspace(1e-5, 100, 500)

        rho   = self.DarkMatter.real(cosmo, r_integral, M, a, mass_def)
        M_tot = np.trapz(4*np.pi*r_integral**2 * rho, r_integral)

        rho   = (self.CollisionlessMatter.real(cosmo, r_integral, M, a, mass_def) +
                 self.Stars.real(cosmo, r_integral, M, a, mass_def) +
                 self.Gas.real(cosmo, r_integral, M, a, mass_def))

        M_tot_dmb = np.trapz(4*np.pi*r_integral**2 * rho, r_integral, axis = -1)

        Factor = M_tot/M_tot_dmb
        
        if np.ndim(Factor) == 1:
            Factor = Factor[:, None]

        prof = (self.CollisionlessMatter.real(cosmo, r, M, a, mass_def) * Factor +
                self.Stars.real(cosmo, r, M, a, mass_def) * Factor +
                self.Gas.real(cosmo, r, M, a, mass_def) * Factor +
                self.TwoHalo.real(cosmo, r, M, a, mass_def))

        return prof
