
import numpy as np
import pyccl as ccl

from scipy import interpolate
from astropy.cosmology import z_at_value, FlatLambdaCDM, FlatwCDM
from astropy import units as u

from ..Profiles.Schneider19 import SchneiderProfiles, Gas, DarkMatterBaryon, TwoHalo


#Define relevant physical constants
Msun_to_Kg = ccl.physical_constants.SOLAR_MASS
Mpc_to_m   = ccl.physical_constants.MPC_TO_METER
G          = ccl.physical_constants.GNEWT / Mpc_to_m**3 * Msun_to_Kg
m_to_cm    = 1e2
kb_in_ev   = ccl.physical_constants.KBOLTZ / ccl.physical_constants.EV_IN_J

#Just define some useful conversions/constants
sigma_T = 6.652458e-29 / Mpc_to_m**2
m_e     = 9.10938e-31 / Msun_to_Kg
m_p     = 1.67262e-27 / Msun_to_Kg
c       = 2.99792458e8 / Mpc_to_m

#Thermodynamic/abundance quantities
Y         = 0.24 #Helium mass ratio
Pth_to_Pe = (4 - 2*Y)/(8 - 5*Y) #Factor to convert gas temp. to electron temp


#Technically P(r -> infty) is zero, but we  may need finite
#value for numerical reasons (interpolator). This is a
#computatational cosntant
Pressure_at_infinity = 0


class BattagliaPressure(ccl.halos.profiles.HaloProfile):

    '''
    Class that implements a Battaglia profile using the
    CCL profile class. This class inheritance allows us to
    easily compute relevant observables using the CCL
    machinery.

    ------------------
    Params:
    ------------------

    Model_def : str
        The available mode calibrations from Battaglia+ 2012.
        Can be one of '200_AGN', '500_AGN', and '500_SH'. The former
        two were calibrated using simulations w/ AGN feedback, whereas
        the latter did not. The initial three-digit number denoted the
        spherical overdensity definition used in the calibration (Either
        M200c or M500c).

    truncate : float
        The radius (in units of R/Rdef, where Rdef is the halo radius defined
        via some chosen spherical overdensity definition) at which to cutoff
        the profiles and set them to zero. Default is False.
     '''

    def __init__(self, Model_def, truncate = False):

        #Set mass definition using the input Model_def
        if Model_def == '200_AGN':
            self.mdef = ccl.halos.massdef.MassDef(200, 'critical')

        elif Model_def == '500_AGN':
            self.mdef = ccl.halos.massdef.MassDef(500, 'critical')

        elif Model_def == '500_SH':
            self.mdef = ccl.halos.massdef.MassDef(500, 'critical')

        else:

            raise ValueError("Input Model_def not valid. Select one of: 200_AGN, 500_AGN, 500_SH")

        self.Model_def = Model_def
        self.truncate  = truncate

        #Import all other parameters from the base CCL Profile class
        super(BattagliaPressure, self).__init__()

        #Constant that helps with the fourier transform convolution integral.
        #This value minimized the ringing due to the transforms
        self.precision_fftlog['plaw_fourier'] = -2

        #Need this to prevent projected profile from artificially cutting off
        self.precision_fftlog['padding_lo_fftlog'] = 1e-4
        self.precision_fftlog['padding_hi_fftlog'] = 1e4

    def _real(self, cosmo, r, M, a, mass_def=None):

        '''
        Function that computes the Battaglia pressure profile for halos.
        Can use three different definitions: 200_AGN, 500_AGN, and 500_SH.

        Based on arxiv:1109.3711

        ------------------
        Params:
        ------------------

        cosmo : pyccl.Cosmology object
            A CCL cosmology object that contains the relevant
            cosmological parameters

        r : float, numpy array, list
            Radii (in comoving Mpc) to evaluate the profiles at.

        M : float, numpy array, list
            The list of halo masses (in Msun) to compute the profiles
            around.

        a : float
            The cosmic scale factor

        mass_def : ccl.halos.massdef.MassDef object
            The mass definition associated with the input, M


        ------------------
        Output:
        ------------------

        numpy array :
            An array of size (M.size, R.size) that contains the electron
            pressure values at radius R of each cluster of a mass given
            by array M. If M.size and/or R.size are simply 1, then the output
            is flattened along that dimension.
        '''

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1
        mass_def = self.mdef

        #Setup parameters as they were calibrated in Battaglia+ 2012
        if self.Model_def == '200_AGN':

            P_0  = 18.1  * (M_use/1e14)**0.154    * (1 + z)**-0.758
            x_c  = 0.497 * (M_use/1e14)**-0.00865 * (1 + z)**0.731
            beta = 4.35  * (M_use/1e14)**0.0393   * (1 + z)**0.415

        elif self.Model_def == '500_AGN':

            P_0  = 7.49  * (M_use/1e14)**0.226   * (1 + z)**-0.957
            x_c  = 0.710 * (M_use/1e14)**-0.0833 * (1 + z)**0.853
            beta = 4.19  * (M_use/1e14)**0.0480  * (1 + z)**0.615

        elif self.Model_def == '500_SH':

            P_0  = 20.7  * (M_use/1e14)**-0.074 * (1 + z)**-0.743
            x_c  = 0.428 * (M_use/1e14)**0.011  * (1 + z)**1.01
            beta = 3.82  * (M_use/1e14)**0.0375 * (1 + z)**0.535


        R = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc
        x = r_use[None, :]/R[:, None]

        #The overdensity constrast related to the mass definition
        Delta    = mass_def.get_Delta(cosmo, a)

        #Cosmological parameters
        Omega_m  = cosmo.cosmo.params.Omega_m
        Omega_b  = cosmo.cosmo.params.Omega_b
        Omega_g  = cosmo.cosmo.params.Omega_g
        h        = cosmo.cosmo.params.h

        RHO_CRIT = ccl.physical_constants.RHO_CRITICAL*h**2 * ccl.background.h_over_h0(cosmo, a)**2 #This is in physical coordinates

        # The self-similar expectation for Pressure
        # Need R*a to convert comoving Mpc to physical
        P_delta = Delta*RHO_CRIT * Omega_b/Omega_m * G * (M_use)/(2*R*a)
        alpha, gamma = 1, -0.3

        P_delta, P_0, beta, x_c = P_delta[:, None], P_0[:, None], beta[:, None], x_c[:, None]
        prof = P_delta * P_0 * (x/x_c)**gamma * (1 + (x/x_c)**alpha)**-beta

        # Battaglia profile has validity limits for redshift, mass, and distance from halo center.
        # Here, we enforce the distance limit at R/R_Delta > X, where X is input by user
        if self.truncate:
            prof[x > self.truncate] = 0
            
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        
        return prof
    

class Pressure(SchneiderProfiles):
    
    def __init__(self, nonthermal_model = None, **kwargs):
        
        self.nonthermal_model = nonthermal_model
        super().__init__(**kwargs)
    
    def _real(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):
        
        
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        r_integral = np.geomspace(1e-3, 100, 500) #Hardcoded ranges

        Total_prof = DarkMatterBaryon(epsilon = self.epsilon, a = self.a, n = self.n,
                                      theta_ej = self.theta_ej, theta_co = self.theta_co, M_c = self.M_c, mu = self.mu,
                                      A = self.A, M1 = self.M1, eta_star = self.eta_star, eta_cga = self.eta_cga, epsilon_h = self.epsilon_h,
                                      p = self.p, q = self.q, xi_mm = self.xi_mm)
        Gas_prof   = Gas(theta_ej = self.theta_ej, theta_co = self.theta_co, M_c = self.M_c, mu = self.mu, A = self.A, M1 = self.M1, eta_star = self.eta_star, epsilon = self.epsilon)
        

        rho_total  = Total_prof.real(cosmo, r_integral, M, a, mass_def = mass_def)
        rho_gas    = Gas_prof.real(cosmo, r_integral, M, a, mass_def = mass_def)
        
        #Remove two halo term from total matter profile. 
        #Only the halo (one halo term) is in equilibrium in our model
        TwoHalo_prof = TwoHalo(p = self.p, q = self.q, xi_mm = self.xi_mm)
        rho_total   -= TwoHalo_prof.real(cosmo, r_integral, M, a, mass_def = mass_def)
        
        dlnr    = np.log(r_integral[1]) - np.log(r_integral[0])
        M_total = 4 * np.pi * np.cumsum(r_integral**3 * rho_total * dlnr, axis = -1)

        #Assuming hydrostatic equilibrium to get dP/dr = -G*M(<r)*rho(r)/r^2
        dP_dr = - G * M_total * rho_gas / r_integral**2
        
        #Make it have the right shape that ccl expects (size(M), size(r)) 
        if len(dP_dr.shape) < 2:
            dP_dr = dP_dr[np.newaxis, :]

        #integrate to get actual pressure, P(r). Boundary condition is P(r -> infty) = 0.
        #So we start from the boundary and integrate inwards. We reverse array once to
        #flip the integral direction, and flip it second time so P(r) goes from r = 0 to r = infty
        prof  = - np.cumsum((dP_dr * r_integral)[:, ::-1] * dlnr, axis = -1)[:, ::-1]
        
        prof  = interpolate.CubicSpline(np.log(r_integral), np.log(prof + Pressure_at_infinity), axis = 1, extrapolate = False)
        prof  = np.exp(prof(np.log(r_use)) - Pressure_at_infinity)
        prof  = np.where(np.isfinite(prof), prof, 0) #Get rid of pesky NaN and inf values! They break CCL spline interpolator
        
        nth_frac = self._nonthermal_fraction(cosmo, r_use, M_use, a, model = self.nonthermal_model, mass_def = mass_def)

        prof     = prof * (1 - nth_frac) #We only want thermal pressure

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)

        assert np.all(prof >= 0), "Something went wrong. Pressure is negative in some places"

        return prof

    def _nonthermal_fraction(self, cosmo, r, M, a, model = 'Green20', mass_def = ccl.halos.massdef.MassDef(200, 'critical')):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        if model is None:
            nth = 0
        
        elif model == 'Green20': #Based on arxiv: 2002.01934, equation 20 and Table 3
            
            #They define the model with R200m, so gotta use that redefinition here.
            M200m = mass_def.translate_mass(cosmo, M_use, a, ccl.halos.massdef.MassDef200m())
            R200m = ccl.halos.massdef.MassDef200m().get_radius(cosmo, M_use, a)/a #in comoving distance

            x = r_use/R200m[:, None]

            a, b, c, d, e, f = 0.495, 0.719, 1.417,-0.166, 0.265, -2.116
            nu_M = M_use/ccl.sigmaM(cosmo, M200m, a)
            nu_M = nu_M[:, None]
            nth  = 1 - a * (1 + np.exp(-(x/b)**c)) * (nu_M/4.1)**(d/(1 + (x/e)**f))
            
        else:
            raise ValueError("Need to use model = None or model = Green20 No other model implemented so far.")

        return nth

class GasNumberDensity(SchneiderProfiles):
    
    def __init__(self, mean_molecular_weight = 1.15, **kwargs):
        
        self.mean_molecular_weight = mean_molecular_weight
        super().__init__(**kwargs)
        
    
    def _real(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):
        
        
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        rho  = Gas(theta_ej = self.theta_ej, theta_co = self.theta_co, M_c = self.M_c, mu = self.mu, A = self.A, M1 = self.M1, eta_star = self.eta_star, epsilon = self.epsilon)
        rho  = rho.real(cosmo, r_use, M, a, mass_def = mass_def)
        prof = rho / (self.mean_molecular_weight * m_p) / (Mpc_to_m * m_to_cm)**3
        
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)

        assert np.all(prof >= 0), "Something went wrong. Temperature is negative in some places"

        return prof


class Temperature(Pressure):
    
    def _real(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):
        
        
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        P   = Pressure(nonthermal_model = self.nonthermal_model, epsilon = self.epsilon, a = self.a, n = self.n,
                                      theta_ej = self.theta_ej, theta_co = self.theta_co, M_c = self.M_c, mu = self.mu,
                                      A = self.A, M1 = self.M1, eta_star = self.eta_star, eta_cga = self.eta_cga, epsilon_h = self.epsilon_h,
                                      p = self.p, q = self.q, xi_mm = self.xi_mm)
        n   = GasNumberDensity(theta_ej = self.theta_ej, theta_co = self.theta_co, M_c = self.M_c, mu = self.mu, A = self.A, M1 = self.M1, eta_star = self.eta_star, epsilon = self.epsilon)

        P   = P.real(cosmo, r_use, M, a, mass_def = mass_def)
        n   = n.real(cosmo, r_use, M, a, mass_def = mass_def)
        
        prof = P/(n * kb_in_ev)
        
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)

        assert np.all(prof >= 0), "Something went wrong. Temperature is negative in some places"

        return prof


class CustomTemperature(ccl.halos.profiles.HaloProfile):
    
    def __init__(self, Pressure, GasNumberDensity):
        
        self.Pressure = Pressure
        self.GasNumberDensity = GasNumberDensity
        super().__init__()
        
    
    def _real(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):
        
        
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        P   = self.Pressure(nonthermal_model = self.nonthermal_model, epsilon = self.epsilon, a = self.a, n = self.n,
                                      theta_ej = self.theta_ej, theta_co = self.theta_co, M_c = self.M_c, mu = self.mu,
                                      A = self.A, M1 = self.M1, eta_star = self.eta_star, eta_cga = self.eta_cga, epsilon_h = self.epsilon_h,
                                      p = self.p, q = self.q, xi_mm = self.xi_mm)
        n   = self.GasNumberDensity(theta_ej = self.theta_ej, theta_co = self.theta_co, M_c = self.M_c, mu = self.mu, A = self.A, M1 = self.M1, eta_star = self.eta_star, epsilon = self.epsilon)

        P   = P.real(cosmo, r_use, M, a, mass_def = mass_def)
        n   = n.real(cosmo, r_use, M, a, mass_def = mass_def)
        
        prof = P/(n * kb_in_ev)
        
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)

        assert np.all(prof >= 0), "Something went wrong. Temperature is negative in some places"

        return prof
    
    
#Base class to generate SZ profiles
class ThermalSZ(object):
    
    
    def __init__(self, Pressure, epsilon_max = 3):
        
        self.Pressure    = Pressure
        self.epsilon_max = epsilon_max
        
    
    def Pgas_to_Pe(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):
        
        return Pth_to_Pe
    
    
    def projected(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):        
        
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        prof = sigma_T/(m_e*c**2) * a * self.Pressure.projected(cosmo, r, M, a, mass_def)
        prof = prof*self.Pgas_to_Pe(cosmo, r, M, a, mass_def)
        
        prof[r_use > R[:, None]*self.epsilon_max] = 0
        
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        
        return prof
    
    
    def real(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):
    
        return np.zeros_like(r)
    
    
    
class XrayLuminosity(ccl.halos.profiles.HaloProfile):
    
    
    def __init__(self, Temperature, GasNumberDensity):
        
        self.Temperature      = Temperature
        self.GasNumberDensity = GasNumberDensity
        self.epsilon_max      = epsilon_max
        
        super().__init__()
        
    
    def _real(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):
        
        
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        T   = self.Temperature.real(cosmo, r_use, M, a, mass_def = mass_def)
        n   = self.GasNumberDensity.real(cosmo, r_use, M, a, mass_def = mass_def)
        
        prof = n**2*T
        
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        
        return prof
