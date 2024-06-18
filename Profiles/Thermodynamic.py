
import numpy as np
import pyccl as ccl

from scipy import interpolate
from astropy.cosmology import z_at_value, FlatLambdaCDM, FlatwCDM
from astropy import units as u

from ..Profiles.Schneider19 import model_params, SchneiderProfiles, Gas, DarkMatterBaryon, TwoHalo


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

sigma_T_cgs = 6.652458e-29 / m_to_cm**2
m_e_cgs     = 9.10938e-31 * 1e3
m_p_cgs     = 1.67262e-27 * 1e3
c_cgs       = 2.99792458e8 * m_to_cm

#Thermodynamic/abundance quantities
Y         = 0.24 #Helium mass ratio
Pth_to_Pe = (4 - 2*Y)/(8 - 5*Y) #Factor to convert gas temp. to electron temp


#Technically P(r -> infty) is zero, but we  may need finite
#value for numerical reasons (interpolator). This is a
#computatational constant
Pressure_at_infinity = 0

class Pressure(SchneiderProfiles):
    """
    Computes the pressure of the GAS.
    Need to use additional factors to get the electron pressure.
    """
    
    def __init__(self, gas = None, darkmatterbaryon = None, nonthermal_model = None, **kwargs):
        
        self.Gas = gas
        self.DarkMatterBaryon = darkmatterbaryon
        
        #The subtraction in DMB case is so we only have the 1halo term
        if self.Gas is None: self.Gas = Gas(**kwargs)
        if self.DarkMatterBaryon is None: self.DarkMatterBaryon = DarkMatterBaryon(**kwargs) - TwoHalo(**kwargs)
            
        #Now make sure the cutoff is sufficiently high
        #We don't want small cutoff when computing the true pressure profile.
        self.Gas.set_parameter('cutoff', 1000)
        self.DarkMatterBaryon.set_parameter('cutoff', 1000)
            
        self.nonthermal_model = nonthermal_model
        super().__init__(**kwargs)
        
    
    def _real(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):
        
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        r_integral = np.geomspace(1e-3, 100, 500) #Hardcoded ranges

        rho_total  = self.DarkMatterBaryon.real(cosmo, r_integral, M, a, mass_def = mass_def)
        rho_gas    = self.Gas.real(cosmo, r_integral, M, a, mass_def = mass_def)
        
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
        
        #Convert to CGS
        prof = prof * (Msun_to_Kg * 1e3) / (Mpc_to_m * 1e2)
        
        
        #Now do cutoff
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
    


class NonThermalFrac(SchneiderProfiles):
    
    def __init__(self, **kwargs):
        
        self.alpha_nt = kwargs['alpha_nt']
        self.nu_nt    = kwargs['nu_nt']
        self.gamma_nt = kwargs['gamma_nt']
        
        super().__init__(**kwargs)
        
    
    def _real(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):
        
        
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        f_max = 6**-self.gamma_nt/self.alpha_nt
        f_z   = np.min([(1 + z)**self.nu_nt, (f_max - 1)*np.tanh(self.nu_nt * z) + 1])
        f_nt  = self.alpha_nt * f_z * (r_use/R[:, None])**self.gamma_nt
        f_nt  = np.clip(f_nt, 0, 1) #Enforce 0 < f_nt < 1
        prof  = f_nt #Rename just for consistency sake
        
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)

        return prof
    
    

class NonThermalFracGreen20(SchneiderProfiles):
    
    def _real(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):
        
        
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        
        #They define the model with R200m, so gotta use that redefinition here.
        M200m = mass_def.translate_mass(cosmo, M_use, a, ccl.halos.massdef.MassDef200m())
        R200m = ccl.halos.massdef.MassDef200m().get_radius(cosmo, M_use, a)/a #in comoving distance

        x = r_use/R200m[:, None]

        a, b, c, d, e, f = 0.495, 0.719, 1.417,-0.166, 0.265, -2.116
        nu_M = 1.686/ccl.sigmaM(cosmo, M200m, a)
        nu_M = nu_M[:, None]
        nth  = 1 - a * (1 + np.exp(-(x/b)**c)) * (nu_M/4.1)**(d/(1 + (x/e)**f))
        prof = nth #Rename just for consistency sake
        
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)

        return prof

    

class ElectronPressure(Pressure):
    
    def _real(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):
        
        prof = Pth_to_Pe * super()._real(cosmo, r, M, a, mass_def)
        
        return prof


    
class GasNumberDensity(SchneiderProfiles):
    
    def __init__(self, gas = None, mean_molecular_weight = 1.15, **kwargs):
        
        self.Gas = gas
        if self.Gas is None: self.Gas = Gas(**kwargs)
        
        self.mean_molecular_weight = mean_molecular_weight
        super().__init__(**kwargs)
        
    
    def _real(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):
        
        
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        rho  = self.Gas = Gas(**self.model_params)
        rho  = rho.real(cosmo, r_use, M, a, mass_def = mass_def)
        prof = rho / (self.mean_molecular_weight * m_p) / (Mpc_to_m * m_to_cm)**3
        
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)

        return prof
    
    
class Temperature(SchneiderProfiles):
    
    def __init__(self, pressure = None, gasnumberdensity = None, **kwargs):
        
        self.Pressure = pressure
        self.GasNumberDensity = gasnumberdensity
        
        if self.Pressure is None: self.Pressure = Pressure(**kwargs)
        if self.GasNumberDensity is None: self.GasNumberDensity = GasNumberDensity(**kwargs)
            
        super().__init__(**kwargs)
        
    
    def _real(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):
        
        
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        P   = self.Pressure.real(cosmo, r_use, M, a, mass_def = mass_def)
        n   = self.GasNumberDensity.real(cosmo, r_use, M, a, mass_def = mass_def)
        
        prof = P/(n * kb_in_ev)
        
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)

        assert np.all(prof >= 0), "Something went wrong. Temperature is negative in some places"

        return prof
    
    

class ThermalSZ(object):
    
    
    def __init__(self, Pressure = None):
        
        self.Pressure = Pressure
        if self.Pressure is None: self.Pressure = Pressure(**kwargs)
        
    
    def Pgas_to_Pe(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):
        
        return Pth_to_Pe
    
    
    def projected(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):        
        
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        prof = sigma_T_cgs/(m_e_cgs*c_cgs**2) * a * self.Pressure.projected(cosmo, r_use, M_use, a, mass_def)
        prof = prof*self.Pgas_to_Pe(cosmo, r_use, M_use, a, mass_def)
        
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        
        return prof
    
    
    def real(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):
        
        #Don't raise ValueError because then we can't pass this object in a TabulatedProfile class
        #Instead just output sentinel value of -99
    
        return np.ones_like(r) * -99
    
    
    
class XrayLuminosity(ccl.halos.profiles.HaloProfile):
    
    
    def __init__(self, temperature = None, gasnumberdensity = None):
        
        self.Temperature      = temperature
        self.GasNumberDensity = gasnumberdensity
        
        if self.Temperature is None: self.Temperature = Temperature(**kwargs)
        if self.GasNumberDensity is None: self.GasNumberDensity = GasNumberDensity(**kwargs)
        
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
