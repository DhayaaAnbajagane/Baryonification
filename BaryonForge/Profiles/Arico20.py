import numpy as np
import pyccl as ccl
from operator import add, mul, sub, truediv, pow, neg, pos, abs
import warnings

from scipy import interpolate, special
from ..utils.Tabulate import _set_parameter
from . import Schneider19 as S19
from .Thermodynamic import (G, Msun_to_Kg, Mpc_to_m, kb_cgs, m_p, m_to_cm)

__all__ = ['model_params', 'AricoProfiles', 
           'DarkMatter', 'TwoHalo', 'Stars', 'Gas', 'BoundGas', 'EjectedGas', 'ReaccretedGas', 'CollisionlessMatter',
           'DarkMatterOnly', 'DarkMatterBaryon', 'Pressure', 'NonThermalFrac', 'Temperature']


model_params = ['cdelta', 'a', 'n', #DM profle params and relaxation params
                'q', 'p', #Two Halo
                'cutoff', 'proj_cutoff', #Cutoff parameters (numerical)
                
                'theta_out', 'theta_inn', 'M_inn', 'M_c', 'mu', 'beta', 
                'M_r', 'beta_r', 'eta', 'theta_rg', 'sigma_rg', 'epsilon_hydro', #Default gas profile param

                'alpha_sat', 'M1_0', 'alpha_g', 'epsilon_h', #Star params

                'A_nt', 'alpha_nt', #Pressure params
                'mean_molecular_weight', #Gas number density params
               ]


projection_params = ['padding_lo_proj', 'padding_hi_proj', 'n_per_decade_proj'] #Projection params

class AricoProfiles(S19.SchneiderProfiles):
    """
    Base class for defining halo density profiles based on Schneider et al. models.

    This class extends the `ccl.halos.profiles.HaloProfile` class and provides 
    additional functionality for handling different halo density profiles. It allows 
    for custom real-space projection methods, control over parameter initialization, 
    and adjustments to the Fourier transform settings to minimize artifacts.

    Parameters
    ----------
    use_fftlog_projection : bool, optional
        If True, the default FFTLog projection method is used for the `projected` method. 
        If False, a custom real-space projection is employed. Default is False.
    padding_lo_proj : float, optional
        The lower padding factor for the projection integral in real-space. Default is 0.1.
    padding_hi_proj : float, optional
        The upper padding factor for the projection integral in real-space. Default is 10.
    n_per_decade_proj : int, optional
        Number of integration points per decade in the real-space projection integral. Default is 10.
    xi_mm : callable, optional
        A function that returns the matter-matter correlation function at different radii.
        Default is None, in which case we use the CCL inbuilt model.
    **kwargs
        Additional keyword arguments for setting specific parameters of the profile. If a parameter 
        is not specified, defaults are assigned based on its type (e.g., mass/redshift/conc-dependence).

    Attributes
    ----------
    model_params : dict
        A dictionary containing all model parameters and their values.
    precision_fftlog : dict
        Dictionary with precision settings for the FFTLog convolution. Can be modified 
        directly or using the update_precision_fftlog() method.

    Methods
    -------
    real(cosmo, r, M, a)
        Computes the real-space density profile.
    projected(cosmo, r, M, a)
        Computes the projected density profile.

    """

    #Define the new param names
    model_param_names      = model_params
    projection_param_names = projection_params

    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        
        #Go through all input params, and assign Nones to ones that don't exist.
        #If mass/redshift/conc-dependence, then set to 1 if don't exist
        for m in self.model_param_names + self.projection_param_names:
            if m in kwargs.keys():
                setattr(self, m, kwargs[m])
            else:
                setattr(self, m, None)

        #Sets the cutoff scale of all profiles, in comoving Mpc. Prevents divergence in FFTLog
        #Also set cutoff of projection integral. Should be the box side length
        self.cutoff      = kwargs['cutoff'] if 'cutoff' in kwargs.keys() else 1e3 #1Gpc is a safe default choice
        self.proj_cutoff = kwargs['proj_cutoff'] if 'proj_cutoff' in kwargs.keys() else self.cutoff
                
    
    def _get_gas_params(self, M, z):
        """
        Computes gas-related parameters based on the mass and redshift.
        Will use concentration is cdelta is specified during Class initialization.
        Uses mass/redshift slopes provided during class initialization.

        Parameters
        ----------
        M : array_like
            Halo mass or array of halo masses.
        z : float
            Redshift.

        Returns
        -------
        beta : ndarray
            Small-scale gas slope.
        theta_out : ndarray
            Ejection radius.
        theta_inn : ndarray
            Core radius parameter.
        delta : ndarray
            Large-scale slope.
        gamma : ndarray
            Intermediate-scale slope.
        """
        
        beta = 3 - (self.M_inn/M)**self.mu * np.ones_like(M)
        
        #Use M_c as the mass-normalization for simplicity sake
        theta_out = self.theta_out * np.ones_like(M) 
        theta_inn = self.theta_inn * np.ones_like(M)
        
        beta     = beta[:, None]
        theta_out = theta_out[:, None]
        theta_inn = theta_inn[:, None]
        
        return beta, theta_out, theta_inn
    

    def _get_star_frac(self, M, z, satellite = False):

        #Based on fitting function of Behroozi+2013 and data from Kravtsov+2018
        #see Eq A16-17 in https://arxiv.org/pdf/1911.08471
        M1_a    = -1.793
        M1_z    = -0.251
        eps_0   = np.log10(0.023)
        eps_a   = -0.006
        eps_a2  = -0.119
        alpha_0 = -1.779
        alpha_a = 0.731
        delta_0 = 4.394
        delta_a = 2.608
        delta_z = -0.043
        gamma_0 = 0.547
        gamma_a = 1.319
        gamma_z = 0.279

        a   = 1/(1 + z)
        nu  = np.exp(-4*a**2)
        M1  = self.M1_0 * np.power(10, (M1_a*(a - 1) + M1_z * z)*nu)
        eps = np.power(10, eps_0 + nu*(eps_a*(a - 1)) + eps_a2 * (a - 1))
        alpha = alpha_0 + nu*(alpha_a*(a - 1))
        delta = delta_0 + nu*(delta_a*(a - 1) + delta_z*z)
        gamma = gamma_0 + nu*(gamma_a*(a - 1) + gamma_z*z)

        if satellite:

            M1     = np.power(M1,  self.alpha_sat)
            eps    = np.power(eps, self.alpha_sat)
            alpha *= self.alpha_sat
            delta *= self.alpha_sat
            gamma *= self.alpha_sat

        x   = np.log10(M/M1)
        g_x = -np.log10(np.power(10, alpha * x) + 1) + delta * np.power(np.log10(1 + np.exp(x)), gamma)/(1 + np.exp(10**-x))
        g_0 = -np.log10(np.power(10, alpha * 0) + 1) + delta * np.power(np.log10(1 + np.exp(0)), gamma)/(1 + np.exp(10**-0))
        fCG = eps * (M1/M) * np.power(10, g_x - g_0)

        return fCG    

    def __str_par__(self):
        '''
        String with all input params and their values
        '''
        
        string = f"("
        for m in model_params:
            string += f"{m} = {self.__dict__[m]}, "
        string = string[:-2] + ')'
        return string


class DarkMatter(AricoProfiles):
    """
    Class representing the total Dark Matter (DM) profile using the NFW (Navarro-Frenk-White) profile.

    This class is derived from the `SchneiderProfiles` class and provides an implementation of the 
    dark matter profile based on the NFW model. It includes a custom `_real` method for calculating 
    the real-space dark matter density profile, considering factors like the concentration-mass 
    relation and truncation radius.

    See `AricoProfiles` for more docstring details.

    Notes
    -----
    The `DarkMatter` class calculates the dark matter density profile using the NFW model with a 
    modification for truncation at a specified radius set by `epsilon`. This profile accounts for the concentration-mass 
    relation, which can be provided as `cdelta` during class init. If none is provided,
    we use the `ConcentrationDiemer15` model.

    The profile also includes an additional exponential cutoff to prevent numerical overflow and 
    artifacts at large radii.

    The dark matter density profile is given by:

    .. math::

        \\rho_{\\text{DM}}(r) = \\frac{\\rho_c}{\\frac{r}{r_s} \\left(1 + \\frac{r}{r_s}\\right)^2} 
        \\cdot \\frac{1}{\\left(1 + \\frac{r}{r_t}\\right)^2}

    where:

    - :math:`\\rho_c` is the characteristic density of the halo.
    - :math:`r_s` is the scale radius of the halo, defined as :math:`r_s = R/c`.
    - :math:`r_t = \\epsilon \\cdot R` is the truncation radius, controlled by the parameter `epsilon`.
    - :math:`r` is the radial distance.


    Examples
    --------
    Create a `DarkMatter` profile and compute the density at specific radii:

    >>> dm_profile = DarkMatter(**parameters)
    >>> cosmo = ...  # Define or load a cosmology object
    >>> r = np.logspace(-2, 1, 50)  # Radii in comoving Mpc
    >>> M = 1e14  # Halo mass in solar masses
    >>> a = 0.5  # Scale factor corresponding to redshift z
    >>> density_profile = dm_profile.real(cosmo, r, M, a)
    """

    def _real(self, cosmo, r, M, a):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        if self.cdelta is None:
            c_M_relation = ccl.halos.concentration.ConcentrationDiemer15(mdef = self.mass_def) #Use the diemer calibration
        else:
            c_M_relation = ccl.halos.concentration.ConcentrationConstant(self.cdelta, mdef = self.mass_def)
            
        c   = c_M_relation.get_concentration(cosmo, M_use, a)
        R   = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc
        r_s = R/c

        #Get the normalization (rho_c) analytically since we don't have a truncation radii like S19 does
        Norm  = 4*np.pi*r_s**3 * (np.log(1 + c) - c/(1 + c))
        rho_c = M_use/Norm

        r_s, c, rho_c = r_s[:, None], c[:, None], rho_c[:, None]
        r_use, R      = r_use[None, :], R[:, None]


        arg  = (r_use - self.cutoff)
        arg  = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        prof = rho_c/(r_use/r_s * (1 + r_use/r_s)**2) * kfac
        prof = np.where(r_use <= R, prof, 0)
        
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)

        return prof


class TwoHalo(S19.TwoHalo, AricoProfiles):
    __doc__ = S19.TwoHalo.__doc__.replace('SchneiderProfiles', 'AricoProfiles')


class Stars(AricoProfiles):
    """
    Class representing the exponential stellar mass profile.

    This class is derived from the `SchneiderProfiles` class and provides an implementation 
    of an exponential stellar mass profile. It calculates the real-space stellar mass 
    density profile, using parameters to account for factors like stellar mass fraction 
    and halo radius.

    See `SchneiderProfiles` for more docstring details.

    Notes
    -----
    The `Stars` class models the stellar mass distribution with an exponential profile, 
    modulated by parameters such as `eta`, `tau`, `A`, and `M1`. These parameters 
    adjust the stellar mass fraction as a function of halo mass. The profile also applies 
    an exponential cutoff controlled by the `epsilon_h` parameter to define the 
    characteristic radius of the stellar distribution.

    The stellar mass density profile is given by:

    .. math::

        \\rho_\\star(r) = \\frac{f_{\\text{cga}} M_{\\text{tot}}}{4 \\pi^{3/2} R_h} \\frac{1}{r^2} 
                          \\exp\\left(-\\frac{r^2}{4 R_h^2}\\right) 

    where:

    - :math:`f_{\\text{cga}}` is the stellar mass fraction, defined as:

      .. math::

          f_{\\text{cga}} = 2 A \\left(\\left(\\frac{M}{M_1}\\right)^{\\tau + \\tau_\\delta} 
          + \\left(\\frac{M}{M_1}\\right)^{\\eta + \\eta_\\delta}\\right)^{-1}

    - :math:`M_{\\text{tot}}` is the total halo mass.
    - :math:`R_h = \\epsilon_h R` is the characteristic scale radius of the stellar distribution.
    - :math:`r` is the radial distance.

    The class overrides specific `precision_fftlog` settings to prevent ringing artifacts 
    in the profiles. This is achieved by setting extreme padding values.

    An additional exponential cutoff is included to prevent numerical overflow and artifacts 
    at large radii.

    Examples
    --------
    Create a `Stars` profile and compute the density at specific radii:

    >>> stars_profile = Stars(**parameters)
    >>> cosmo = ...  # Define or load a cosmology object
    >>> r = np.logspace(-2, 1, 50)  # Radii in comoving Mpc
    >>> M = 1e14  # Halo mass in solar masses
    >>> a = 0.5  # Scale factor corresponding to redshift z
    >>> density_profile = stars_profile.real(cosmo, r, M, a)
    """
    
    def __init__(self, **kwargs):
        
        AricoProfiles.__init__(self, **kwargs)
        
        #For some reason, we need to make this extreme in order
        #to prevent ringing in the profiles. Haven't figured out
        #why this is the case
        self.update_precision_fftlog(padding_lo_fftlog = 1e-5, padding_hi_fftlog = 1e5)

    
    def _real(self, cosmo, r, M, a):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        R     = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc
        z     = 1/a - 1

        f_cga = self._get_star_frac(M_use, z)[:, None]
        R_h   = self.epsilon_h * R[:, None]

        #Integrate over wider region in radii to get normalization of star profile
        #There's no way the profile has any support than 5Mpc. So use a narrower range.
        r_integral    = np.geomspace(1e-6, 5, 500)
        prof_integral = 1 / R_h / np.power(r_integral, self.alpha_g) * np.exp(-np.power(r_integral/2/R_h, 2))
        Normalization = np.trapz(4 * np.pi * r_integral**2 * prof_integral, r_integral, axis = -1)[:, None]
        
        #Final profile. No truncation needed since exponential cutoff already does that for us
        prof = f_cga*M_use[:, None] / R_h / np.power(r_use, self.alpha_g) * np.exp(-np.power(r_use/2/R_h, 2)) / Normalization

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)

        return prof


class BoundGas(AricoProfiles):

    """
    Class representing the gas density profile.

    This class is derived from the `SchneiderProfiles` class and provides an implementation 
    of a gas density profile. It calculates the real-space gas density profile, using
    the general NFW (GNFW) model of `Nagai, Kravtsov & Vikhlinin 2009 <https://arxiv.org/pdf/astro-ph/0703661>`_.

    See `SchneiderProfiles` for more docstring details.

    Notes
    -----
    The `Gas` class models the gas distribution in halos by considering the gas fraction, 
    which is computed based on the total baryonic fraction minus the stellar fraction. 
    The gas density profile is defined using parameters such as `beta`, `delta`, `gamma`, 
    `theta_inn`, and `theta_out`. These parameters characterize the core and ejection properties 
    of the gas distribution.

    The gas density profile is given by:

    .. math::

        \\rho_{\\text{gas}}(r) = \\frac{f_{\\text{gas}} M_{\\text{tot}}}{N} \\cdot 
        \\frac{1}{(1 + u)^{\\beta}} \\cdot \\frac{1}{(1 + v)^{(\\delta - \\beta)/\\gamma}}

    where:

    - :math:`f_{\\text{gas}} = f_{\\text{bar}} - f_{\\star}` is the gas fraction.
    - :math:`f_{\\text{bar}}` is the cosmic baryon fraction.
    - :math:`f_{\\star}` is the stellar mass fraction, defined as:

      .. math::

          f_{\\star} = 2A \\left(\\left(\\frac{M}{M_1}\\right)^{\\tau} + \\left(\\frac{M}{M_1}\\right)^{\\eta}\\right)^{-1}

    - :math:`M_{\\text{tot}}` is the total halo mass.
    - :math:`N` is the normalization factor to ensure mass conservation.
    - :math:`u = \\frac{r}{R_{\\text{co}}}` and :math:`v = \\frac{r}{R_{\\text{ej}}}` are dimensionless radii.
    - :math:`\\beta` is the power-law slope for :math:`R_{\\text{co}} \lesssim r \lesssim R_{\\text{ej}}`
    - :math:`\\delta` is the power-law slope at :math:`r \sim \lesssim R_{\\text{ej}}`
    - :math:`\\gamma` is the power-law slope for :math:`r \gg R_{\\text{ej}}`
    - :math:`R_{\\text{co}} = \\theta_{\\text{co}} R` is the core radius.
    - :math:`R_{\\text{ej}} = \\theta_{\\text{ej}} R` is the ejection radius.
    - :math:`r` is the radial distance.

    Examples
    --------
    Create a `Gas` profile and compute the density at specific radii:

    >>> gas_profile = Gas(**parameters)
    >>> cosmo = ...  # Define or load a cosmology object
    >>> r = np.logspace(-2, 1, 50)  # Radii in comoving Mpc
    >>> M = 1e14  # Halo mass in solar masses
    >>> a = 0.5  # Scale factor corresponding to redshift z
    >>> density_profile = gas_profile.real(cosmo, r, M, a)
    """

    def _real(self, cosmo, r, M, a):


        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        f_cg  = self._get_star_frac(M_use, z)
        f_sg  = self._get_star_frac(M_use, z, satellite = True)
        f_bar = cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m
        beta  = self._get_gas_params(M_use, z)[0]
        f_hg  = (f_bar - f_cg - f_sg) / (1 + np.power(self.M_c/M_use, self.beta))
        f_rg  = (f_bar - f_cg - f_sg - f_hg) / (1 + np.power(self.M_r/M_use, self.beta_r))
        f_bg  = (f_hg - f_rg)

        #Get gas params
        beta, theta_out, theta_inn = self._get_gas_params(M_use, z)
        R_co = theta_inn*R[:, None]
        R_ej = theta_out*R[:, None]
        f_bg = f_bg[:, None]
        
        u = r_use/R_co
        v = r_use/R_ej
        
        #Integrate over wider region in radii to get normalization of gas profile
        #Using a number narrower range than Schneider cause we only need to go to R200c
        r_integral = np.geomspace(1e-6, 10, 500)
        u_integral = r_integral/R_co
        v_integral = r_integral/R_ej        

        prof_integral = 1/(1 + u_integral)**beta / (1 + v_integral**2)**2
        prof_integral = np.where(r_integral[None, :] < R[:, None], prof_integral, 0)
        Normalization = np.trapz(4 * np.pi * r_integral**2 * prof_integral, r_integral, axis = -1)[:, None]

        del u_integral, v_integral, prof_integral

        arg   = (r_use[None, :] - self.cutoff)
        arg   = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac  = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        prof  = 1/(1 + u)**beta / (1 + v**2)**2 * kfac
        prof *= f_bg*M_use[:, None]/Normalization #This profile is allowed to go beyond R200c!

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)


        return prof
    

class EjectedGas(AricoProfiles):

    """
    Class representing the gas density profile.

    This class is derived from the `SchneiderProfiles` class and provides an implementation 
    of a gas density profile. It calculates the real-space gas density profile, using
    the general NFW (GNFW) model of `Nagai, Kravtsov & Vikhlinin 2009 <https://arxiv.org/pdf/astro-ph/0703661>`_.

    See `SchneiderProfiles` for more docstring details.

    Notes
    -----
    The `Gas` class models the gas distribution in halos by considering the gas fraction, 
    which is computed based on the total baryonic fraction minus the stellar fraction. 
    The gas density profile is defined using parameters such as `beta`, `delta`, `gamma`, 
    `theta_inn`, and `theta_out`. These parameters characterize the core and ejection properties 
    of the gas distribution.

    The gas density profile is given by:

    .. math::

        \\rho_{\\text{gas}}(r) = \\frac{f_{\\text{gas}} M_{\\text{tot}}}{N} \\cdot 
        \\frac{1}{(1 + u)^{\\beta}} \\cdot \\frac{1}{(1 + v)^{(\\delta - \\beta)/\\gamma}}

    where:

    - :math:`f_{\\text{gas}} = f_{\\text{bar}} - f_{\\star}` is the gas fraction.
    - :math:`f_{\\text{bar}}` is the cosmic baryon fraction.
    - :math:`f_{\\star}` is the stellar mass fraction, defined as:

      .. math::

          f_{\\star} = 2A \\left(\\left(\\frac{M}{M_1}\\right)^{\\tau} + \\left(\\frac{M}{M_1}\\right)^{\\eta}\\right)^{-1}

    - :math:`M_{\\text{tot}}` is the total halo mass.
    - :math:`N` is the normalization factor to ensure mass conservation.
    - :math:`u = \\frac{r}{R_{\\text{co}}}` and :math:`v = \\frac{r}{R_{\\text{ej}}}` are dimensionless radii.
    - :math:`\\beta` is the power-law slope for :math:`R_{\\text{co}} \lesssim r \lesssim R_{\\text{ej}}`
    - :math:`\\delta` is the power-law slope at :math:`r \sim \lesssim R_{\\text{ej}}`
    - :math:`\\gamma` is the power-law slope for :math:`r \gg R_{\\text{ej}}`
    - :math:`R_{\\text{co}} = \\theta_{\\text{co}} R` is the core radius.
    - :math:`R_{\\text{ej}} = \\theta_{\\text{ej}} R` is the ejection radius.
    - :math:`r` is the radial distance.

    Examples
    --------
    Create a `Gas` profile and compute the density at specific radii:

    >>> gas_profile = Gas(**parameters)
    >>> cosmo = ...  # Define or load a cosmology object
    >>> r = np.logspace(-2, 1, 50)  # Radii in comoving Mpc
    >>> M = 1e14  # Halo mass in solar masses
    >>> a = 0.5  # Scale factor corresponding to redshift z
    >>> density_profile = gas_profile.real(cosmo, r, M, a)
    """

    def _real(self, cosmo, r, M, a):


        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1
        R = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        f_cg  = self._get_star_frac(M_use, z)
        f_sg  = self._get_star_frac(M_use, z, satellite = True)
        f_bar = cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m
        beta  = self._get_gas_params(M_use, z)[0]
        f_hg  = (f_bar - f_cg - f_sg) / (1 + np.power(self.M_c/M_use, self.beta))
        f_eg  =  f_bar - f_cg - f_sg - f_hg
        f_eg  = f_eg[:, None]

        #Now use the escape radius, which is r_esc = v_esc * t_hubble
        #and this reduces down to just 1/2 * sqrt(Delta) * R_Delta
        R_esc = 1/2 * np.sqrt(200) * R
        R_ej  = self.eta * 0.75 * R_esc
        R_ej  = R_ej[:, None]

        arg   = (r_use[None, :] - self.cutoff)
        arg   = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac  = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        prof  = f_eg * M_use[:, None] / np.power(2*np.pi*R_ej**2, 3/2) * np.exp(-np.power(r_use/R_ej, 2)/2) * kfac

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)

        return prof



class ReaccretedGas(AricoProfiles):

    """
    Class representing the gas density profile.

    This class is derived from the `SchneiderProfiles` class and provides an implementation 
    of a gas density profile. It calculates the real-space gas density profile, using
    the general NFW (GNFW) model of `Nagai, Kravtsov & Vikhlinin 2009 <https://arxiv.org/pdf/astro-ph/0703661>`_.

    See `SchneiderProfiles` for more docstring details.

    Notes
    -----
    The `Gas` class models the gas distribution in halos by considering the gas fraction, 
    which is computed based on the total baryonic fraction minus the stellar fraction. 
    The gas density profile is defined using parameters such as `beta`, `delta`, `gamma`, 
    `theta_inn`, and `theta_out`. These parameters characterize the core and ejection properties 
    of the gas distribution.

    The gas density profile is given by:

    .. math::

        \\rho_{\\text{gas}}(r) = \\frac{f_{\\text{gas}} M_{\\text{tot}}}{N} \\cdot 
        \\frac{1}{(1 + u)^{\\beta}} \\cdot \\frac{1}{(1 + v)^{(\\delta - \\beta)/\\gamma}}

    where:

    - :math:`f_{\\text{gas}} = f_{\\text{bar}} - f_{\\star}` is the gas fraction.
    - :math:`f_{\\text{bar}}` is the cosmic baryon fraction.
    - :math:`f_{\\star}` is the stellar mass fraction, defined as:

      .. math::

          f_{\\star} = 2A \\left(\\left(\\frac{M}{M_1}\\right)^{\\tau} + \\left(\\frac{M}{M_1}\\right)^{\\eta}\\right)^{-1}

    - :math:`M_{\\text{tot}}` is the total halo mass.
    - :math:`N` is the normalization factor to ensure mass conservation.
    - :math:`u = \\frac{r}{R_{\\text{co}}}` and :math:`v = \\frac{r}{R_{\\text{ej}}}` are dimensionless radii.
    - :math:`\\beta` is the power-law slope for :math:`R_{\\text{co}} \lesssim r \lesssim R_{\\text{ej}}`
    - :math:`\\delta` is the power-law slope at :math:`r \sim \lesssim R_{\\text{ej}}`
    - :math:`\\gamma` is the power-law slope for :math:`r \gg R_{\\text{ej}}`
    - :math:`R_{\\text{co}} = \\theta_{\\text{co}} R` is the core radius.
    - :math:`R_{\\text{ej}} = \\theta_{\\text{ej}} R` is the ejection radius.
    - :math:`r` is the radial distance.

    Examples
    --------
    Create a `Gas` profile and compute the density at specific radii:

    >>> gas_profile = Gas(**parameters)
    >>> cosmo = ...  # Define or load a cosmology object
    >>> r = np.logspace(-2, 1, 50)  # Radii in comoving Mpc
    >>> M = 1e14  # Halo mass in solar masses
    >>> a = 0.5  # Scale factor corresponding to redshift z
    >>> density_profile = gas_profile.real(cosmo, r, M, a)
    """

    def _real(self, cosmo, r, M, a):


        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        f_cg  = self._get_star_frac(M_use, z)
        f_sg  = self._get_star_frac(M_use, z, satellite = True)
        f_bar = cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m
        beta  = self._get_gas_params(M_use, z)[0]
        f_hg  = (f_bar - f_cg - f_sg) / (1 + np.power(self.M_c/M_use, self.beta))
        f_rg  = (f_bar - f_cg - f_sg - f_hg) / (1 + np.power(self.M_r/M_use, self.beta_r))
        
        #Get gas params
        R_rg = self.theta_rg*R[:, None]
        S_rg = self.sigma_rg*R[:, None]
        f_rg = f_rg[:, None]
        R    = R[:, None]
        
        #Can get normalization analytically
        t1   =  2 * np.sqrt(2 * np.pi) * (np.exp(-R_rg**2 / (2 * S_rg**2)) * R_rg - np.exp(-(R_rg - R)**2 / (2 * S_rg**2)) * (R_rg + R))
        t2   =  2 * np.pi * (R_rg**2 + S_rg**2) * special.erf(R_rg / (np.sqrt(2) * S_rg))
        t3   = -2 * np.pi * (R_rg**2 + S_rg**2) * special.erf((R_rg - R) / (np.sqrt(2) * S_rg))
        Norm = t1 * S_rg + t2 + t3

        arg   = (r_use[None, :] - self.cutoff)
        arg   = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac  = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        prof  = 1/np.sqrt(2*np.pi*S_rg**2) * np.exp(-np.power((r_use - R_rg)/S_rg, 2)/2)
        prof *= f_rg*M_use[:, None]/Norm
        prof  = np.where(r_use[None, :] <= R, prof, 0)

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)


        return prof
    

class Gas(AricoProfiles):
    '''
    Convenience class that combines the Bound, Ejected, and Reaccreted gas components
    '''
    def __init__(self, **kwargs): self.myprof = BoundGas(**kwargs) + EjectedGas(**kwargs) + ReaccretedGas(**kwargs)
    def __getattr__(self, name):  return getattr(self.myprof, name)
    
    #Need to explicitly set these two methods (to enable pickling)
    #since otherwise the getattr call above leads to infinite recursions.
    def __getstate__(self): self.__dict__.copy()    
    def __setstate__(self, state): self.__dict__.update(state)


class ModifiedDarkMatter(AricoProfiles):

    """
    Class representing the gas density profile.

    This class is derived from the `SchneiderProfiles` class and provides an implementation 
    of a gas density profile. It calculates the real-space gas density profile, using
    the general NFW (GNFW) model of `Nagai, Kravtsov & Vikhlinin 2009 <https://arxiv.org/pdf/astro-ph/0703661>`_.

    See `SchneiderProfiles` for more docstring details.

    Notes
    -----
    The `Gas` class models the gas distribution in halos by considering the gas fraction, 
    which is computed based on the total baryonic fraction minus the stellar fraction. 
    The gas density profile is defined using parameters such as `beta`, `delta`, `gamma`, 
    `theta_inn`, and `theta_out`. These parameters characterize the core and ejection properties 
    of the gas distribution.

    The gas density profile is given by:

    .. math::

        \\rho_{\\text{gas}}(r) = \\frac{f_{\\text{gas}} M_{\\text{tot}}}{N} \\cdot 
        \\frac{1}{(1 + u)^{\\beta}} \\cdot \\frac{1}{(1 + v)^{(\\delta - \\beta)/\\gamma}}

    where:

    - :math:`f_{\\text{gas}} = f_{\\text{bar}} - f_{\\star}` is the gas fraction.
    - :math:`f_{\\text{bar}}` is the cosmic baryon fraction.
    - :math:`f_{\\star}` is the stellar mass fraction, defined as:

      .. math::

          f_{\\star} = 2A \\left(\\left(\\frac{M}{M_1}\\right)^{\\tau} + \\left(\\frac{M}{M_1}\\right)^{\\eta}\\right)^{-1}

    - :math:`M_{\\text{tot}}` is the total halo mass.
    - :math:`N` is the normalization factor to ensure mass conservation.
    - :math:`u = \\frac{r}{R_{\\text{co}}}` and :math:`v = \\frac{r}{R_{\\text{ej}}}` are dimensionless radii.
    - :math:`\\beta` is the power-law slope for :math:`R_{\\text{co}} \lesssim r \lesssim R_{\\text{ej}}`
    - :math:`\\delta` is the power-law slope at :math:`r \sim \lesssim R_{\\text{ej}}`
    - :math:`\\gamma` is the power-law slope for :math:`r \gg R_{\\text{ej}}`
    - :math:`R_{\\text{co}} = \\theta_{\\text{co}} R` is the core radius.
    - :math:`R_{\\text{ej}} = \\theta_{\\text{ej}} R` is the ejection radius.
    - :math:`r` is the radial distance.

    Examples
    --------
    Create a `Gas` profile and compute the density at specific radii:

    >>> gas_profile = Gas(**parameters)
    >>> cosmo = ...  # Define or load a cosmology object
    >>> r = np.logspace(-2, 1, 50)  # Radii in comoving Mpc
    >>> M = 1e14  # Halo mass in solar masses
    >>> a = 0.5  # Scale factor corresponding to redshift z
    >>> density_profile = gas_profile.real(cosmo, r, M, a)
    """

    def __init__(self, gas = None, gravityonly = None, **kwargs):
        
        self.Gas   = gas
        self.GravityOnly = gravityonly
        
        if self.Gas is None: self.Gas = BoundGas(**kwargs) 
        if self.GravityOnly is None: self.GravityOnly = DarkMatter(**kwargs)

        AricoProfiles.__init__(self, **kwargs)
    

    def _safe_Pchip_minimize(self, x, y):

        assert (np.min(x) < 0) & (np.max(x) > 0), f"Cannot minimize. Range {np.min(x)} < LHS - RHS {np.max(x)} does not include zero!"
        ind = np.argmin(np.abs(x - 0)) #Find the point around which we should search for minima
        buf = 5 #Large enough (one-sided) buffer in case any weird interpolator effects from using too few points
        ind = slice(ind - buf, ind + buf)

        return interpolate.PchipInterpolator(x[ind], y[ind])(0)
     
    def _real(self, cosmo, r, M, a):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        if self.cdelta is None:
            c_M_relation = ccl.halos.concentration.ConcentrationDiemer15(mdef = self.mass_def) #Use the diemer calibration
        else:
            c_M_relation = ccl.halos.concentration.ConcentrationConstant(self.cdelta, mdef = self.mass_def)
            
        c   = c_M_relation.get_concentration(cosmo, M_use, a)
        R   = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc
        r_s = R/c        
        r_s = r_s[:, None]
        fDM = 1 - cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m

        #Solving equation A10 of https://arxiv.org/pdf/1911.08471 through minimization
        rp    = np.geomspace(1e-6, 10, 500)
        pGro  = np.array([self.GravityOnly.real(cosmo, r, m, a) for r, m in zip(R, M_use)])[:, None]
        pBG   = np.array([self.Gas.real(cosmo, r, m, a) for r, m in zip(R, M_use)])[:, None]
        LHS   = rp * np.power(rp + r_s, 2) * (pGro - pBG) * (np.log(1 + rp/r_s) - 1/(1 + r_s/rp)) + (pGro - pBG)/3 * (R**3 - rp**3)
        RHS   = fDM * M_use[None, :] / (4*np.pi)
        rp    = np.exp([self._safe_Pchip_minimize((LHS - RHS)[m_i], np.log(rp)) for m_i in range(LHS.shape[0])])[:, None]
        
        #Get the normalization based on equation A8 of https://arxiv.org/pdf/1911.08471
        rho_c = (pGro - pBG) * (rp/r_s) * np.power(1 + rp/r_s, 2)

        #Now the final profile
        prof  = rho_c / (r_use/r_s) / np.power(1 + r_use/r_s, 2)
        prof  = np.where(r_use[None, :] < rp, prof, (pGro - pBG))
        
        arg   = (r_use[None, :] - self.cutoff)
        arg   = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac  = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        prof  = prof * kfac
        prof  = np.where(r_use[None, :] <= R, prof, 0)

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)
        
        return prof


class CollisionlessMatter(AricoProfiles):
    __doc__ = S19.CollisionlessMatter.__doc__.replace('SchneiderProfiles', 'AricoProfiles')
    
    def __init__(self, gas = None, stars = None, darkmatter = None, max_iter = 10, reltol = 1e-2, r_min_int = 1e-8, r_max_int = 1e1, r_steps = 5000, **kwargs):
        
        self.Gas   = gas
        self.Stars = stars
        self.DarkMatter = darkmatter
        
        if self.Gas is None: self.Gas = Gas(**kwargs) 
        if self.Stars is None: self.Stars = Stars(**kwargs)
        if self.DarkMatter is None: self.DarkMatter = ModifiedDarkMatter(**kwargs) #Arico uses modified DM as default
            
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
        
        AricoProfiles.__init__(self, **kwargs)


    def _real(self, cosmo, r, M, a):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        if np.min(r) < self.r_min_int: 
            warnings.warn(f"Decrease integral lower limit, r_min_int ({self.r_min_int}) < minimum radius ({np.min(r)})", UserWarning)
        if np.max(r) > self.r_max_int: 
            warnings.warn(f"Increase integral upper limit, r_max_int ({self.r_max_int}) < maximum radius ({np.max(r)})", UserWarning)

        z = 1/a - 1
        R = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        f_sg   = self._get_star_frac(M_use, z, satellite = True)
        f_dm   = 1 - cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m
        f_clm  = f_dm + f_sg
        
        rho_clm = np.ones([M_use.shape[0], r_use.shape[0]], dtype = float)
        for m_i in range(M_use.shape[0]):

            #Def radius sampling for doing iteration.
            #This is different from Schneider version, because we have to do everything
            #halo by halo due to the sharp truncation radius (which induces oscillations,
            #in the cubic interpolations otherwise). Make sure the lower bound is always
            #sufficiently wide though
            r_integral = np.geomspace(self.r_min_int, R[m_i], self.r_steps)
            safe_range = (r_integral > 2 * np.min(r_integral) )

            #The DarkMatter profile may already have the f_dm normalization, but
            #this doesn't matter since we anyway renormalize the profiles later so it
            #gets to M_clm(<R200c) = M_tot * f_clm
            rho_i      = self.DarkMatter.real(cosmo, r_integral, M_use[m_i], a)
            rho_cga    = self.Stars.real(cosmo, r_integral, M_use[m_i], a)
            rho_gas    = self.Gas.real(cosmo, r_integral, M_use[m_i], a)

            dlnr  = np.log(r_integral[1]) - np.log(r_integral[0])
            M_i   = 4 * np.pi * np.cumsum(r_integral**3 * rho_i   * dlnr)
            M_cga = 4 * np.pi * np.cumsum(r_integral**3 * rho_cga * dlnr)
            M_gas = 4 * np.pi * np.cumsum(r_integral**3 * rho_gas * dlnr)

            #Assume extrapolation is used only for r > r_max. In this case, the extrapolation
            #coefficients are just the integrated mass at r_max. Our r_min is sufficientyly
            #low that we will not suffer extrapolation errors there (and even if we do it
            #should not matter at all given the infinitesimal volume element)
            M_i_max   = M_i[-1]
            M_cga_max = M_cga[-1]
            M_gas_max = M_gas[-1]

            #Set Extrapolate = False. We only need to extrapolate if r > R200c, where profile should be 0
            #and mass should be M(<R200c) so we'll just set it to that
            ln_M_NFW = interpolate.PchipInterpolator(np.log(r_integral), np.log(M_i),   extrapolate = False)
            ln_M_cga = interpolate.PchipInterpolator(np.log(r_integral), np.log(M_cga), extrapolate = False)
            ln_M_gas = interpolate.PchipInterpolator(np.log(r_integral), np.log(M_gas), extrapolate = False)

            del M_cga, M_gas, rho_i, rho_cga, rho_gas

            relaxation_fraction = np.ones_like(M_i)            
            counter      = 0
            max_rel_diff = np.inf #Initializing variable at infinity
            
            while max_rel_diff > self.reltol:

                with np.errstate(over = 'ignore'):
                    r_f  = r_integral*relaxation_fraction
                    M_f1 = f_clm[m_i]*M_i
                    M_f2 = np.exp(ln_M_cga(np.log(r_f)))
                    M_f3 = np.exp(ln_M_gas(np.log(r_f)))
                    M_f  = (np.where(np.isfinite(M_f1), M_f1, M_i_max) + 
                            np.where(np.isfinite(M_f2), M_f2, M_cga_max) + 
                            np.where(np.isfinite(M_f3), M_f3, M_gas_max)
                            )

                #Solve for the relaxation fraction following Equation A11 in https://arxiv.org/pdf/1911.08471
                relaxation_fraction_new = 1 + self.a*(np.power(M_i/M_f, self.n) - 1)

                #Normalize so the relaxation is at 1 at R200c
                #then make sure no r_f is greater than R200c
                norm = np.interp(R[m_i], r_integral, relaxation_fraction_new)
                relaxation_fraction_new /= norm

                diff     = relaxation_fraction_new/relaxation_fraction - 1
                abs_diff = np.abs(diff)
                
                max_rel_diff = np.max(abs_diff[safe_range])
                
                relaxation_fraction = relaxation_fraction_new * 1 #Multiple to avoid pointer assignment

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

            
            #Compute the relaxed DM profile, and the normalize so it 
            #has the right mass fraction within R200c.
            ln_M_clm  = np.log(f_clm[m_i]) + ln_M_NFW(np.log(r_integral/relaxation_fraction))
            ln_M_clm += np.log(f_clm[m_i] * M_use[m_i]) - np.interp(np.log(R[m_i]), np.log(r_integral), ln_M_clm)
            

            log_M    = interpolate.CubicSpline(np.log(r_integral), ln_M_clm, extrapolate = False)
            log_der  = log_M.derivative(nu = 1)(np.log(r_integral))
            lin_der  = log_der * np.exp(ln_M_clm) / r_integral
            prof     = 1/(4*np.pi*r_integral**2) * lin_der
            prof     = interpolate.PchipInterpolator(np.log(r_integral), prof, extrapolate = False)(np.log(r_use))
            
            arg  = (r_use - self.cutoff)
            arg  = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
            kfac = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
            prof = np.where(np.isnan(prof), 0, prof) * kfac
            prof = np.where(r_use <= R[m_i], prof, 0)

            rho_clm[m_i] = prof
        
        prof = rho_clm #Pointer just so naming is all consistent
        

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)

        return prof


class DarkMatterOnly(S19.DarkMatterOnly, AricoProfiles):

    __doc__ = S19.DarkMatterOnly.__doc__.replace('SchneiderProfiles', 'AricoProfiles')

    def __init__(self, darkmatter = None, **kwargs):
        
        self.DarkMatter = darkmatter
        self.TwoHalo    = TwoHalo(**kwargs) * 0 #Should not add 2-halo in Arico method

        if self.DarkMatter is None: self.DarkMatter = DarkMatter(**kwargs)
            
        AricoProfiles.__init__(self, **kwargs)


class DarkMatterBaryon(S19.DarkMatterBaryon, AricoProfiles):

    __doc__ = S19.DarkMatterBaryon.__doc__.replace('SchneiderProfiles', 'AricoProfiles')

    def __init__(self, gas = None, stars = None, collisionlessmatter = None, darkmatter = None, **kwargs):
        
        self.Gas   = gas
        self.Stars = stars
        self.TwoHalo    = TwoHalo(**kwargs) * 0 #Should not add 2-halo in Arico method
        self.DarkMatter = darkmatter
        self.CollisionlessMatter = collisionlessmatter
        
        if self.Gas is None:        self.Gas        = Gas(**kwargs)        
        if self.Stars is None:      self.Stars      = Stars(**kwargs)
        if self.DarkMatter is None: self.DarkMatter = DarkMatter(**kwargs)
        if self.CollisionlessMatter is None: self.CollisionlessMatter = CollisionlessMatter(**kwargs)

        AricoProfiles.__init__(self, **kwargs)
    

class ExtendedBoundGas(AricoProfiles):

    """
    Class representing the gas density profile.

    This class is derived from the `SchneiderProfiles` class and provides an implementation 
    of a gas density profile. It calculates the real-space gas density profile, using
    the general NFW (GNFW) model of `Nagai, Kravtsov & Vikhlinin 2009 <https://arxiv.org/pdf/astro-ph/0703661>`_.

    See `SchneiderProfiles` for more docstring details.

    Notes
    -----
    The `Gas` class models the gas distribution in halos by considering the gas fraction, 
    which is computed based on the total baryonic fraction minus the stellar fraction. 
    The gas density profile is defined using parameters such as `beta`, `delta`, `gamma`, 
    `theta_inn`, and `theta_out`. These parameters characterize the core and ejection properties 
    of the gas distribution.

    The gas density profile is given by:

    .. math::

        \\rho_{\\text{gas}}(r) = \\frac{f_{\\text{gas}} M_{\\text{tot}}}{N} \\cdot 
        \\frac{1}{(1 + u)^{\\beta}} \\cdot \\frac{1}{(1 + v)^{(\\delta - \\beta)/\\gamma}}

    where:

    - :math:`f_{\\text{gas}} = f_{\\text{bar}} - f_{\\star}` is the gas fraction.
    - :math:`f_{\\text{bar}}` is the cosmic baryon fraction.
    - :math:`f_{\\star}` is the stellar mass fraction, defined as:

      .. math::

          f_{\\star} = 2A \\left(\\left(\\frac{M}{M_1}\\right)^{\\tau} + \\left(\\frac{M}{M_1}\\right)^{\\eta}\\right)^{-1}

    - :math:`M_{\\text{tot}}` is the total halo mass.
    - :math:`N` is the normalization factor to ensure mass conservation.
    - :math:`u = \\frac{r}{R_{\\text{co}}}` and :math:`v = \\frac{r}{R_{\\text{ej}}}` are dimensionless radii.
    - :math:`\\beta` is the power-law slope for :math:`R_{\\text{co}} \lesssim r \lesssim R_{\\text{ej}}`
    - :math:`\\delta` is the power-law slope at :math:`r \sim \lesssim R_{\\text{ej}}`
    - :math:`\\gamma` is the power-law slope for :math:`r \gg R_{\\text{ej}}`
    - :math:`R_{\\text{co}} = \\theta_{\\text{co}} R` is the core radius.
    - :math:`R_{\\text{ej}} = \\theta_{\\text{ej}} R` is the ejection radius.
    - :math:`r` is the radial distance.

    Examples
    --------
    Create a `Gas` profile and compute the density at specific radii:

    >>> gas_profile = Gas(**parameters)
    >>> cosmo = ...  # Define or load a cosmology object
    >>> r = np.logspace(-2, 1, 50)  # Radii in comoving Mpc
    >>> M = 1e14  # Halo mass in solar masses
    >>> a = 0.5  # Scale factor corresponding to redshift z
    >>> density_profile = gas_profile.real(cosmo, r, M, a)
    """

    def _real(self, cosmo, r, M, a):


        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        f_cg  = self._get_star_frac(M_use, z)
        f_sg  = self._get_star_frac(M_use, z, satellite = True)
        f_bar = cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m
        beta  = self._get_gas_params(M_use, z)[0]
        f_hg  = (f_bar - f_cg - f_sg) / (1 + np.power(self.M_c/M_use, self.beta))
        f_rg  = (f_bar - f_cg - f_sg - f_hg) / (1 + np.power(self.M_r/M_use, self.beta_r))
        f_bg  = (f_hg - f_rg)

        #Get gas params
        beta, theta_out, theta_inn = self._get_gas_params(M_use, z)
        R_inn = theta_inn*R[:, None]
        R_out = theta_out*R[:, None]
        f_bg = f_bg[:, None]
        
        u = r_use/R_inn
        v = r_use/R_out

        #Now compute the large-scale behavior (which is an NFW profile)
        if self.cdelta is None:
            c_M_relation = ccl.halos.concentration.ConcentrationDiemer15(mdef = self.mass_def) #Use the diemer calibration
        else:
            c_M_relation = ccl.halos.concentration.ConcentrationConstant(self.cdelta, mdef = self.mass_def)
            
        c     = c_M_relation.get_concentration(cosmo, M_use, a)
        r_s   = (R/c)[:, None]
        x     = r_use / r_s
        y1    = np.power(1 + R_out/R_inn, -beta)/4 * (R_out/r_s) * np.power(1 + R_out/r_s, 2)
        
        #Integrate over wider region in radii to get normalization of gas profile
        #Using a number narrower range than Schneider cause we only need to go to R200c
        r_integral = np.geomspace(1e-6, 10, 500)
        u_integral = r_integral/R_inn
        v_integral = r_integral/R_out

        prof_integral = 1/(1 + u_integral)**beta / (1 + v_integral**2)**2
        prof_integral = np.where(r_integral[None, :] < R[:, None], prof_integral, 0)
        Normalization = np.trapz(4 * np.pi * r_integral**2 * prof_integral, r_integral, axis = -1)[:, None]

        del u_integral, v_integral, prof_integral

        prof  = 1/(1 + u)**beta / (1 + v**2)**2
        nfw   = y1 / x / np.power(1 + x, 2)
        prof  = np.where(v <= 1, prof, nfw) / Normalization
        prof *= f_bg*M_use[:, None] #This profile is allowed to go beyond R200c!
        
        arg   = (r_use[None, :] - self.cutoff)
        arg   = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac  = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        prof  = prof * kfac
        
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)


        return prof
    
class Pressure(AricoProfiles):

    """
    Class representing the gas density profile.

    This class is derived from the `SchneiderProfiles` class and provides an implementation 
    of a gas density profile. It calculates the real-space gas density profile, using
    the general NFW (GNFW) model of `Nagai, Kravtsov & Vikhlinin 2009 <https://arxiv.org/pdf/astro-ph/0703661>`_.

    See `SchneiderProfiles` for more docstring details.

    Notes
    -----
    The `Gas` class models the gas distribution in halos by considering the gas fraction, 
    which is computed based on the total baryonic fraction minus the stellar fraction. 
    The gas density profile is defined using parameters such as `beta`, `delta`, `gamma`, 
    `theta_inn`, and `theta_out`. These parameters characterize the core and ejection properties 
    of the gas distribution.

    The gas density profile is given by:

    .. math::

        \\rho_{\\text{gas}}(r) = \\frac{f_{\\text{gas}} M_{\\text{tot}}}{N} \\cdot 
        \\frac{1}{(1 + u)^{\\beta}} \\cdot \\frac{1}{(1 + v)^{(\\delta - \\beta)/\\gamma}}

    where:

    - :math:`f_{\\text{gas}} = f_{\\text{bar}} - f_{\\star}` is the gas fraction.
    - :math:`f_{\\text{bar}}` is the cosmic baryon fraction.
    - :math:`f_{\\star}` is the stellar mass fraction, defined as:

      .. math::

          f_{\\star} = 2A \\left(\\left(\\frac{M}{M_1}\\right)^{\\tau} + \\left(\\frac{M}{M_1}\\right)^{\\eta}\\right)^{-1}

    - :math:`M_{\\text{tot}}` is the total halo mass.
    - :math:`N` is the normalization factor to ensure mass conservation.
    - :math:`u = \\frac{r}{R_{\\text{co}}}` and :math:`v = \\frac{r}{R_{\\text{ej}}}` are dimensionless radii.
    - :math:`\\beta` is the power-law slope for :math:`R_{\\text{co}} \lesssim r \lesssim R_{\\text{ej}}`
    - :math:`\\delta` is the power-law slope at :math:`r \sim \lesssim R_{\\text{ej}}`
    - :math:`\\gamma` is the power-law slope for :math:`r \gg R_{\\text{ej}}`
    - :math:`R_{\\text{co}} = \\theta_{\\text{co}} R` is the core radius.
    - :math:`R_{\\text{ej}} = \\theta_{\\text{ej}} R` is the ejection radius.
    - :math:`r` is the radial distance.

    Examples
    --------
    Create a `Gas` profile and compute the density at specific radii:

    >>> gas_profile = Gas(**parameters)
    >>> cosmo = ...  # Define or load a cosmology object
    >>> r = np.logspace(-2, 1, 50)  # Radii in comoving Mpc
    >>> M = 1e14  # Halo mass in solar masses
    >>> a = 0.5  # Scale factor corresponding to redshift z
    >>> density_profile = gas_profile.real(cosmo, r, M, a)
    """

    def __init__(self, gas = None, **kwargs):
        
        self.Gas = gas
        if self.Gas is None: self.Gas = BoundGas(**kwargs)        

        AricoProfiles.__init__(self, **kwargs)


    def _real(self, cosmo, r, M, a):


        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        if self.cdelta is None:
            c_M_relation = ccl.halos.concentration.ConcentrationDiemer15(mdef = self.mass_def) #Use the diemer calibration
        else:
            c_M_relation = ccl.halos.concentration.ConcentrationConstant(self.cdelta, mdef = self.mass_def)

        #Get concentration values, and the effective equation of state, Gamma    
        c    = c_M_relation.get_concentration(cosmo, M_use, a)[:, None]
        r_s  = R[:, None]/c
        Norm = 4*np.pi*r_s**3 * (np.log(1 + c) - c/(1 + c))
        rhoc = M_use[:, None]/Norm
        xp   = c * self.theta_out
        Geff = 1 + ((1 + xp)*np.log(1 + xp) - xp) / ((1 + 3*xp) * np.log(1 + xp))
        
        #Normalization from Equation 5 in https://arxiv.org/pdf/2406.01672v1
        rho0  = self.Gas.real(cosmo, np.atleast_1d([0]), M_use, a) #To get normalization of gas profile
        P0    = 4*np.pi*G * (rhoc * r_s**2)/np.power(rho0, Geff - 1) * (1 - 1/Geff) 
        P0    = P0 * (Msun_to_Kg * 1e3) / (Mpc_to_m * 1e2) #Convert to CGS. Using only one factor of Mpc_to_m is correct!

        #Now compute the final profile
        rhoBG = self.Gas.real(cosmo, r_use, M_use, a)
        prof  = P0 * np.power(rhoBG, Geff)
        
        arg   = (r_use[None, :] - self.cutoff)
        arg   = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac  = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        prof  = prof * kfac

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)

        return prof
    

class NonThermalFrac(AricoProfiles):
    
    """
    Class for computing the non-thermal pressure fraction profile using the Green et al. (2020) model.

    
    Notes
    -----
    The model is based on parameters calibrated to simulations and is specifically defined 
    with respect to \( R_{200m} \), the radius within which the mean density is 200 times 
    the mean matter density of the universe.

    The non-thermal pressure fraction \( f_{\\text{nt}}(r) \) is calculated using:

    .. math::

        f_{\\text{nt}}(r) = 1 - a \\left(1 + \\exp\\left(-\\left(\\frac{x}{b}\\right)^c\\right)\\right) 
                            \\left(\\frac{\\nu_M}{4.1}\\right)^{\\frac{d}{1 + \\left(\\frac{x}{e}\\right)^f}}

    where:
        - \( x = \\frac{r}{R_{200m}} \)
        - \( \\nu_M = \\frac{1.686}{\\sigma(M_{200m})} \) is the peak height parameter.
        - \( a, b, c, d, e, f \) are model parameters calibrated to fit simulation data.

    There are no free parameters in this model; it is completely specified by the halo mass and redshift.
    """

    def _real(self, cosmo, r, M, a):
        
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc
        
        #They define the model with R200m, so gotta use that redefinition here.
        mdef  = ccl.halos.massdef.MassDef(200, 'matter')
        cnvrt = ccl.halos.mass_translator(mass_in = self.mass_def, mass_out = mdef, concentration = self.mass_def.concentration)
        M200m = cnvrt(cosmo, M_use, a)
        R200m = mdef.get_radius(cosmo, M_use, a)/a #in comoving distance

        x = r_use/R200m[:, None]

        a, b, c, d, e, f = 0.495, 0.719, 1.417,-0.166, 0.265, -2.116 #Values from Green20
        a    = self.A_nt * np.power(1 + z, self.alpha_nt)
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
    
class Temperature(AricoProfiles):
    """
    Class for computing the temperature profile in halos.

    The temperature is derived from the thermal pressure and the number density profiles, 
    of a species using the ideal gas law. The temperature profile is important for understanding 
    the thermal state of the intracluster medium and its impact on various astrophysical processes.

    For this model to be correct, the input pressure must be the *thermal pressure*, i.e. the
    non-thermal pressure must have already been accounted for in the model passed to this class.


    Parameters
    ----------
    pressure : Pressure, optional
        An instance of the `Pressure` class defining the thermal gas pressure profile. 
        If non-thermal pressure is relevant for your problem, it must be included in this
        profile; see `Pressure` or `NonThermalFrac` for more details.
        If this parameter is not provided, a default `Pressure` object is created using `kwargs`.
    gasnumberdensity : GasNumberDensity, optional
        An instance of the `GasNumberDensity` class defining the gas number density profile. 
        If not provided, a default `GasNumberDensity` object is created using `kwargs`.
    **kwargs
        Additional keyword arguments passed to initialize the `Pressure`, `GasNumberDensity`, 
        and other parameters from `SchneiderProfiles`.

    Notes
    -----
    The `Temperature` class computes the temperature profile of the gas in halos by dividing 
    the gas pressure by the gas number density and the Boltzmann constant. This calculation 
    assumes the ideal gas law, which relates pressure, number density, and temperature.

    The gas temperature \( T \) is calculated using:

    .. math::

        T(r) = \\frac{P}(r)}{n(r) \\cdot k_B}

    where:
        - \( P(r) \) is the Thermal pressure profile of a species.
        - \( n(r) \) is the number density profile of a species.
        - \( k_B \) is the Boltzmann constant (in eV).
    """
    
    def __init__(self, pressure = None, gas = None, **kwargs):
        
        self.Pressure = pressure
        self.Gas      = gas
        
        if self.Pressure is None: self.Pressure = Pressure(**kwargs) * (1 - NonThermalFrac(**kwargs))
        if self.Gas is None:      self.Gas      = BoundGas(**kwargs)
            
        super().__init__(**kwargs)
        
    
    def _real(self, cosmo, r, M, a):
        
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        P   = self.Pressure.real(cosmo, r_use, M, a)
        n   = self.Gas.real(cosmo, r_use, M, a) / (self.mean_molecular_weight * m_p) / (Mpc_to_m * m_to_cm)**3
        
        #We'll have instances of n == 0, which isn't a problem so let's ignore
        #warnings of divide errors, because we know they happen here.
        #Instead we will fix them by replacing the temperature with 0s,
        #since there is no gas in those regions to use anyway.
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            prof = P/(n * kb_cgs)
            prof = np.where(n == 0, 0, prof)
        
        return prof
    

    def projected(self, cosmo, r, M, a):
        """
        Need a custom projected class, because we want the "average temperature"
        along the line of sight. The "integrated temperature" is not a meaningful
        physical quantity.
        """

        P   = self.Pressure.projected(cosmo, r, M, a)
        n   = self.Gas.projected(cosmo, r, M, a) / (self.mean_molecular_weight * m_p) / (Mpc_to_m * m_to_cm)**3

        #We'll have instances of n == 0, which isn't a problem so let's ignore
        #warnings of divide errors, because we know they happen here.
        #Instead we will fix them by replacing the temperature with 0s,
        #since there is no gas in those regions to use anyway.
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            prof = P/(n * kb_cgs)
            prof = np.where(n == 0, 0, prof)

        return prof
    
class BoundGasDeprecated(AricoProfiles):

    """
    Class representing the gas density profile.

    This class is derived from the `SchneiderProfiles` class and provides an implementation 
    of a gas density profile. It calculates the real-space gas density profile, using
    the general NFW (GNFW) model of `Nagai, Kravtsov & Vikhlinin 2009 <https://arxiv.org/pdf/astro-ph/0703661>`_.

    See `SchneiderProfiles` for more docstring details.

    Notes
    -----
    The `Gas` class models the gas distribution in halos by considering the gas fraction, 
    which is computed based on the total baryonic fraction minus the stellar fraction. 
    The gas density profile is defined using parameters such as `beta`, `delta`, `gamma`, 
    `theta_inn`, and `theta_out`. These parameters characterize the core and ejection properties 
    of the gas distribution.

    The gas density profile is given by:

    .. math::

        \\rho_{\\text{gas}}(r) = \\frac{f_{\\text{gas}} M_{\\text{tot}}}{N} \\cdot 
        \\frac{1}{(1 + u)^{\\beta}} \\cdot \\frac{1}{(1 + v)^{(\\delta - \\beta)/\\gamma}}

    where:

    - :math:`f_{\\text{gas}} = f_{\\text{bar}} - f_{\\star}` is the gas fraction.
    - :math:`f_{\\text{bar}}` is the cosmic baryon fraction.
    - :math:`f_{\\star}` is the stellar mass fraction, defined as:

      .. math::

          f_{\\star} = 2A \\left(\\left(\\frac{M}{M_1}\\right)^{\\tau} + \\left(\\frac{M}{M_1}\\right)^{\\eta}\\right)^{-1}

    - :math:`M_{\\text{tot}}` is the total halo mass.
    - :math:`N` is the normalization factor to ensure mass conservation.
    - :math:`u = \\frac{r}{R_{\\text{co}}}` and :math:`v = \\frac{r}{R_{\\text{ej}}}` are dimensionless radii.
    - :math:`\\beta` is the power-law slope for :math:`R_{\\text{co}} \lesssim r \lesssim R_{\\text{ej}}`
    - :math:`\\delta` is the power-law slope at :math:`r \sim \lesssim R_{\\text{ej}}`
    - :math:`\\gamma` is the power-law slope for :math:`r \gg R_{\\text{ej}}`
    - :math:`R_{\\text{co}} = \\theta_{\\text{co}} R` is the core radius.
    - :math:`R_{\\text{ej}} = \\theta_{\\text{ej}} R` is the ejection radius.
    - :math:`r` is the radial distance.

    Examples
    --------
    Create a `Gas` profile and compute the density at specific radii:

    >>> gas_profile = Gas(**parameters)
    >>> cosmo = ...  # Define or load a cosmology object
    >>> r = np.logspace(-2, 1, 50)  # Radii in comoving Mpc
    >>> M = 1e14  # Halo mass in solar masses
    >>> a = 0.5  # Scale factor corresponding to redshift z
    >>> density_profile = gas_profile.real(cosmo, r, M, a)
    """

    def _real(self, cosmo, r, M, a):


        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1
        R = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        f_cg  = self._get_star_frac(M_use, z)
        f_bar = cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m
        f_bg  = (f_bar - f_cg) / (1 + np.power(self.M_c/M_use, self.beta))
        f_bg  = f_bg[:, None]
        
        if self.cdelta is None:
            c_M_relation = ccl.halos.concentration.ConcentrationDiemer15(mdef = self.mass_def) #Use the diemer calibration
        else:
            c_M_relation = ccl.halos.concentration.ConcentrationConstant(self.cdelta, mdef = self.mass_def)
            
        c    = c_M_relation.get_concentration(cosmo, M_use, a)
        r_s  = (R/c)[:, None]
        eps  = self.epsilon_hydro
        e5   = c[:, None] / eps
        Geff = (1 + 3*c/eps) * np.log(1 + c/eps) / ((1 + c/eps)*np.log(1 + c/eps) - c/eps)
        y1   = np.power(np.log(1 + e5)/e5, Geff) * (e5*(1 + e5)**2) #Set y1 based on continuity
        
        #Integrate over wider region in radii to get normalization of gas profile
        #Only go till 10Mpc since profile is cut at R200c
        r_integral = np.geomspace(1e-6, 10, 500)
        x_integral = r_integral / r_s

        u_integral = np.power(np.log(1 + x_integral)/x_integral, Geff)
        v_integral = y1 * np.power(1 + x_integral, -2)/x_integral
        y_integral = np.where(r_integral < R/eps, u_integral, v_integral)
        y_integral = np.where(r_integral > R, 0, y_integral)
        Norm       = np.trapz(4 * np.pi * r_integral**2 * y_integral, r_integral, axis = -1)[:, None]

        del r_integral, x_integral, u_integral, v_integral, y_integral

        #Now define the actual profile
        x = r_use / r_s
        u = np.power(np.log(1 + x)/x, Geff)
        v = y1 * np.power(1 + x, -2)/x
        
        prof  = np.where(r_use < R/eps, u, v)
        prof  = np.where(r_use > R, 0, prof)
        prof  = f_bg * M_use * prof / Norm

        arg   = (r_use[None, :] - self.cutoff)
        arg   = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac  = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        prof *= kfac

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)


        return prof