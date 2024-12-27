import numpy as np
import pyccl as ccl
from operator import add, mul, sub, truediv, pow, neg, pos, abs
import warnings

from scipy import interpolate, special
from ..utils.Tabulate import _set_parameter
from . import Schneider19 as S19, Arico20 as A20
from .Thermodynamic import (G, Msun_to_Kg, Mpc_to_m, kb_cgs, m_p, m_to_cm)

__all__ = ['model_params', 'MeadProfiles', 
           'DarkMatter', 'TwoHalo', 'Stars', 'Gas', 'BoundGas', 'EjectedGas', 'ReaccretedGas', 'CollisionlessMatter',
           'DarkMatterOnly', 'DarkMatterBaryon']

model_params = ['cdelta', 'eps1', 'nu_eps1', 'eps2', #DM profle param and relaxation params
                'cutoff', 'proj_cutoff', #Cutoff parameters (numerical)
                'p', 'q',  #Two halo terms
                
                'M_0', 'beta', 'Gamma', 'nu_Gamma', 'eta_b', #Default gas profile param

                'A_star', 'nu_A_star', 'M_star', 'nu_M_star', 'sigma_star', 'epsilon_h', 'eta', #Star params

                'T_w', 'nu_T_w', #Temperature params
                'mean_molecular_weight', #Gas number density params
               ]


class MeadProfiles(A20.AricoProfiles):
    __doc__ = A20.AricoProfiles.__doc__.replace('AricoProfiles', 'MeadProfiles')

    #Define the new param names
    model_param_names = model_params


    def _get_fstar(self, M_use, a):
        
        z     = 1/a - 1
        Astr  = self.A_star + self.nu_A_star * z
        Mstr  = self.M_star * np.exp(z * self.nu_M_star)
        f_str = Astr * np.exp(-np.power(np.log10(M_use/Mstr)/self.sigma_star, 2)/2)
        f_str = np.where(M_use > Mstr, np.max([f_str, Astr/3 * np.ones_like(f_str)]), f_str)
        f_cen = f_str * np.where(M_use < Mstr, 1, np.power(M_use/Mstr, self.eta))
        f_sat = f_str * np.where(M_use < Mstr, 0, 1 - np.power(M_use/Mstr, self.eta))
        
        return f_str, f_cen, f_sat    


class DarkMatter(MeadProfiles):
    """
    Class representing the total Dark Matter (DM) profile using the NFW (Navarro-Frenk-White) profile.

    This class is derived from the `SchneiderProfiles` class and provides an implementation of the 
    dark matter profile based on the NFW model. It includes a custom `_real` method for calculating 
    the real-space dark matter density profile, considering factors like the concentration-mass 
    relation and truncation radius.

    See `MeadProfiles` for more docstring details.

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
            #Use the Duffy08 calibration following Equation 33 in https://arxiv.org/pdf/2005.00009
            #Can in principle swap this around as you want
            c_M_relation = ccl.halos.concentration.ConcentrationDuffy08(mdef = self.mass_def)
        else:
            c_M_relation = ccl.halos.concentration.ConcentrationConstant(self.cdelta, mdef = self.mass_def)
            
        #No modification of DMO concentration here
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
    

class TwoHalo(S19.TwoHalo, MeadProfiles):
    __doc__ = S19.TwoHalo.__doc__.replace('SchneiderProfiles', 'MeadProfiles')


class Stars(MeadProfiles):
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
        
        super().__init__(**kwargs)
        
        #For some reason, we need to make this extreme in order
        #to prevent ringing in the profiles. Haven't figured out
        #why this is the case
        self.update_precision_fftlog(padding_lo_fftlog = 1e-5, padding_hi_fftlog = 1e5)

    
    def _real(self, cosmo, r, M, a):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        R     = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc
        f_str, f_cen, f_sat = self._get_fstar(M_use, a)
        f_cen = f_cen[:, None]
        R_h   = self.epsilon_h * R[:, None]

        #Final profile. No truncation needed since exponential cutoff already does that for us
        prof = f_cen*M_use[:, None] / (4*np.pi**(3/2)*R_h) * 1/r_use**2 * np.exp(-(r_use/2/R_h)**2)

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)

        return prof
    

class BoundGas(MeadProfiles):

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

        if self.cdelta is None:
            #Use the Duffy08 calibration following Equation 33 in https://arxiv.org/pdf/2005.00009
            #Can in principle swap this around as you want
            c_M_relation = ccl.halos.concentration.ConcentrationDuffy08(mdef = self.mass_def)
        else:
            c_M_relation = ccl.halos.concentration.ConcentrationConstant(self.cdelta, mdef = self.mass_def)

        z     = 1/a - 1
        c     = c_M_relation.get_concentration(cosmo, M_use, a)
        R     = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc
        r_s   = R/c
        r_s   = r_s[:, None]
        Geff  = self.Gamma + self.nu_Gamma * z

        f_str, f_cen, f_sat = self._get_fstar(M_use, a)
        f_bar = cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m
        f_bnd = (f_bar - f_str) * np.power(self.M_0/M_use, self.beta) / (1 + np.power(self.M_0/M_use, self.beta))
        f_bnd = f_bnd[:, None]
        
        #Do normalization halo-by-halo, since we want custom radial ranges.
        #This way, we can handle sharp transition at R200c without needing
        #super fine resolution in the grid.
        Normalization = np.ones_like(M_use)
        for m_i in range(M_use.shape[0]):
            r_integral    = np.geomspace(1e-6, R[m_i], 500)
            x_integral    = r_integral/r_s[m_i]
            prof_integral = np.power(np.log(1 + x_integral) / x_integral, 1/(Geff - 1))
            Normalization[m_i] = np.trapz(4 * np.pi * r_integral**2 * prof_integral, r_integral)
        Normalization = Normalization[:, None]

        del prof_integral, x_integral

        arg   = (r_use[None, :] - self.cutoff)
        arg   = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac  = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        x_use = r_use / r_s
        prof  = np.power(np.log(1 + x_use) / x_use, 1/(Geff - 1))
        prof  = np.where(r_use[None, :] <= R[:, None], prof, 0)
        prof *= f_bnd*M_use[:, None]/Normalization

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)


        return prof
    

class EjectedGas(MeadProfiles):

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

    _safe_Pchip_minimize = A20.ModifiedDarkMatter._safe_Pchip_minimize

    def _real(self, cosmo, r, M, a):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1
        R = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        f_str, f_cen, f_sat = self._get_fstar(M_use, a)
        f_bar = cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m
        f_bnd = (f_bar - f_str) * np.power(self.M_0/M_use, self.beta) / (1 + np.power(self.M_0/M_use, self.beta))
        f_ej  = ((f_bar - f_str) - f_bnd)[:, None]

        #Now use the escape radius, which is r_esc = v_esc * t_hubble
        #and this reduces down to just 1/2 * sqrt(Delta) * R_Delta
        R_esc = 1/2 * np.sqrt(200) * R[:, None]
        rgrid = np.geomspace(1e-2, 100, 100)
        Term1 = 1 - special.erf(self.eta_b * R_esc / np.sqrt(2) / rgrid)
        Term2 = np.sqrt(2/np.pi) * self.eta_b * R_esc / rgrid * np.exp(-np.power(self.eta_b*R_esc/rgrid, 2)/2)
        Diff  = Term1 + Term2 - 1/f_bar * f_ej

        R_ej  = np.exp([self._safe_Pchip_minimize(Diff[m_i], np.log(rgrid)) for m_i in range(Diff.shape[0])])[:, None]

        arg   = (r_use[None, :] - self.cutoff)
        arg   = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac  = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        prof  = f_ej * M_use[:, None] / np.power(2*np.pi*R_ej**2, 3/2) * np.exp(-np.power(r_use/R_ej, 2)/2) * kfac

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)

        return prof


class Gas(MeadProfiles):
    '''
    Convenience class that combines the Bound, Ejected, and Reaccreted gas components
    '''
    def __init__(self, **kwargs): self.myprof = BoundGas(**kwargs) + EjectedGas(**kwargs)
    def __getattr__(self, name):  return getattr(self.myprof, name)
    
    #Need to explicitly set these two methods (to enable pickling)
    #since otherwise the getattr call above leads to infinite recursions.
    def __getstate__(self): self.__dict__.copy()    
    def __setstate__(self, state): self.__dict__.update(state)


class CollisionlessMatter(MeadProfiles):
    """
    Class representing the total Dark Matter (DM) profile using the NFW (Navarro-Frenk-White) profile.

    This class is derived from the `SchneiderProfiles` class and provides an implementation of the 
    dark matter profile based on the NFW model. It includes a custom `_real` method for calculating 
    the real-space dark matter density profile, considering factors like the concentration-mass 
    relation and truncation radius.

    See `MeadProfiles` for more docstring details.

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

    def _modify_concentration(self, cosmo, c, M, a):

        z      = 1/a - 1
        f_bar  = cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m
        f_bnd  = f_bar * np.power(self.M_0/M, self.beta) / (1 + np.power(self.M_0/M, self.beta))
        eps1   = self.eps1 + z * self.nu_eps1
        factor = (1 + eps1 + (self.eps2 - eps1) * f_bnd / f_bar)

        return c * factor
    
    def _real(self, cosmo, r, M, a):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        if self.cdelta is None:
            #Use the Duffy08 calibration following Equation 33 in https://arxiv.org/pdf/2005.00009
            #Can in principle swap this around as you want
            c_M_relation = ccl.halos.concentration.ConcentrationDuffy08(mdef = self.mass_def)
        else:
            c_M_relation = ccl.halos.concentration.ConcentrationConstant(self.cdelta, mdef = self.mass_def)
            
        c   = c_M_relation.get_concentration(cosmo, M_use, a)
        c   = self._modify_concentration(cosmo, c, M_use, a)
        R   = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc
        r_s = R/c

        #Get the normalization (rho_c) analytically since we don't have a truncation radii like S19 does
        Norm  = 4*np.pi*r_s**3 * (np.log(1 + c) - c/(1 + c))
        rho_c = M_use/Norm
        f_str, f_cen, f_sat = self._get_fstar(M_use, a)
        f_bar = cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m
        rho_c = rho_c * (1 - f_bar + f_sat) #Rescale to correct fraction of mass

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


class DarkMatterOnly(S19.DarkMatterOnly, MeadProfiles):

    __doc__ = S19.DarkMatterOnly.__doc__.replace('SchneiderProfiles', 'MeadProfiles')

    def __init__(self, darkmatter = None, **kwargs):
        
        self.DarkMatter = darkmatter
        self.TwoHalo    = TwoHalo(**kwargs) * 0 #Should not add 2-halo in Arico method

        if self.DarkMatter is None: self.DarkMatter = DarkMatter(**kwargs)
        MeadProfiles.__init__(self, **kwargs)

class DarkMatterBaryon(S19.DarkMatterBaryon, MeadProfiles):

    __doc__ = S19.DarkMatterBaryon.__doc__.replace('SchneiderProfiles', 'MeadProfiles')

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

        MeadProfiles.__init__(self, **kwargs)


class DarkMatterOnlywithLSS(S19.DarkMatterOnly, MeadProfiles):

    __doc__ = S19.DarkMatterOnly.__doc__.replace('SchneiderProfiles', 'MeadProfiles')

    def __init__(self, darkmatter = None, twohalo = None, **kwargs):
        
        self.DarkMatter = darkmatter
        self.TwoHalo    = twohalo

        if self.DarkMatter is None: self.DarkMatter = DarkMatter(**kwargs)
        if self.TwoHalo is None: self.TwoHalo = TwoHalo(**kwargs)

        MeadProfiles.__init__(self, **kwargs)


class DarkMatterBaryonwithLSS(S19.DarkMatterBaryon, MeadProfiles):

    __doc__ = S19.DarkMatterBaryon.__doc__.replace('SchneiderProfiles', 'MeadProfiles')

    def __init__(self, gas = None, stars = None, collisionlessmatter = None, darkmatter = None, twohalo = None, **kwargs):
        
        self.Gas   = gas
        self.Stars = stars
        self.TwoHalo    = twohalo
        self.DarkMatter = darkmatter
        self.CollisionlessMatter = collisionlessmatter
        
        if self.Gas is None:        self.Gas        = Gas(**kwargs)        
        if self.Stars is None:      self.Stars      = Stars(**kwargs)
        if self.TwoHalo is None:    self.TwoHalo    = TwoHalo(**kwargs)
        if self.DarkMatter is None: self.DarkMatter = DarkMatter(**kwargs)
        if self.CollisionlessMatter is None: self.CollisionlessMatter = CollisionlessMatter(**kwargs)

        MeadProfiles.__init__(self, **kwargs)


class Temperature(MeadProfiles):
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
    
    def _real(self, cosmo, r, M, a):
        
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1
        R = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc
        
        if self.cdelta is None:
            #Use the Duffy08 calibration following Equation 33 in https://arxiv.org/pdf/2005.00009
            #Can in principle swap this around as you want
            c_M_relation = ccl.halos.concentration.ConcentrationDuffy08(mdef = self.mass_def)
        else:
            c_M_relation = ccl.halos.concentration.ConcentrationConstant(self.cdelta, mdef = self.mass_def)
            
        c    = c_M_relation.get_concentration(cosmo, M_use, a)
        R    = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc
        r_s  = (R/c)[:, None]
    
        #Characteristic energy scale, convert to cgs and then convert to temperature (Kelvin)
        E0   = G * M_use * m_p * self.mean_molecular_weight / (a * R) * (Msun_to_Kg * 1e3) * (Mpc_to_m * 1e2)**2 
        T0   = E0 / (3/2 * kb_cgs)
        prof = T0[:, None] * np.log(1 + r_use/r_s) / (r_use/r_s)

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)

        return prof
    
    
    def projected(self, cosmo, r, M, a):
        '''
        Rewrite projected function to include a normalization 
        because we want "averaged" (not summed) temperature when projecting.
        '''

        r_max = self.padding_hi_proj * np.max(r)
        if self.proj_cutoff is not None: r_max = self.proj_cutoff

        return super().projected(cosmo, r, M, a) / (2 * r_max)
    


class Pressure(MeadProfiles):

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

    def __init__(self, boundgas = None, ejectedgas = None, temperature = None, **kwargs):
        
        self.BoundGas    = boundgas
        self.EjectedGas  = ejectedgas
        self.Temperature = temperature
        if self.BoundGas is None:    self.BoundGas    = BoundGas(**kwargs)
        if self.EjectedGas is None:  self.EjectedGas  = EjectedGas(**kwargs)
        if self.Temperature is None: self.Temperature = Temperature(**kwargs)        

        super().__init__(**kwargs)


    def _real(self, cosmo, r, M, a):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        #The first "bound" component
        T    = self.Temperature.real(cosmo, r_use, M, a)
        n    = self.BoundGas.real(cosmo, r_use, M, a) / (self.mean_molecular_weight * m_p) / (Mpc_to_m * m_to_cm)**3
        P1   = T * n * kb_cgs

        #The second, "ejected" component
        T    = self.T_w * np.exp(self.nu_T_w * z)
        n    = self.EjectedGas.real(cosmo, r_use, M, a) / (self.mean_molecular_weight * m_p) / (Mpc_to_m * m_to_cm)**3
        P2   = T * n * kb_cgs

        prof = P1 + P2

        return prof
    

#Default params, provided in Table 2 of https://arxiv.org/pdf/2005.00009
Params_TAGN_7p6 = {'A_star' : 0.0346, 'nu_A_star' : -0.0092, 'M_star' : np.power(10, 12.5506), 'nu_M_star' : -0.4615,
                   'eta' : -0.4970, 'eps1' : 0.4021, 'nu_eps1' : 0.0435, 'Gamma' : 1.2763, 'nu_Gamma' : -0.0554, 
                   'M_0' : np.power(10, 13.0978), 'T_w' : np.power(10, 6.6762), 'nu_T_w' : -0.5566,
                   'eps2' : 0, 'mean_molecular_weight' : 0.59, 'eta_b' : 0.5, 'sigma_star' : 1.2, 'beta' : 0.6,
                   'epsilon_h' : 0.015, 'p' : 0.3, 'q' : 0.707}

Params_TAGN_7p8 = {'A_star' : 0.0342, 'nu_A_star' : -0.0105, 'M_star' : np.power(10, 12.3715), 'nu_M_star' : 0.0149,
                   'eta' : -0.4052, 'eps1' : 0.1236, 'nu_eps1' : -0.0187, 'Gamma' : 1.2956, 'nu_Gamma' : -0.0937, 
                   'M_0' : np.power(10, 13.4854), 'T_w' : np.power(10, 6.6545), 'nu_T_w' : -0.3652,
                   'eps2' : 0, 'mean_molecular_weight' : 0.59, 'eta_b' : 0.5, 'sigma_star' : 1.2, 'beta' : 0.6,
                   'epsilon_h' : 0.015, 'p' : 0.3, 'q' : 0.707}

Params_TAGN_8p0 = {'A_star' : 0.0321, 'nu_A_star' : -0.0094, 'M_star' : np.power(10, 12.3032), 'nu_M_star' : -0.0817,
                   'eta' : -0.3443, 'eps1' : -0.1158, 'nu_eps1' : 0.1408, 'Gamma' : 1.2861, 'nu_Gamma' : -0.1382, 
                   'M_0' : np.power(10, 14.1254), 'T_w' : np.power(10, 6.6615), 'nu_T_w' : -0.0617,
                   'eps2' : 0, 'mean_molecular_weight' : 0.59, 'eta_b' : 0.5, 'sigma_star' : 1.2, 'beta' : 0.6,
                   'epsilon_h' : 0.015, 'p' : 0.3, 'q' : 0.707}