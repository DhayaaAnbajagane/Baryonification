
import numpy as np
import pyccl as ccl

from scipy import interpolate
from astropy.cosmology import z_at_value, FlatLambdaCDM, FlatwCDM
from astropy import units as u

from .Schneider19Profiles import DarkMatterOnly, DarkMatterBaryon

class Baryonification3D(object):

    def __init__(self, DMO, DMB, R_range = [1e-5, 50], N_samples = 500, epsilon_max = 4):


        self.DMO = DMO
        self.DMB = DMB
        self.R_range     = R_range
        self.epsilon_max = epsilon_max
        self.N_samples = N_samples

    def displacement_func_shell(self, cosmo, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):
        '''
        This routine generates a function that is used to do
        baryonification in a 2D density image
        '''

        r    = np.geomspace(self.R_range[0], self.R_range[1], self.N_samples)
        dlnr = np.log(r[1]) - np.log(r[0])

        z = 1/a - 1

        M_use = np.atleast_1d(M)
        R = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc


        rho_DMO = self.DMO.real(cosmo, r, M, a, mass_def = mass_def)
        rho_DMB = self.DMB.real(cosmo, r, M, a, mass_def = mass_def)

        #Extra factor of "a" accounts for projection in ccl being done in comoving, not physical units
        M_DMO   = np.cumsum(4*np.pi*r**3 * rho_DMO * dlnr) * a
        M_DMB   = np.cumsum(4*np.pi*r**3 * rho_DMB * dlnr) * a

        del rho_DMO, rho_DMB

        interp_DMO = interpolate.interp1d(np.log(r), np.log(M_DMO))
        interp_DMB = interpolate.interp1d(np.log(M_DMB), np.log(r), bounds_error = False) #Reverse needed for practical application

        #Func takes radius in DMO sim. Computes enclosed DMO mass.
        #Finds radius in DMB that has same enclosed mass.
        #Gets distance offset between the two.
        def func(x):

            offset = np.zeros_like(x)
            inside = (x > r[0]) & (x < self.epsilon_max*R)

            x = x[inside]

            one = interp_DMO(np.log(x))
            two = interp_DMB(one)
            two = np.where(np.isnan(two), np.log(x), two) #When two is np.NaN, offset is just 0

            offset[inside] = np.exp(two) - x

            return offset

        return func


class Baryonification2D(Baryonification3D):

    def displacement_func_shell(self, cosmo, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):
        '''
        This routine generates a function that is used to do
        baryonification in a 2D density image
        '''

        r    = np.geomspace(self.R_range[0], self.R_range[1], self.N_samples)
        dlnr = np.log(r[1]) - np.log(r[0])

        z = 1/a - 1

        M_use = np.atleast_1d(M)
        R = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        Sigma_DMO = self.DMO.projected(cosmo, r, M, a, mass_def = mass_def)
        Sigma_DMB = self.DMO.projected(cosmo, r, M, a, mass_def = mass_def)

        #Extra factor of "a" accounts for projection in ccl being done in comoving, not physical units
        M_DMO   = np.cumsum(2*np.pi*r**2 * Sigma_DMO * dlnr) * a
        M_DMB   = np.cumsum(2*np.pi*r**2 * Sigma_DMB * dlnr) * a

        del rho_DMO, rho_DMB

        interp_DMO = interpolate.interp1d(np.log(r), np.log(M_DMO))
        interp_DMB = interpolate.interp1d(np.log(M_DMB), np.log(r), bounds_error = False) #Reverse needed for practical application

        #Func takes radius in DMO sim. Computes enclosed DMO mass.
        #Finds radius in DMB that has same enclosed mass.
        #Gets distance offset between the two.
        def func(x):

            offset = np.zeros_like(x)
            inside = (x > r[0]) & (x < self.epsilon_max*R)

            x = x[inside]

            one = interp_DMO(np.log(x))
            two = interp_DMB(one)
            two = np.where(np.isnan(two), np.log(x), two) #When two is np.NaN, offset is just 0

            offset[inside] = np.exp(two) - x

            return offset

        return func
