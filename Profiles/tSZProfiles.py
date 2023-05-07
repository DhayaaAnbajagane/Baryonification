
import numpy as np
import pyccl as ccl

from scipy import interpolate
from astropy.cosmology import z_at_value, FlatLambdaCDM, FlatwCDM
from astropy import units as u

from ..Profiles.Schneider19Profiles import SchneiderProfiles, Gas, DarkMatterBaryon

class Pressure(SchneiderProfiles):

    def _real(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):

        Msun_to_Kg = ccl.physical_constants.SOLAR_MASS
        Mpc_to_m   = ccl.physical_constants.MPC_TO_METER
        G          = ccl.physical_constants.GNEWT / Mpc_to_m**3 * Msun_to_Kg

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        #The overdensity constrast related to the mass definition
        Delta    = mass_def.get_Delta(cosmo, a)

        #Cosmological parameters
        Omega_m  = cosmo.cosmo.params.Omega_m
        Omega_b  = cosmo.cosmo.params.Omega_b
        h        = cosmo.cosmo.params.h

        RHO_CRIT = ccl.physical_constants.RHO_CRITICAL*h**2 * ccl.background.h_over_h0(cosmo, a)**2 #This is in physical coordinates

        # The self-similar expectation for Pressure
        # Need R*a to convert comoving Mpc to physical
        P_delta  = Delta*RHO_CRIT * Omega_b/Omega_m * G * (M_use)/(2*R*a)
        P_delta  = P_delta[:, None]

        r_integral = np.geomspace(1e-10, 1e10, 5000)

        Total_prof = DarkMatterBaryon(epsilon = self.epsilon, a = self.a, n = self.n,
                                      theta_ej = self.theta_ej, theta_co = self.theta_co, M_c = self.M_c, mu = self.mu,
                                      A = self.A, M1 = self.M1, eta_star = self.eta_star, eta_cga = self.eta_cga, epsilon_h = self.epsilon_h,
                                      p = self.p, q = self.q)
        Gas_prof   = Gas(theta_ej = self.theta_ej, theta_co = self.theta_co, M_c = self.M_c, mu = self.mu, A = self.A, M1 = self.M1, eta_star = self.eta_star)

        rho_total  = Total_prof._real(cosmo, r_integral, M, a)
        rho_gas    = Gas_prof._real(cosmo, r_integral, M, a)

        dlnr    = np.log(r_integral[1]) - np.log(r_integral[0])
        M_total = 4 * np.pi * np.cumsum(r_integral**3 * rho_total * dlnr)

#         print(rho_gas)
#         print(rho_total)
#         print(M_total)

        dP_dr = - G * M_total * rho_gas / r_integral**2 #Assuming hydrostatic equilibrium to get dP(r)/dr

        print(dP_dr)
        #integrate to get actual pressure, P(r), and use normalizing coefficient from self-similar expectation
        prof  = np.cumsum((dP_dr * r_integral)[::-1] * dlnr, axis = -1)[::-1]
        prof  = interpolate.CubicSpline(np.log(r_integral), np.log(prof), axis = 1)
        prof  = np.exp(prof(np.log(r_use)))

        nth_frac = self._nonthermal_fraction(cosmo, r_use, M_use, a, model = 'Green20', mass_def = mass_def)

        prof     = prof * (1 - 0*nth_frac) #We only want thermal pressure

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)

        assert np.all(prof >= 0), "Something went wrong. Pressure is negative in some places"

        return prof

    def _real_dP(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):

        Msun_to_Kg = ccl.physical_constants.SOLAR_MASS
        Mpc_to_m   = ccl.physical_constants.MPC_TO_METER
        G          = ccl.physical_constants.GNEWT / Mpc_to_m**3 * Msun_to_Kg

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc


        #The overdensity constrast related to the mass definition
        Delta    = mass_def.get_Delta(cosmo, a)

        #Cosmological parameters
        Omega_m  = cosmo.cosmo.params.Omega_m
        Omega_b  = cosmo.cosmo.params.Omega_b
        h        = cosmo.cosmo.params.h

        RHO_CRIT = ccl.physical_constants.RHO_CRITICAL*h**2 * ccl.background.h_over_h0(cosmo, a)**2 #This is in physical coordinates

        # The self-similar expectation for Pressure
        # Need R*a to convert comoving Mpc to physical
        P_delta  = Delta*RHO_CRIT * Omega_b/Omega_m * G * (M_use)/(2*R*a)
        P_delta  = P_delta[:, None]

        r_integral = np.geomspace(1e-10, 1e10, 5000)

        Total_prof = DarkMatterBaryon(epsilon = self.epsilon, a = self.a, n = self.n,
                                      theta_ej = self.theta_ej, theta_co = self.theta_co, M_c = self.M_c, mu = self.mu,
                                      A = self.A, M1 = self.M1, eta_star = self.eta_star, eta_cga = self.eta_cga, epsilon_h = self.epsilon_h,
                                      p = self.p, q = self.q)
        Gas_prof   = Gas(theta_ej = self.theta_ej, theta_co = self.theta_co, M_c = self.M_c, mu = self.mu, A = self.A, M1 = self.M1, eta_star = self.eta_star)

        rho_total  = Total_prof._real(cosmo, r_integral, M, a)
        rho_gas    = Gas_prof._real(cosmo, r_integral, M, a)

        dlnr    = np.log(r_integral[1]) - np.log(r_integral[0])
        M_total = 4 * np.pi * np.cumsum(r_integral**3 * rho_total * dlnr)

#         print(rho_gas)
#         print(rho_total)
#         print(M_total)

        dP_dr = - G * M_total * rho_gas / r_integral**2 #Assuming hydrostatic equilibrium to get dP(r)/dr

        prof  = interpolate.CubicSpline(np.log(r_integral), np.log(np.abs(dP_dr)), axis = 1)
        prof  = np.exp(prof(np.log(r_use)))
        prof  = prof

        assert np.all(prof >= 0), "Something went wrong. Pressure is negative in some places"

        return prof

    def _nonthermal_fraction(self, cosmo, r, M, a, model = 'Green20', mass_def = ccl.halos.massdef.MassDef(200, 'critical')):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        M200m = mass_def.translate_mass(cosmo, M_use, a, ccl.halos.massdef.MassDef200m())
        R200m = ccl.halos.massdef.MassDef200m().get_radius(cosmo, M_use, a)/a

        x = r_use/R200m[:, None]

        if model == 'Green20':
            a, b, c, d, e, f = 0.495, 0.719, 1.417,-0.166, 0.265, -2.116
            nu_M = M_use/ccl.sigmaM(cosmo, M200m, a)
            nu_M = nu_M[:, None]
            nth  = a * (1 + np.exp(-(x/b)**c)) * (nu_M/4.1)**(d/(1 + (x/e)**f))
        else:
            raise ValueError("Need to use model = Green20. No other model implemented so far.")

        return nth
