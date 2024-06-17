import numpy as np
import pyccl as ccl

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
        
        #Convert to CGS
        prof = prof * (Msun_to_Kg * 1e3) / (Mpc_to_m * 1e2)

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
    
    
class BattagliaElectronPressure(BattagliaPressure):
    
    def _real(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):
        
        prof = Pth_to_Pe * super()._real(cosmo, r, M, a, mass_def)
        
        return prof
    
    
class BattagliaGasDensity(ccl.halos.profiles.HaloProfile):

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

        self.mdef = ccl.halos.massdef.MassDef(200, 'critical')

        self.Model_def = Model_def
        self.truncate  = truncate

        #Import all other parameters from the base CCL Profile class
        super().__init__()

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

        #These two are fixed parameters
        x_c   = 0.5
        gamma = -0.2

        #Setup parameters as they were calibrated in Battaglia+ 2012
        if self.Model_def == '200_AGN':

            rho_0  = 4e3   * (M_use/1e14)**0.29   * (1 + z)**-0.66
            alpha  = 0.88  * (M_use/1e14)**-0.03  * (1 + z)**0.19
            beta   = 3.83  * (M_use/1e14)**0.04   * (1 + z)**-0.025

        elif self.Model_def == '200_SH':

            rho_0  = 1.9e4 * (M_use/1e14)**0.09   * (1 + z)**-0.95
            alpha  = 0.70 * (M_use/1e14)**-0.017  * (1 + z)**0.27
            beta   = 4.43 * (M_use/1e14)**0.005   * (1 + z)**0.037


        R = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc
        x = r_use[None, :]/R[:, None]

        #Cosmological parameters
        h        = cosmo.cosmo.params.h
        Omega_m  = cosmo.cosmo.params.Omega_m
        Omega_b  = cosmo.cosmo.params.Omega_b
        fb       = Omega_b/Omega_m
        RHO_CRIT = ccl.physical_constants.RHO_CRITICAL*h**2 * ccl.background.h_over_h0(cosmo, a)**2 #This is in physical coordinates

        rho_0, alpha, beta = rho_0[:, None], alpha[:, None], beta[:, None]
        prof = RHO_CRIT * fb * rho_0 * (x/x_c)**gamma * (1 + (x/x_c)**alpha)**-((beta - gamma)/alpha)

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