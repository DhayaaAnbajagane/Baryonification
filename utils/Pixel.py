import pyccl as ccl
import numpy as np, healpy as hp
from scipy import interpolate, special


#Define a shorthand to use everywhere
fftlog = ccl.pyutils._fftlog_transform


class ConvolvedProfile(object):
    """
    A class that takes in a ccl profile object and returns a new object 
    that computes profiles while including the pixel window function of 
    the map that we paint into.
    
    It assumes the window function is isotropic
    """
    
    def __init__(self, Profile, Pixel):
        
        self.Profile = Profile
        self.Pixel   = Pixel
        
        self.fft_par = Profile.precision_fftlog
        
        self.isHarmonic = Pixel.isHarmonic
        
        
    def __getattr__(self, name):
        """
        Delegate attribute and method access 
        to the Profile object passesd to the class,
        but only if attr/method is not already found in the class.
        """
        
        try:
            return super().__getattribute__(name)
        
        except AttributeError:
            return getattr(self.Profile, name)
    
    
    def real(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):
        
        r_min = np.min(r) * self.fft_par['padding_lo_fftlog']
        r_max = np.max(r) * self.fft_par['padding_hi_fftlog']
        n     = self.fft_par['n_per_decade'] * np.int32(np.log10(r_max/r_min))
        
        r_fft = np.geomspace(r_min, r_max, n)
        prof  = self.Profile.real(cosmo, r_fft, M, a, mass_def)
        
        k_out, Pk   = fftlog(r_fft, prof, 3, 0, self.fft_par['plaw_fourier'])
        r_out, prof = fftlog(k_out, Pk * self.Pixel.real(k_out), 3, 0, self.fft_par['plaw_fourier'] + 1)
        
        r    = np.clip(r, self.Pixel.size / 5, None) #Set minimum radius according to pixel, to prevent ringing on small-scale outputs
        prof = interpolate.CubicSpline(np.log(r_out), prof, extrapolate = False, axis = -1)(np.log(r))
        prof = np.where(np.isnan(prof), 0, prof) * (2*np.pi)**3
        
        return prof
    
    
    def projected(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):
        
        if self.isHarmonic:
            
            assert a < 1, f"You cannot set a = 1, z = 0 when computing harmonic sky projections"
            D_A = ccl.background.angular_diameter_distance(cosmo, a)
            
        r_min = np.min(r) * self.fft_par['padding_lo_fftlog']
        r_max = np.max(r) * self.fft_par['padding_hi_fftlog']
        n     = self.fft_par['n_per_decade'] * np.int32(np.log10(r_max/r_min))
        
        r_fft = np.geomspace(r_min, r_max, n)
        prof  = self.Profile.projected(cosmo, r_fft, M, a, mass_def)
        
        if self.isHarmonic: r_fft = r_fft * a / D_A
        
        k_out, Pk   = fftlog(r_fft, prof, 2, 0, self.fft_par['plaw_fourier'] + 1)
        r_out, prof = fftlog(k_out, Pk * self.Pixel.projected(k_out), 2, 0, self.fft_par['plaw_fourier'] + 1)
        
        if self.isHarmonic: 
            r_out = r_out / a * D_A
            r     = np.clip(r, self.Pixel.size / 5 * D_A/a, None)
        else:
            r     = np.clip(r, self.Pixel.size / 5, None)
            
        prof = interpolate.CubicSpline(np.log(r_out), prof, extrapolate = False, axis = -1)(np.log(r))
        prof = np.where(np.isnan(prof), 0, prof) * (2*np.pi)**2
        
        return prof
    
    
    
    
class GridPixelApprox(object):
    """
    A class for holding the window of a Grid's pixel.
    In reality we approximate the Grid pixel window function
    as being a circular tophat
    """
    
    def __init__(self, size):
        
        self.size = size
        self.isHarmonic = False
        
    
    def beam(self, k, R):
        
        kr = k * (2*R) #Factor of 2 because the window function needs diameter, not radius

        with np.errstate(invalid = 'ignore', divide = 'ignore'):
            beam = np.where(kr > 0, 3*special.spherical_jn(1, kr)/kr, 1)
            
        return beam
        
        
    def real(self, k):
        
        R = np.cbrt(self.size**3 / (4/3 * np.pi) )
        
        return self.beam(k, R)
        
        
    
    def projected(self, k):
        
        R = np.sqrt(self.size**2 / np.pi)
        
        return self.beam(k, R)
            
            

class HealPixel(object):
    """
    A class for holding the window of a healpix pixel
    
    We use an analytic profile -- the Gaussian beam -- instead of the
    inbuilt pixel window in healpix. This is because the latter only
    exists up to 3*NSIDE - 1 whereas for stabel FFTlogs, we want a smooth
    window function to large ell ranges. The Gaussian beam, with a FWHM that
    is 1/sqrt(2) smaller is similar to the healpix pixel window with <0.1%
    for most scales and 1% at the smallest scales.
    """
    
    def __init__(self, NSIDE):
        
        self.NSIDE      = NSIDE
        self.size       = hp.nside2resol(NSIDE)
        self.isHarmonic = True
        
    
    def real(self, k):
        
        return np.zeros_like(k)
        
    
    def projected(self, k):
        
        sig  = hp.nside2resol(self.NSIDE) / np.sqrt(8 * np.log(2)) / np.sqrt(2)
        beam = np.exp(-k*(1 + k)/2 * sig**2)
        
        return beam
        
        