
import numpy as np
import pyccl as ccl
import healpy as hp
from numba import njit

from scipy import interpolate
from astropy.cosmology import FlatwCDM
from astropy import units as u
from astropy.io import fits

from tqdm import tqdm

MY_FILL_VAL = np.NaN


@njit
def regrid_pixels_hpix(hmap, parent_pix_vals, child_pix, child_weights):
    '''
    Function that quickly assigns displaced healpix pixels back to the original
    map grid.
    
    hmap: new array that is the healpix map to assign pixels to
    parent_pix_vals: the values of the shifted pixels
    child_pix: the 4 hmap pixels that each displaced pixel contribute to
    child_weight: the weight of the contribution to each of the 4 pixels.
    
    In practice, get child_pix and child_weight from hp.interp_values().
    '''
    
    for i in range(parent_pix_vals.size):

        for j in range(4):
            
            hmap[child_pix[i, j]] += child_weights[i, j] * parent_pix_vals[i]

    
    return hmap


                    
class DefaultRunner(object):
    '''
    A class that contains relevant utils for input/output
    '''
    
    def __init__(self, HaloLightConeCatalog, LightconeShell, model = None, use_ellipticity = False,
                 mass_def = ccl.halos.massdef.MassDef(200, 'critical'), verbose = True):

        self.HaloLightConeCatalog  = HaloLightConeCatalog
        self.LightconeShell        = LightconeShell
        self.cosmo  = HaloLightConeCatalog.cosmology
        self.model  = model
        
        
        self.mass_def = mass_def
        self.verbose  = verbose
        
        self.use_ellipticity = use_ellipticity
        
        if use_ellipticity:
            raise NotImplementedError("You have set use_ellipticity = True, but this not yet implemented for HealpixRunner")
    
    
    def build_Rmat(self, A, ref):

        A   /= np.linalg.norm(A)
        ref /= np.linalg.norm(ref)
    
        ang  = np.arccos(np.dot(A, ref))
        Rmat = np.array([[np.cos(ang), -np.sin(ang)], 
                         [np.sin(ang), np.cos(ang)]])
        
        return Rmat


    def coord_array(self, *args):

        return np.vstack([a.flatten() for a in args]).T
    

class BaryonifyShell(DefaultRunner):

    def process(self):

        cosmo_fiducial = FlatwCDM(H0 = self.cosmo['h'] * 100. * u.km / u.s / u.Mpc,
                                  Om0 = self.cosmo['Omega_m'], w0 = self.cosmo['w0'])


        cosmo = ccl.Cosmology(Omega_c = self.cosmo['Omega_m'] - self.cosmo['Omega_b'],
                              Omega_b = self.cosmo['Omega_b'], h = self.cosmo['h'],
                              sigma8  = self.cosmo['sigma8'],  n_s = self.cosmo['n_s'],
                              matter_power_spectrum = 'linear')
        cosmo.compute_sigma()

        healpix_inds = np.arange(hp.nside2npix(self.LightconeShell.NSIDE), dtype = int)

        orig_map = self.LightconeShell.map
        NSIDE    = self.LightconeShell.NSIDE

        #Build interpolator between redshift and ang-diam-dist. Assume we never use z > 30
        z_t = np.linspace(0, 30, 1000)
        D_a = interpolate.interp1d(z_t, cosmo_fiducial.angular_diameter_distance(z_t).value)
        
        pix_offsets = np.zeros([orig_map.size, 3]) 
        
        for j in tqdm(range(self.HaloLightConeCatalog.cat.size), desc = 'Baryonifying matter', disable = not self.verbose):

            M_j = self.HaloLightConeCatalog.cat['M'][j]
            z_j = self.HaloLightConeCatalog.cat['z'][j]
            a_j = 1/(1 + z_j)
            R_j = self.mass_def.get_radius(cosmo, M_j, a_j) #in physical Mpc
            D_j = D_a(z_j)
            
            c_j = self.HaloNDCatalog.cat['c'][j] if self.model.use_concentration else None

            ra_j   = self.HaloLightConeCatalog.cat['ra'][j]
            dec_j  = self.HaloLightConeCatalog.cat['dec'][j]
            vec_j  = hp.ang2vec(ra_j, dec_j, lonlat = True)
            
            radius = R_j * self.epsilon_max / D_j
            pixind = hp.query_disc(self.LightconeShell.NSIDE, vec_j, radius, inclusive = False, nest = False)
            
            #If there are less than 4 particles, use the 4 nearest particles
            if pixind.size < 4:
                pixind = hp.get_interp_weights(NSIDE, ra_j, dec_j, lonlat = True)[0]
                
            vec    = np.stack(hp.pix2vec(nside = NSIDE, ipix = pixind), axis = 1)
            
            pos_j  = vec_j * D_j #We assume flat cosmologies, where D_a is the right distance to use here
            pos    = vec   * D_j
            diff   = pos - pos_j
            r_sep  = np.sqrt(np.sum(diff**2, axis = 1))
            
            #Compute the displacement needed
            offset = self.model.displacement(r_sep/a_j, M_j, a_j, c = c_j) * a_j
            offset = offset[:, None] * (diff/r_sep[:, None]) #Add direction
            offset = np.where(np.isfinite(offset), offset, 0) #If offset is weird, set it to 0
            
            #Now convert the 3D offset into a shift in the unit vector of the pixel
            nw_pos = pos + offset #New position
            nw_vec = nw_pos/np.sqrt(np.sum(nw_pos**2, axis = 1))[:, None] #Get unit vector of new position
            offset = nw_vec - vec #Subtract from it the pixel's original unit vector
            
            #Accumulate the offsets in the UNIT VECTORS of the hpixels
            pix_offsets[pixind, :] += offset
        
            
        new_map = np.zeros(orig_map.size, dtype = float)
        
        new_vec = np.stack( hp.pix2vec(NSIDE, np.arange(orig_map.size)), axis = 1) + pix_offsets
        new_ang = np.stack( hp.vec2ang(new_vec, lonlat = True), axis = 1)
        p_pix   = np.where(orig_map > 0)[0] #Only select regions with positive mass. Zero mass pixels don't matter
        
        c_pix, c_weight = hp.get_interp_weights(NSIDE, new_ang[p_pix, 0], new_ang[p_pix, 1], lonlat = True)
        c_pix, c_weight = c_pix.T, c_weight.T
        
        new_map = regrid_pixels_hpix(new_map, orig_map[p_pix], c_pix, c_weight)
        
        self.output(new_map)

        return new_map
    

class PaintProfilesShell(DefaultRunner):

    def process(self):

        cosmo_fiducial = FlatwCDM(H0 = self.cosmo['h'] * 100. * u.km / u.s / u.Mpc,
                                  Om0 = self.cosmo['Omega_m'], w0 = self.cosmo['w0'])


        cosmo = ccl.Cosmology(Omega_c = self.cosmo['Omega_m'] - self.cosmo['Omega_b'],
                              Omega_b = self.cosmo['Omega_b'], h = self.cosmo['h'],
                              sigma8  = self.cosmo['sigma8'],  n_s = self.cosmo['n_s'],
                              matter_power_spectrum = 'linear')
        cosmo.compute_sigma()

        healpix_inds = np.arange(hp.nside2npix(self.LightconeShell.NSIDE), dtype = int)

        orig_map = self.LightconeShell.map
        new_map  = np.zeros_like(orig_map).astype(np.float64)
        NSIDE    = self.LightconeShell.NSIDE

        assert self.model is not None, "You must provide a model"
        Baryons  = self.model

        z_t = np.linspace(0, 30, 1000)
        D_a = interpolate.interp1d(z_t, cosmo_fiducial.angular_diameter_distance(z_t).value)
        
        for j in tqdm(range(self.HaloLightConeCatalog.cat.size), desc = 'Painting SZ', disable = not self.verbose):

            M_j = self.HaloLightConeCatalog.cat['M'][j]
            z_j = self.HaloLightConeCatalog.cat['z'][j]
            a_j = 1/(1 + z_j)
            R_j = self.mass_def.get_radius(cosmo, M_j, a_j) #in physical Mpc
            D_j = D_a(z_j) #also physical Mpc since Ang. Diam. Dist.
            
            ra_j   = self.HaloLightConeCatalog.cat['ra'][j]
            dec_j  = self.HaloLightConeCatalog.cat['dec'][j]
            vec_j  = hp.ang2vec(ra_j, dec_j, lonlat = True)
            
            radius = R_j * self.epsilon_max / D_j
            pixind = hp.query_disc(self.LightconeShell.NSIDE, vec_j, radius, inclusive = False, nest = False)
            vec    = np.stack(hp.pix2vec(nside = NSIDE, ipix = pixind), axis = 1)
            
            pos_j  = vec_j * D_j #We assume flat cosmologies, where D_a is the right distance to use here
            pos    = vec   * D_j
            diff   = pos - pos_j
            r_sep  = np.sqrt(np.sum(diff**2, axis = 1))
            
            #Compute the painted map
            Paint  = Baryons.projected(cosmo, r_sep/a_j, M_j, a_j)
            Paint  = np.where(np.isfinite(Paint), Paint, 0) #Set non-finite tSZ values to 0
            
            #Add the profiles to the new healpix map
            new_map[pixind] += Paint         

        self.output(new_map)

        return new_map
