
import numpy as np
import pyccl as ccl
import healpy as hp

from scipy import interpolate
from astropy.cosmology import FlatwCDM
from astropy import units as u
from astropy.io import fits

from tqdm import tqdm

MY_FILL_VAL = np.NaN


class DefaultRunner(object):
    '''
    A class that contains relevant utils for input/output
    '''
    
    def __init__(self, HaloLightConeCatalog, LightconeShell, config, model = None, use_ellipticity = False,
                 mass_def = ccl.halos.massdef.MassDef(200, 'critical'), verbose = True):

        self.HaloLightConeCatalog  = HaloLightConeCatalog
        self.LightconeShell        = LightconeShell
        self.cosmo = HaloLightConeCatalog.cosmology
        self.model = model
        
        
        self.mass_def = mass_def
        self.verbose  = verbose
        
        self.config   = self.set_config(config)

        self.use_ellipticity = use_ellipticity


    def set_config(self, config):

        #Dictionary to hold all the params
        out = {}

        out['OutPath']   = config.get('OutPath', None)
        out['Name']      = config.get('Name', '')

        out['epsilon_max_Cutout'] = config.get('epsilon_max_Cutout', 5)
        out['epsilon_max_Offset'] = config.get('epsilon_max_Offset', 5)
        out['pixel_scale_factor'] = config.get('pixel_scale_factor', 0.5)


        if self.verbose:
            #Print args for debugging state
            print('-------UPDATING INPUT PARAMS----------')
            for p in out.keys():
                print('%s : %s'%(p.upper(), out[p]))
            print('-----------------------------')
            print('-----------------------------')

        return out
    
    
    def output(self, X):
        
        if isinstance(self.config['OutPath'], str):

            path_ = self.config['OutPath']
            hdu   = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(X)])
            hdu.writeto(path_, overwrite = True)
                        
            if self.verbose: print("WRITING TO ", path_)
            
        else:
            
            if self.verbose: print("OutPath is not string. Map is not saved to disk")
    
    
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
        new_map  = orig_map.copy()

        res        = self.config['pixel_scale_factor'] * hp.nside2resol(self.LightconeShell.NSIDE)
        res_arcmin = res * 180/np.pi * 60

        z_t = np.linspace(0, 10, 1000)
        D_a = interpolate.interp1d(z_t, cosmo_fiducial.angular_diameter_distance(z_t).value)
        
        for j in tqdm(range(self.HaloLightConeCatalog.cat.size), desc = 'Baryonifying matter', disable = not self.verbose):

            M_j = self.HaloLightConeCatalog.cat['M'][j]
            z_j = self.HaloLightConeCatalog.cat['z'][j]
            a_j = 1/(1 + z_j)
            R_j = self.mass_def.get_radius(cosmo, M_j, a_j) #in physical Mpc
            D_j = D_a(z_j)
            
            c_j = self.HaloNDCatalog.cat['c'][j] if self.model.use_concentration else None

            if self.use_ellipticity:
                ar_j = self.HaloNDCatalog.cat['a_ell'][j]
                br_j = self.HaloNDCatalog.cat['b_ell'][j]
                A_j  = self.HaloNDCatalog.cat['A'][j]
                A_j  = A_j/np.sqrt(np.sum(A_j**2))
            
            
            
            ra_j   = self.HaloLightConeCatalog.cat['ra'][j]
            dec_j  = self.HaloLightConeCatalog.cat['dec'][j]

            Nsize  = 2 * self.config['epsilon_max_Cutout'] * R_j / D_j / res
            Nsize  = int(Nsize // 2)*2 #Force it to be even
            
            if Nsize < 2:
                continue

            x      = np.linspace(-Nsize/2, Nsize/2, Nsize) * res * D_j

            x_grid, y_grid = np.meshgrid(x, x, indexing = 'xy')

            r_grid = np.sqrt(x_grid**2 + y_grid**2)

            x_hat = x_grid/r_grid
            y_hat = y_grid/r_grid

            GnomProjector     = hp.projector.GnomonicProj(xsize = Nsize, reso = res_arcmin)

            map_cutout = GnomProjector.projmap(orig_map, #map1,
                                               lambda x, y, z: hp.vec2pix(self.LightconeShell.NSIDE, x, y, z),
                                               rot=(ra_j, dec_j))

            #Need this because map value doesn't account for pixel
            #size changes when reprojecting. It only resamples the map
            map_cutout *= self.config['pixel_scale_factor']**2

            p_ind      = GnomProjector.projmap(healpix_inds,
                                               lambda x, y, z: hp.vec2pix(self.LightconeShell.NSIDE, x, y, z),
                                               rot=(ra_j, dec_j)).flatten().astype(int)

            p_ind, ind, inv_ind = np.unique(p_ind, return_index = True, return_inverse = True)
            interp_map = interpolate.RegularGridInterpolator((x, x), map_cutout.T, bounds_error = False, fill_value = MY_FILL_VAL)

            #If ellipticity exists, then account for it
            if self.use_ellipticity:
                assert ar_j*br_j > 0, "The axis ratio in halo %d is zero" % j

                Rmat   = self.build_Rmat(A_j, np.array([1, 0]))
                x_grid_ell, y_grid_ell = (self.coord_array(x_grid, y_grid) @ Rmat).T
                r_grid = np.sqrt(x_grid_ell**2/ar_j**2 +  y_grid_ell**2/br_j**2).reshape(x_grid.shape)
                
            
            #Compute the displacement needed
            offset     = self.model.displacements(r_grid.flatten()/a_j, M_j, a_j, c = c_j).reshape(r_grid.shape) * a_j
            
            in_coords  = np.vstack([(x_grid + offset*x_hat).flatten(), (y_grid + offset*y_hat).flatten()]).T
            modded_map = interp_map(in_coords)

            mask       = np.isfinite(modded_map) #Find which part of map cannot be modified due to out-of-bounds errors
            
            if mask.sum() == 0: continue
            
            mass_offsets        = np.where(mask, modded_map - map_cutout.flatten(), 0) #Set those offsets to 0
            mass_offsets[mask] -= np.mean(mass_offsets[mask]) #Enforce mass conservation by making sure total mass moving around is 0

            #Find which healpix pixels each subpixel corresponds to.
            #Get total mass offset per healpix pixel
            healpix_map_offsets = np.bincount(np.arange(len(p_ind))[inv_ind], weights = mass_offsets)

            #Add the offsets to the new healpix map
            new_map[p_ind] += healpix_map_offsets
            

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

        assert self.model is not None, "You MUST provide a model"
        Baryons = self.model

        res        = self.config['pixel_scale_factor'] * hp.nside2resol(self.LightconeShell.NSIDE)
        res_arcmin = res * 180/np.pi * 60

        z_t = np.linspace(0, 10, 1000)
        D_a = interpolate.interp1d(z_t, cosmo_fiducial.angular_diameter_distance(z_t).value)
        
        for j in tqdm(range(self.HaloLightConeCatalog.cat.size), desc = 'Painting SZ', disable = not self.verbose):

            M_j = self.HaloLightConeCatalog.cat['M'][j]
            z_j = self.HaloLightConeCatalog.cat['z'][j]
            a_j = 1/(1 + z_j)
            R_j = self.mass_def.get_radius(cosmo, M_j, a_j) #in physical Mpc
            D_j = D_a(z_j)
            
            dA = (res * D_j)**2 / (a_j**2) #comoving area

            if self.use_ellipticity:
                ar_j = self.HaloNDCatalog.cat['a_ell'][j]
                br_j = self.HaloNDCatalog.cat['b_ell'][j]
                A_j  = self.HaloNDCatalog.cat['A'][j]
                A_j  = A_j/np.sqrt(np.sum(A_j**2))

            ra_j   = self.HaloLightConeCatalog.cat['ra'][j]
            dec_j  = self.HaloLightConeCatalog.cat['dec'][j]

            Nsize  = 2 * self.config['epsilon_max_Cutout'] * R_j / D_j / res
            Nsize  = int(Nsize // 2)*2 #Force it to be even
            
            if Nsize < 2:
                continue

            x      = np.linspace(-Nsize/2, Nsize/2, Nsize) * res * D_j

            x_grid, y_grid = np.meshgrid(x, x, indexing = 'xy')

            r_grid = np.sqrt(x_grid**2 + y_grid**2)

            GnomProjector = hp.projector.GnomonicProj(xsize = Nsize, reso = res_arcmin)
            p_ind         = GnomProjector.projmap(healpix_inds,
                                                  lambda x, y, z: hp.vec2pix(self.LightconeShell.NSIDE, x, y, z),
                                                  rot=(ra_j, dec_j)).flatten().astype(int)

            p_ind, ind, inv_ind = np.unique(p_ind, return_index = True, return_inverse = True)
            
            #If ellipticity exists, then account for it
            if self.use_ellipticity:
                assert ar_j*br_j > 0, "The axis ratio in halo %d is zero" % j

                Rmat   = self.build_Rmat(A_j, np.array([1, 0]))
                x_grid_ell, y_grid_ell = (self.coord_array(x_grid, y_grid) @ Rmat).T
                r_grid = np.sqrt(x_grid_ell**2/ar_j**2 +  y_grid_ell**2/br_j**2).reshape(x_grid.shape)

            Painting = Baryons.projected(cosmo, r_grid.flatten()/a_j, M_j, a_j) * dA
            
            mask = np.isfinite(Painting) #Find which part of map cannot be modified due to out-of-bounds errors
            if mask.sum() == 0: continue
                
            Painting = np.where(mask, Painting, 0) #Set those tSZ values to 0
            
            #Find which healpix pixels each subpixel corresponds to.
            #Get total pressure per healpix pixel
            healpix_map_offsets = np.bincount(np.arange(len(p_ind))[inv_ind], weights = Painting)

            #Add the pressure to the new healpix map
            new_map[p_ind] += healpix_map_offsets            

        self.output(new_map)

        return new_map
