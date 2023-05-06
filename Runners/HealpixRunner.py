
import numpy as np
import pyccl as ccl

from scipy import interpolate
from astropy.cosmology import z_at_value, FlatLambdaCDM, FlatwCDM
from astropy import units as u

from ..Profiles.Schneider19Profiles import DarkMatterOnly, DarkMatterBaryon
from ..ComputeOffsets import Baryonification3D, Baryonification2D
from ..utils.io import HaloCatalog, LightconeShell

from tqdm import tqdm

MY_FILL_VAL = np.NaN

class Baryonify2D(object):

    def __init__(self, HaloCatalog, LightconeShell, config, Baryonification2D = None):

        self.HaloCatalog    = HaloCatalog
        self.LightconeShell = LightconeShell
        self.cosmo = HaloCatalog.cosmology()
        self.Baryonification2D = Baryonification2D

        self.config = self.set_config(config)

    def set_config(self, config):

        #Dictionary to hold all the params
        out = {}

        out['OutputDir'] = config.get('OutputDir', None)
        out['Name']      = config.get('Name', '')

        out['epsilon']   = config.get('epsilon',  4.0)  #Truncation radius for DM NFW profile normalized by R200c
        out['theta_ej']  = config.get('theta_ej', 4.0)  #Radius up to which gas is ejected, normalized by R200c
        out['theta_co']  = config.get('theta_co', 0.1)  #Radius within which gas is fully collapsed/bound, normalized by R200c
        out['M_c']       = config.get('M_c',      2e14) #in Msun, normalization of the relation between gas profile slope and mass
        out['mu']        = config.get('mu',       0.4)  #slope of the relation between gas profile slope and mass
        out['eta_star']  = config.get('eta_star', 0.3)  #
        out['eta_cga']   = config.get('eta_star', 0.6)
        out['A']         = config.get('A',        0.09)
        out['M1']        = config.get('M1',       3e11) #in Msun
        out['epsilon_h'] = config.get('epsilon_h', 0.015)
        out['a']         = config.get('a', 0.3)
        out['n']         = config.get('n', 2.0)
        out['p']         = config.get('p', 0.3)
        out['q']         = config.get('q', 0.707)

        out['epsilon_max_Cutout'] = config.get('epsilon_max_Cutout', 5)
        out['epsilon_max_Offset'] = config.get('epsilon_max_Offset', 3)
        out['pixel_scale_factor'] = config.get('pixel_scale_factor', 0.1)


        #Print args for debugging state
        print('-------UPDATING INPUT PARAMS----------')
        for p in out.keys():
            print('%s : %s'%(p.upper(), out[p]))
        print('-----------------------------')
        print('-----------------------------')

        return out


    def baryonify(self):

        cosmo_fiducial = FlatwCDM(H0 = self.cosmo['h'] * 100. * u.km / u.s / u.Mpc,
                                  Om0 = self.cosmo['Omega_m'], w0 = self.cosmo['w0'])


        cosmo = ccl.Cosmology(Omega_c = self.cosmo['Omega_m'] - self.cosmo['Omega_b'],
                              Omega_b = self.cosmo['Omega_b'], h = self.cosmo['h'],
                              sigma8  = self.cosmo['sigma8'],  n_s = self.cosmo['n_s'],
                              matter_power_spectrum = 'linear')
        cosmo.compute_sigma()

        healpix_inds = np.arange(hp.nside2npix(self.LightconeShell.NSIDE), dtype = int)

        orig_map = self.LightconeShell.data
        new_map  = orig_map.copy()

        if self.Baryonification2D is None:

            #We interpolate just the 2pt correlation function part
            #since recomputing that for every halo is SLOW
            r_temp  = np.geomspace(1e-10, 1e10, 10_000)
            xi_temp = ccl.correlation_3d(cosmo, a, r_temp)
            xi_temp = interpolate.interp1d(r_temp, xi_temp)


            DMO = DarkMatterOnly(epsilon = self.config['epsilon'],
                                 q = self.config['q'], p = self.config['p'], xi_mm = xi_temp, R_range = [1e-5, 40])

            DMB = DarkMatterBaryon(epsilon = self.config['epsilon'], a = self.config['a'], n = self.config['n'],
                                   theta_ej = self.config['theta_ej'], theta_co = self.config['theta_co'],
                                   M_c = self.config['M_c'], mu = self.config['mu'],
                                   A = self.config['A'], M1 = self.config['M1'], epsilon_h = self.config['epsilon_h'],
                                   eta_star = self.config['eta_star'], eta_cga = self.config['eta_cga'],
                                   q = self.config['q'], p = self.config['p'], xi_mm = xi_temp, R_range = [1e-5, 40])

            Baryons = Baryonification2D(DMO = DMO, DMB = DMB, R_range = [1e-5, 50], N_samples = 500,
                                        epsilon_max = self.config['epsilon_max_Offset'])
        else:

            Baryons = self.Baryonification2D


        res        = self.config['pixel_scale_factor'] * hp.nside2resol(self.LightconeShell.NSIDE)
        res_arcmin = res * 180/np.pi * 60

        for j in tqdm(range(self.cat.size)):

            R_j = self.cat['R'][j]
            M_j = self.cat['M'][j]
            z_j = self.cat['z'][j]
            a_j = 1/(1 + z_j)
            D_a = cosmo_fiducial.angular_diameter_distance(z_j).value

            ra_j   = self.cat['ra'][j]
            dec_j  = self.cat['ra'][j]

            Nsize  = 2 * self.config['epsilon_max_Cutout'] * R_j*a_j / D_a / res
            Nsize  = int(Nsize // 2)*2 #Force it to be even

            x      = np.linspace(-Nsize/2, Nsize/2, Nsize) * res * D_a

            x_grid, y_grid = np.meshgrid(x, x, indexing = 'xy')

            r_grid = np.sqrt(x_grid**2 + y_grid**2)

            x_hat = x_grid/r_grid
            y_hat = y_grid/r_grid

            GnomProjector     = hp.projector.GnomonicProj(xsize = Nsize, reso = res_arcmin)
            displacement_func = Baryons.displacement_func_shell(cosmo, M_j, a_j, epsilon_max = self.config['epsilon_max_Offset'])

            map_cutout = GnomProjector.projmap(orig_map, #map1,
                                               lambda x, y, z: hp.vec2pix(args['NSIDE'], x, y, z),
                                               rot=(ra_j, deca_j))

            #Need this because map value doesn't account for pixel
            #size changes when reprojecting. It only resamples the map
            map_cutout *= self.config['pixel_scale_factor']**2

            p_ind      = GnomProjector.projmap(healpix_inds,
                                               lambda x, y, z: hp.vec2pix(args['NSIDE'], x, y, z),
                                               rot=(ra_j, deca_j)).flatten().astype(int)

            p_ind, ind, inv_ind = np.unique(p_ind, return_index = True, return_inverse = True)
            interp_map = interpolate.RegularGridInterpolator((x, x), map_cutout.T, bounds_error = False, fill_value = MY_FILL_VAL)

            #Compute the displacement needed
            offset     = displacement_func(r_grid.flatten()/a_j).reshape(r_grid.shape) * a_j

            in_coords  = np.vstack([(x_grid + offset*x_hat).flatten(), (y_grid + offset*y_hat).flatten()]).T
            new_map    = interp_map(in_coords)

            mask         = np.isfinite(new_map) #Find which part of map cannot be modified due to out-of-bounds errors
            mass_offsets = np.where(mask, new_map - map_cutout.flatten(), 0) #Set those offsets to 0
            mass_offsets[mask] -= np.mean(mass_offsets[mask]) #Enforce mass conservation by making sure total mass moving around is 0

            #Find which healpix pixels each subpixel corresponds to.
            #Get total mass offset per healpix pixel
            healpix_map_offsets = np.bincount(np.arange(len(p_ind))[inv_ind], weights = mass_offsets)

            #Add the offsets to the new healpix map
            new_map[p_ind] += healpix_map_offsets


        if isinstance(self.config['OutputDir'], str):

            Name = 'Baryonified_Density_shell' + ('_%s'%self.config['Name'] if self.config['Name'] is not '' else '')
            path_ = self.config['OutputDir'] + '/' +  Name + '.fits'
            hdu   = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(map1)])
            hdu.writeto(path_, overwrite = True)

            if os.path.exists(path_ + '.fz'): os.remove(path_ + '.fz')

            #Perform fpack. Remove existing fits, and keep only fits.fz
            os.system('fpack -q 8192 %s'%path_)
            os.system('rm %s'%path_)

        return new_map
