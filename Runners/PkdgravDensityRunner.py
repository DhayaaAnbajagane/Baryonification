
import numpy as np
import pyccl as ccl

from scipy import interpolate
from astropy.cosmology import z_at_value, FlatLambdaCDM, FlatwCDM
from astropy import units as u

from .Schneider19Profiles import DarkMatterOnly, DarkMatterBaryon
from .ComputeOffsets import Baryonification3D, Baryonification2D
from .utils.io import HaloCatalog, LightconeShell

from tqdm import tqdm

MY_FILL_VAL = np.NaN

class Baryonify2D(object):

    def __init__(self, HaloCatalog, LightconeShell):

        self.HaloCatalog    = HaloCatalog
        self.LightconeShell = LightconeShell
        self.cosmo = HaloCatalog.cosmology()

    def parse_args(self):

        import argparse

        my_parser = argparse.ArgumentParser()

        #Metaparams
        my_parser.add_argument('--OutputDir', action='store', type = str, required = True)
        my_parser.add_argument('--Name',      action='store', type = str, default = '')


        #Schneider Baryonification parameters as described in 1810.08629
        my_parser.add_argument('--epsilon',   action='store', type = float, default = 4.0)
        my_parser.add_argument('--theta_ej',  action='store', type = float, default = 4.0)
        my_parser.add_argument('--theta_co',  action='store', type = float, default = 0.1)
        my_parser.add_argument('--M_c',       action='store', type = float, default = 2e14) #in Msun
        my_parser.add_argument('--mu',        action='store', type = float, default = 0.4)
        my_parser.add_argument('--eta_star',  action='store', type = float, default = 0.3)
        my_parser.add_argument('--eta_cga',   action='store', type = float, default = 0.6)
        my_parser.add_argument('--A',         action='store', type = float, default = 0.09)
        my_parser.add_argument('--M1',        action='store', type = float, default = 3e11) #in Msun
        my_parser.add_argument('--epsilon_h', action='store', type = float, default = 0.015)
        my_parser.add_argument('--a',         action='store', type = float, default = 0.3)
        my_parser.add_argument('--n',         action='store', type = float, default = 2.0)
        my_parser.add_argument('--p',         action='store', type = float, default = 0.3)
        my_parser.add_argument('--q',         action='store', type = float, default = 0.707)

        #HyperParameters
        my_parser.add_argument('--epsilon_max_Cutout', action='store', type = float, default = 10)
        my_parser.add_argument('--epsilon_max_Offset', action='store', type = float, default = 5)
        my_parser.add_argument('--pixel_scale_factor', action='store', type = float, default = 0.5)

        args = vars(my_parser.parse_args())

        #Print args for debugging state
        print('-------INPUT PARAMS----------')
        for p in args.keys():
            print('%s : %s'%(p.upper(), args[p]))
        print('-----------------------------')
        print('-----------------------------')

        self.args = args


    def baryonify(self):

        cosmo_fiducial = FlatwCDM(H0 = self.cosmo['h'] * 100. * u.km / u.s / u.Mpc,
                                  Om0 = self.cosmo['Omega_m'], w0 = self.cosmology['w0'])


        cosmo = ccl.Cosmology(Omega_c = self.cosmo['Omega_m'] - self.cosmo['Omega_b'],
                              Omega_b = self.cosmo['Omega_b'], h = self.cosmo['h'],
                              sigma8  = self.cosmo['sigma8'],  n_s = self.cosmo['n_s'],
                              matter_power_spectrum = 'linear')
        cosmo.compute_sigma()

        healpix_inds = np.arange(hp.nside2npix(self.LightconeShell.NSIDE), dtype = int)

        orig_map = self.LightconeShell.data
        new_map  = orig_map.copy()

        #We interpolate just the 2pt correlation function part
        #since recomputing that for every halo is SLOW
        r_temp  = np.geomspace(1e-10, 1e10, 10_000)
        xi_temp = ccl.correlation_3d(cosmo, a, r_temp)
        xi_temp = interpolate.interp1d(r_temp, xi_temp)


        DMO = DarkMatterOnly(epsilon = self.args['epsilon'],
                             q = self.args['q'], p = self.args['p'], xi_mm = xi_temp, R_range = [1e-5, 40])

        DMB = DarkMatterBaryon(epsilon = self.args['epsilon'], a = self.args['a'], n = self.args['n'],
                               theta_ej = self.args['theta_ej'], theta_co = self.args['theta_co'],
                               M_c = self.args['M_c'], mu = self.args['mu'],
                               A = self.args['A'], M1 = self.args['M1'], epsilon_h = self.args['epsilon_h'],
                               eta_star = self.args['eta_star'], eta_cga = self.args['eta_cga'],
                               q = self.args['q'], p = self.args['p'], xi_mm = xi_temp, R_range = [1e-5, 40])

        Baryons = Baryonification2D(DMO = DMO, DMB = DMB, R_range = [1e-5, 50], N_samples = 500,
                                    epsilon_max = self.args['epsilon_max_Offset'])


        res        = self.args['pixel_scale_factor'] * hp.nside2resol(self.LightconeShell.NSIDE)
        res_arcmin = res * 180/np.pi * 60

        for j in tqdm(range(self.cat.size)):

            R_j = self.cat['R'][j]
            M_j = self.cat['M'][j]
            z_j = self.cat['z'][j]
            a_j = 1/(1 + z_j)
            D_a = cosmo_fiducial.angular_diameter_distance(z_j).value

            ra_j   = self.cat['ra'][j]
            dec_j  = self.cat['ra'][j]

            Nsize  = 2 * self.args['epsilon_max_Cutout'] * R_j*a_j / D_a / res
            Nsize  = int(Nsize // 2)*2 #Force it to be even

            x      = np.linspace(-Nsize/2, Nsize/2, Nsize) * res * D_a

            x_grid, y_grid = np.meshgrid(x, x, indexing = 'xy')

            r_grid = np.sqrt(x_grid**2 + y_grid**2)

            x_hat = x_grid/r_grid
            y_hat = y_grid/r_grid

            GnomProjector = hp.projector.GnomonicProj(xsize = Nsize, reso = res_arcmin)

            displacement_func = Baryons.displacement_func_shell(cosmo, M_j, a_j, epsilon_max = self.args['epsilon_max_Offset'])

            map_cutout = GnomProjector.projmap(orig_map, #map1,
                                               lambda x, y, z: hp.vec2pix(args['NSIDE'], x, y, z),
                                               rot=(ra_j, deca_j))

            map_cutout *= self.args['pixel_scale_factor']**2 #Need this because pixel value doesn't account for pixel size changes

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


        Name = 'Baryonified_Density_shell' + ('_%s'%args['Name'] if args['Name'] is not '' else '')
        path_ = self.args['OutputDir'] + '/' +  Name + '.fits'
        hdu   = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(map1)])
        hdu.writeto(path_, overwrite = True)

        if os.path.exists(path_ + '.fz'): os.remove(path_ + '.fz')

        #Perform fpack. Remove existing fits, and keep only fits.fz
        os.system('fpack -q 8192 %s'%path_)
        os.system('rm %s'%path_)

        return 0


if __name__ == '__main__':

    pass
