import numpy as np
from astropy.io import fits
import healpy as hp
import pyccl as ccl


class HaloCatalog(object):
    '''
    Class that reads in a halo catalog (in a lightcone)
    and stores it along with other useful quantities.
    Used in baryonification pipeline
    '''

    def __init__(self, ra = None, dec = None, M = None, z = None, path = None, cosmo = None):

        if (path is None) & ((ra is None) | (dec is None) | (M is None) |(z is None)):

            raise ValueError("Need to provide either path to file, or provide all of ra, dec, halo mass and halo redshift")

        elif isinstance(path, str):

            if ('npy' in path) or ('npz' in path):
                cat = np.load(path)
            elif 'fits' in path:
                cat = fits.open(path)[1].data
            else:
                raise ValueError("Please provide a path to one of npy, npz, or fits Table files")

        elif (isinstance(ra, np.ndarray) & isinstance(dec, np.ndarray) &
              isinstance(z, np.ndarray)  & isinstance(M, np.ndarray)):

            dtype = [('M', '>f'), ('z', '>f'), ('ra', '>f'), ('dec', '>f')]
            cat = np.zeros(len(ra), dtype)

            cat['ra']  = ra
            cat['dec'] = dec
            cat['z']   = z
            cat['M']   = M

        self.cat   = cat

        keys = cosmo.keys()
        if not (('Omega_m' in keys) & ('sigma8' in keys) & ('h' in keys) &
                ('Omega_b' in keys) & ('n_s' in keys) & ('w0' in keys)):

            raise ValueError("Not all cosmology parameters provided. I need Omega_m, sigma8, h, sigma8, Omega_b, n_s, w0")
        else:
            self.cosmo = cosmo

    def data(self):

        return self.cat

    def cosmology(self):

        return self.cosmo


class LightconeShell(object):

    '''
    Class that reads in a lightcone shell (i.e a healpix map)
    and stores it along with other useful quantities.
    Used in baryonification pipeline
    '''

    def __init__(self, path, cosmo = None):

        self.map   = hp.read_map(path)
        self.NSIDE = hp.npix2nside(self.map.size)

        keys = cosmo.keys()
        if not (('Omega_m' in keys) & ('sigma8' in keys) & ('h' in keys) &
                ('Omega_b' in keys) & ('n_s' in keys) & ('w0' in keys)):

            raise ValueError("Not all cosmology parameters provided. I need Omega_m, sigma8, h, sigma8, Omega_b, n_s, w0")
        else:
            self.cosmo = cosmo

    def data(self):

        return self.map

    def cosmology(self):

        return self.cosmo
