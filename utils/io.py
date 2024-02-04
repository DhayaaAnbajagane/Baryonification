import numpy as np
from astropy.io import fits
import healpy as hp
import pyccl as ccl


class HaloLightConeCatalog(object):
    '''
    Class that reads in a halo catalog (in a lightcone)
    and stores it along with other useful quantities.
    Used in baryonification pipeline
    '''

    def __init__(self, ra = None, dec = None, M = None, z = None, path = None, cosmo = None, **arrays):

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
            dtype = dtype + [(name, '>f') for name, arr in arrays.items()]
                
            N   = 1 if not isinstance(ra, (list, np.ndarray, tuple)) else len(ra)
            cat = np.zeros(len(ra), dtype)

            cat['ra']  = ra
            cat['dec'] = dec
            cat['z']   = z
            cat['M']   = M
            
            for name, arr in arrays.items(): cat[name] = arr

        self.cat   = cat

        keys = cosmo.keys()
        if not (('Omega_m' in keys) & ('sigma8' in keys) & ('h' in keys) &
                ('Omega_b' in keys) & ('n_s' in keys) & ('w0' in keys)):

            raise ValueError("Not all cosmology parameters provided. I need Omega_m, sigma8, h, sigma8, Omega_b, n_s, w0")
        else:
            self.cosmo = cosmo

    @property
    def data(self):

        return self.cat

    @property
    def cosmology(self):

        return self.cosmo
    
    
    def __getitem__(self, key):
        
        ra  = self.cat['ra'][key]
        dec = self.cat['dec'][key]
        z   = self.cat['z'][key]
        M   = self.cat['M'][key]
        c   = self.cat['c'][key] if 'c' in self.cat.dtype.names else None
                    
        return HaloLightConeCatalog(ra, dec, M, z, c, cosmo = self.cosmo)
    

class HaloNDCatalog(object):
    '''
    Class that reads in a halo catalog (in a 2D or 3D field)
    and stores it along with other useful quantities.
    Used in baryonification pipeline
    '''

    def __init__(self, x = None, y = None, z = None, M = None, redshift = None, cosmo = None, **arrays):

        dtype = [('M', '>f'), ('x', '>f'), ('y', '>f'), ('z', '>f')]
        dtype = dtype + [(name, '>f', arr.shape[1:] if len(arr.shape) > 1 else '') for name, arr in arrays.items()]
        
        
        N = 1 if not isinstance(x, (list, np.ndarray, tuple)) else len(x)
        cat = np.zeros(N, dtype)

        cat['x'] = x
        cat['y'] = y
        cat['z'] = 0 if z is None else z #We'll just add filler to z-column for now
        cat['M'] = M
        
        for name, arr in arrays.items(): cat[name] = arr

        self.cat = cat
        self.redshift = redshift

        keys = cosmo.keys()
        if not (('Omega_m' in keys) & ('sigma8' in keys) & ('h' in keys) &
                ('Omega_b' in keys) & ('n_s' in keys) & ('w0' in keys)):

            raise ValueError("Not all cosmology parameters provided. I need Omega_m, sigma8, h, sigma8, Omega_b, n_s, w0")
        else:
            self.cosmo = cosmo

    @property
    def data(self):

        return self.cat

    @property
    def cosmology(self):

        return self.cosmo
    
    
    def __getitem__(self, key):
        
        x = self.cat['x'][key]
        y = self.cat['y'][key]
        z = self.cat['z'][key]
        M = self.cat['M'][key]
        
        other = {}
        for key in self.cat.dtype.names:
            if key not in ['x', 'y', 'z', 'M']:
                other[key] = self.cat[key]
        
        return HaloNDCatalog(x, y, z, M, c, self.redshift, self.cosmo, **other)
    
    
    def __str__(self):
        
        string = f"""
        HaloNDCatalog with {self.cat.size} Halos at z = {self.redshift}.
        Minimum Mass = {self.cat['M'].min()}
        Maximum Mass = {self.cat['M'].max()}
        Cosmology set to {self.cosmo}.
        """
        return string.strip()
    

    def __repr__(self):
        
        return f"HaloNDCatalog(cat = {self.cat!r}, \nredshift = {self.redshift!r}, \ncosmo = {self.cosmo})"


class LightconeShell(object):

    '''
    Class that reads in a lightcone shell (i.e a healpix map)
    and stores it along with other useful quantities.
    Used in baryonification pipeline
    '''

    def __init__(self, map = None, path = None, cosmo = None, M_part = None):

        if (path is None) & (map is None):
            raise ValueError("Need to provide either path to map, or provide map values in healpix ring configuration")

        elif isinstance(path, str):
            self.map = hp.read_map(path)

        elif isinstance(map, np.ndarray):
            self.map = map

            
        self.NSIDE  = hp.npix2nside(self.map.size)
        self.M_part = M_part

        
        keys = cosmo.keys()
        if not (('Omega_m' in keys) & ('sigma8' in keys) & ('h' in keys) &
                ('Omega_b' in keys) & ('n_s' in keys) & ('w0' in keys)):

            raise ValueError("Not all cosmology parameters provided. I need Omega_m, sigma8, h, sigma8, Omega_b, n_s, w0")
        else:
            self.cosmo = cosmo

    @property
    def data(self):

        return self.map

    @property
    def cosmology(self):

        return self.cosmo
    

class GriddedMap(object):

    '''
    Class that reads in a Gridded map (either 2D or 3D)
    and stores it along with other useful quantities.
    Used in baryonification pipeline
    '''

    def __init__(self, map = None, redshift = None, bins = None, cosmo = None):
        '''
        bins: Must be coordinates of map along a given axis, in physical Mpc
        '''
        self.map      = map
        self.redshift = redshift
        self.Npix     = self.map.shape[0]
        self.res      = bins[1] - bins[0]
        self.bins     = bins
        
        self.is2D = True if len(self.map.shape) == 2 else False

        if self.is2D:
            assert self.map.shape[0] == self.map.shape[1] #Maps have to be square maps
            self.grid = np.meshgrid(bins, bins, indexing = 'xy')
        else:
            assert (self.map.shape[0] == self.map.shape[1]) & (self.map.shape[1] == self.map.shape[2]) #Maps have to be cubic maps
            self.grid = np.meshgrid(bins, bins, bins, indexing = 'xy')
            
        self.inds = np.arange(self.grid[0].size).reshape(self.grid[0].shape)

        keys = cosmo.keys()
        if not (('Omega_m' in keys) & ('sigma8' in keys) & ('h' in keys) &
                ('Omega_b' in keys) & ('n_s' in keys) & ('w0' in keys)):

            raise ValueError("Not all cosmology parameters provided. I need Omega_m, sigma8, h, sigma8, Omega_b, n_s, w0")
        else:
            self.cosmo = cosmo

    @property
    def data(self):

        return self.map

    @property
    def cosmology(self):

        return self.cosmo


class ParticleSnapshot(object):
    '''
    Class that reads in a particle snapshot (in a 2D or 3D field)
    and stores it along with other useful quantities.
    Used in baryonification pipeline
    '''

    def __init__(self, x = None, y = None, z = None, M = None, L = None, redshift = None, cosmo = None):

        dtype = [('M', np.float64), ('x', np.float64), ('y', np.float64), ('z', np.float64)]
        
        cat = np.zeros(len(x), dtype)

        cat['x'] = x
        cat['y'] = y
        cat['z'] = 0 if z is None else z #We'll just add filler to z-column for now
        cat['M'] = M

        self.L   = L
        self.cat = cat
        self.redshift = redshift

        self.is2D = True if z is None else False

        keys = cosmo.keys()
        if not (('Omega_m' in keys) & ('sigma8' in keys) & ('h' in keys) &
                ('Omega_b' in keys) & ('n_s' in keys) & ('w0' in keys)):

            raise ValueError("Not all cosmology parameters provided. I need Omega_m, sigma8, h, sigma8, Omega_b, n_s, w0")
        else:
            self.cosmo = cosmo


    @property
    def data(self):

        return self.cat

    @property
    def cosmology(self):

        return self.cosmo
    
    
    def make_map(self, N_grid):

        assert np.isnan(self.cat['M']).sum() == 0, "If you want to make a map, provide a value for the particle mass"
        
        bins = np.linspace(0, self.L, N_grid + 1)
        
        if self.is2D:
            coords = np.vstack([self.cat['x'], self.cat['y']]).T
            bins   = (bins, bins)
            
        else:
            coords = np.vstack([self.cat['x'], self.cat['y'], self.cat['z']]).T
            bins   = (bins, bins, bins)
            

        Map = np.histogramdd(coords, bins = bins, weights = self.cat['M'])[0]

        return Map