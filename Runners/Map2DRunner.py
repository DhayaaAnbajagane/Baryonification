
import numpy as np
import pyccl as ccl

from scipy import interpolate
from tqdm import tqdm
from numba import njit

MY_FILL_VAL = np.NaN

from ..utils.debug import log_time


@njit
def regrid_pixels_2D(grid, pix_positions, pix_values):

    for pix_pos, pix_value in zip(pix_positions, pix_values):

        N = grid.shape[0]
        x_start, y_start = pix_pos
        x_start, y_start = x_start % N, y_start % N #To handle edge-case where offset >> Lbox_sim
        x_end, y_end     = x_start + 1, y_start + 1

        for i in range(N):
            for j in range(N):

                #Find intersection length
                dx = min(j + 1, x_end) - max(j, x_start)
                dy = min(i + 1, y_end) - max(i, y_start)

                #Now account for periodic boundary conditions
                if dx < 0: dx = min(j + 1, x_end + N) - max(j, x_start + N)
                if dx < 0: dx = min(j + 1, x_end - N) - max(j, x_start - N)

                if dy < 0: dy = min(i + 1, y_end + N) - max(i, y_start + N)
                if dy < 0: dy = min(i + 1, y_end - N) - max(i, y_start - N)

                #If there is some intersection, then add
                if (dx > 0) & (dy > 0):
                    overlap_area = dx * dy
                    grid[i, j] += overlap_area * pix_value


@njit
def regrid_pixels_3D(grid, pix_positions, pix_values):

    for pix_pos, pix_value in zip(pix_positions, pix_values):

        N = grid.shape[0]
        x_start, y_start, z_start = pix_pos
        x_start, y_start, z_start = x_start % N, y_start % N, z_start % N #To handle edge-case where offset >> Lbox_sim
        x_end, y_end, z_end       = x_start + 1, y_start + 1, z_start + 1

        for i in range(N):
            for j in range(N):
                for k in range(N):

                    #Find intersection length
                    dx = min(j + 1, x_end) - max(j, x_start)
                    dy = min(i + 1, y_end) - max(i, y_start)
                    dz = min(k + 1, z_end) - max(k, z_start)

                    #Now account for periodic boundary conditions
                    if dx < 0: dx = min(j + 1, x_end + N) - max(j, x_start + N)
                    if dx < 0: dx = min(j + 1, x_end - N) - max(j, x_start - N)

                    if dy < 0: dy = min(i + 1, y_end + N) - max(i, y_start + N)
                    if dy < 0: dy = min(i + 1, y_end - N) - max(i, y_start - N)
                        
                    if dz < 0: dz = min(k + 1, z_end + N) - max(k, z_start + N)
                    if dz < 0: dz = min(k + 1, z_end - N) - max(k, z_start - N)

                    #If there is some intersection, then add
                    if (dx > 0) & (dy > 0) & (dz > 0):
                        overlap_vol = dx * dy * dz
                        grid[i, j, k] += overlap_vol * pix_value
                    

#Quickly run the function once so it compiles and initializes
regrid_pixels_2D(np.zeros([5, 5]),    np.ones([2, 2]), np.ones(2))
regrid_pixels_3D(np.zeros([5, 5, 5]), np.ones([2, 3]), np.ones(2))

                        
class DefaultRunnerGrid(object):
    '''
    A class that contains relevant utils for input/output
    '''
    
    def __init__(self, HaloNDCatalog, GriddedMap, config, model = None, use_ellipticity = False,
                 mass_def = ccl.halos.massdef.MassDef(200, 'critical'), verbose = True):

        self.HaloNDCatalog = HaloNDCatalog
        self.GriddedMap    = GriddedMap
        self.cosmo = HaloNDCatalog.cosmology
        self.model = model
        
        
        self.mass_def = mass_def
        self.verbose  = verbose
        
        self.config   = self.set_config(config)

        self.use_ellipticity = use_ellipticity
        
        #Assert that all the required quantities are in the input catalog
        if use_ellipticity:
            
            names = HaloNDCatalog.cat.dtype.names
            
            assert 'q_ell' in names, "The 'q_ell' column is missing, but you set use_ellipticity = True"
            if not GriddedMap.is2D: assert 'c_ell' in names, "The 'c_ell' column is missing, but you set use_ellipticity = True"
            assert 'A_ell' in names, "The 'A_ell' column is missing, but you set use_ellipticity = True"


    def set_config(self, config):

        #Dictionary to hold all the params
        out = {}

        out['OutPath']   = config.get('OutPath', None)
        out['Name']      = config.get('Name', '')

        out['epsilon_max_Cutout'] = config.get('epsilon_max_Cutout', 5)
        out['epsilon_max_Offset'] = config.get('epsilon_max_Offset', 5)

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
            np.save(path_, X)
                        
            if self.verbose: print("WRITING TO ", path_)
            
        else:
            
            if self.verbose: print("OutPath is not string. Map is not saved to disk")

    def build_Rmat(self, A, q):

        A /= np.linalg.norm(A)

        if len(A) == 1:
            raise  ValueError("Can't rotate a 1-dimensional vector")
        
        elif len(A) == 2:
            
            #The 2D rotation is done using routines implemented in the galsim Shear class
            
            ref  = np.array([1., 0.])
            beta = np.arccos(np.dot(A, ref))
            eta  = -np.log(q) 
            
            if eta > 1e-4:
                eta2g = np.tanh(0.5*eta)/eta
            else:
                etasq = eta * eta
                eta2g = 0.5 + etasq*((-1/24) + etasq*(1/240))

            g   = eta2g * eta * np.exp(2j * beta)
            g1  = g.real
            g2  = g.imag

            det  = np.sqrt(1 - np.abs(g)**2)
            Rmat = np.array([[1 + g1, g2],
                             [g2, 1 - g1]]) / det
        
        elif len(A) == 3:
            
            raise ValueError("This method has not yet been verified. Use 2D ellipticity method instead")

            v = np.cross(A, ref)
            c = np.dot(A, ref)
            s = np.linalg.norm(v)
            kmat = np.array([[0, -v[2], v[1]], 
                             [v[2], 0, -v[0]], 
                             [-v[1], v[0], 0]])

            Rmat = np.eye(3) + kmat + np.dot(kmat, kmat) * ((1 - c) / (s ** 2))

        return Rmat
        
    def coord_array(self, *args):

        return np.vstack([a.flatten() for a in args]).T

    
class BaryonifyGrid(DefaultRunnerGrid):

    
    
    def pick_indices(self, center, width, Npix):
        
        inds = np.arange(center - width, center + width)
        inds = np.where((inds) < 0,     inds + Npix, inds)
        inds = np.where((inds) >= Npix, inds - Npix, inds)
        
        return inds
    
    def process(self):

        
        cosmo = ccl.Cosmology(Omega_c = self.cosmo['Omega_m'] - self.cosmo['Omega_b'],
                              Omega_b = self.cosmo['Omega_b'], h = self.cosmo['h'],
                              sigma8  = self.cosmo['sigma8'],  n_s = self.cosmo['n_s'],
                              matter_power_spectrum = 'linear')
        cosmo.compute_sigma()

        orig_map = self.GriddedMap.map
        new_map  = np.zeros(orig_map.shape, dtype = np.float64)
        bins     = self.GriddedMap.bins

        orig_map_flat = orig_map.flatten()
        pix_offsets   = np.zeros([orig_map_flat.size, len(orig_map.shape)])

        for j in tqdm(range(self.HaloNDCatalog.cat.size), desc = 'Baryonifying matter', disable = not self.verbose):

            M_j = self.HaloNDCatalog.cat['M'][j]
            x_j = self.HaloNDCatalog.cat['x'][j]
            y_j = self.HaloNDCatalog.cat['y'][j]
            z_j = self.HaloNDCatalog.cat['z'][j] #THIS IS A CARTESIAN COORDINATE, NOT REDSHIFT
            
            #Other properties
            keys = vars(self.model).get('p_keys', []) #Check if model has property keys
            o_j = {key : self.HaloNDCatalog.cat[key][j] for key in keys} 

            a_j = 1/(1 + self.HaloNDCatalog.redshift)
            R_j = self.mass_def.get_radius(cosmo, M_j, a_j) #in physical Mpc
            R_q = self.config['epsilon_max_Cutout'] * R_j/a_j
            R_q = np.clip(R_q, 0, np.max(self.GriddedMap.bins)/2) #Can't query distances more than half box-size.
            
            if self.use_ellipticity:
                q_j = self.HaloNDCatalog.cat['q_ell'][j]
                A_j = self.HaloNDCatalog.cat['A_ell'][j]
                A_j = A_j/np.sqrt(np.sum(A_j**2))
            
            res    = self.GriddedMap.res
            Nsize  = 2 * R_q / res
            Nsize  = int(Nsize // 2)*2 #Force it to be even
            if Nsize < 2: continue #Skip if halo is too small because the displacements will be zero anyway then.

            x  = np.linspace(-Nsize/2, Nsize/2, Nsize) * res
            cutout_width = Nsize//2
            
            if self.GriddedMap.is2D:

                shape = (Nsize, Nsize)
                
                x_cen  = np.argmin(np.abs(bins - x_j))
                y_cen  = np.argmin(np.abs(bins - y_j))
                x_inds = self.pick_indices(x_cen, cutout_width, self.GriddedMap.Npix)
                y_inds = self.pick_indices(y_cen, cutout_width, self.GriddedMap.Npix)
                inds   = self.GriddedMap.inds[x_inds, :][:, y_inds].flatten()
                
                #Get offsets between halo position and pixel center
                dx = bins[x_cen] - x_j
                dy = bins[y_cen] - y_j
                
                assert np.logical_and(dx <= res, dy <= res), "Halo offsets (%0.2f, %0.2f) are larger than res (%0.2f)" % (dx, dy, res)
                
                x_grid, y_grid = np.meshgrid(x, x, indexing = 'xy')
                r_grid = np.sqrt( (x_grid + dx)**2 +  (y_grid + dy)**2 )

                x_hat  = (x_grid + dx)/r_grid
                y_hat  = (y_grid + dy)/r_grid

                #If ellipticity exists, then account for it
                if self.use_ellipticity:
                    assert q_j > 0, "The axis ratio in halo %d is not positive" % j

                    Rmat = self.build_Rmat(A_j, q_j)
                    x_grid_ell, y_grid_ell = (self.coord_array(x_grid + dx, y_grid + dy) @ Rmat).T
                    r_grid = np.sqrt(x_grid_ell**2 + y_grid_ell**2).reshape(x_grid_ell.shape)

                #Compute the displacement needed and add it to pixel offsets
                offset = self.model.displacement(r_grid.flatten()/a_j, M_j, a_j, **o_j) * a_j / res
                pix_offsets[inds, 0] += offset * x_hat.flatten()
                pix_offsets[inds, 1] += offset * y_hat.flatten()
                
            
            else:
                shape = (Nsize, Nsize, Nsize)

                x_cen  = np.argmin(np.abs(bins - x_j))
                y_cen  = np.argmin(np.abs(bins - y_j))
                z_cen  = np.argmin(np.abs(bins - z_j))
                x_inds = self.pick_indices(x_cen, cutout_width, self.GriddedMap.Npix)
                y_inds = self.pick_indices(y_cen, cutout_width, self.GriddedMap.Npix)
                z_inds = self.pick_indices(z_cen, cutout_width, self.GriddedMap.Npix)
                inds   = self.GriddedMap.inds[x_inds, ...][:, y_inds, :][..., z_inds].flatten()
                
                #Get offsets between halo position and pixel center
                dx = bins[x_cen] - x_j
                dy = bins[y_cen] - y_j
                dz = bins[z_cen] - z_j
                
                x_grid, y_grid, z_grid = np.meshgrid(x, x, x, indexing = 'xy')
                r_grid = np.sqrt( (x_grid + dx)**2 +  (y_grid + dy)**2 +  (z_grid + dz)**2 )

                x_hat  = (x_grid + dx)/r_grid
                y_hat  = (y_grid + dy)/r_grid
                z_hat  = (z_grid + dz)/r_grid
                
                #If ellipticity exists, then account for it
                if self.use_ellipticity:
                    assert q_j > 0, "The axis ratio in halo %d is zero" % j

                    Rmat = self.build_Rmat(A_j, np.array([0., 1., 0.]))
                    x_grid_ell, y_grid_ell, z_grid_ell = (self.coord_array(x_grid + dx, y_grid + dy, z_grid + dz) @ Rmat).T
                    r_grid = np.sqrt(x_grid_ell**2/ar_j**2 + 
                                     y_grid_ell**2/br_j**2 +
                                     z_grid_ell**2/cr_j**2).reshape(x_grid_ell.shape)

                
                #Compute the displacement needed    
                offset = self.model.displacement(r_grid.flatten()/a_j, M_j, a_j, **o_j) * a_j / res
                pix_offsets[inds, 0] += offset * x_hat.flatten()
                pix_offsets[inds, 1] += offset * y_hat.flatten()
                pix_offsets[inds, 2] += offset * z_hat.flatten()
            
            
        #Now that pixels have all been offset, let's regrid the map
        N = orig_map.shape[0]
        x = np.arange(N)
        
         #Need to split  2D vs 3D since we have separate numba functions for each
        if self.GriddedMap.is2D:
            x_grid, y_grid = np.meshgrid(x, x, indexing = 'xy')
        
            pix_offsets = np.where(np.isfinite(pix_offsets), pix_offsets, 0)
            pix_offsets[:, 0] += x_grid.flatten()
            pix_offsets[:, 1] += y_grid.flatten()
            
            #Add pixels to the array. Calculations happen in-place
            regrid_pixels_2D(new_map, pix_offsets, orig_map_flat)
            
        else:
            x_grid, y_grid, z_grid = np.meshgrid(x, x, x, indexing = 'xy')
        
            pix_offsets = np.where(np.isfinite(pix_offsets), pix_offsets, 0)
            pix_offsets[:, 0] += x_grid.flatten()
            pix_offsets[:, 1] += y_grid.flatten()
            pix_offsets[:, 2] += z_grid.flatten()
            
            #Add pixels to the array. Calculations happen in-place
            regrid_pixels_3D(new_map, pix_offsets, orig_map_flat)
            
            
        #Do a quick check that the sum is the same
        new_sum = np.sum(new_map)
        old_sum = np.sum(orig_map_flat)
        assert np.isclose(new_sum, old_sum), "ERROR in pixel regridding, sum(new_map) [%0.14e] != sum(oldmap) [%0.14e]" % (new_sum, old_sum)
            
        self.output(new_map)

        return new_map


class PaintProfilesGrid(DefaultRunnerGrid):

    
    def pick_indices(self, center, width, Npix):
        
        inds = np.arange(center - width, center + width)
        inds = np.where((inds) < 0,     inds + Npix, inds)
        inds = np.where((inds) >= Npix, inds - Npix, inds)
        
        return inds
    
    
    def process(self):

        cosmo = ccl.Cosmology(Omega_c = self.cosmo['Omega_m'] - self.cosmo['Omega_b'],
                              Omega_b = self.cosmo['Omega_b'], h = self.cosmo['h'],
                              sigma8  = self.cosmo['sigma8'],  n_s = self.cosmo['n_s'],
                              matter_power_spectrum = 'linear')
        cosmo.compute_sigma()

        orig_map = self.GriddedMap.map
        new_map  = np.zeros(orig_map.size, dtype = np.float64)
        
        grid = self.GriddedMap.grid
        bins = self.GriddedMap.bins


        for j in tqdm(range(self.HaloNDCatalog.cat.size), desc = 'Baryonifying matter', disable = not self.verbose):

            M_j = self.HaloNDCatalog.cat['M'][j]
            x_j = self.HaloNDCatalog.cat['x'][j]
            y_j = self.HaloNDCatalog.cat['y'][j]
            z_j = self.HaloNDCatalog.cat['z'][j] #THIS IS A CARTESIAN COORDINATE, NOT REDSHIFT

            #Other properties
            keys = vars(self.model).get('p_keys', []) #Check if model has property keys
            o_j = {key : self.HaloNDCatalog.cat[key][j] for key in keys} 
            
            a_j = 1/(1 + self.HaloNDCatalog.redshift)
            R_j = self.mass_def.get_radius(cosmo, M_j, a_j) #in physical Mpc

            if self.use_ellipticity:
                q_j = self.HaloNDCatalog.cat['q_ell'][j]
                A_j = self.HaloNDCatalog.cat['A_ell'][j]
                A_j = A_j/np.sqrt(np.sum(A_j**2))
            
            res    = self.GriddedMap.res
            Nsize  = 2 * self.config['epsilon_max_Cutout'] * R_j / res
            Nsize  = int(Nsize // 2)*2 #Force it to be even
            Nsize  = np.clip(Nsize, 2, bins.size//2) #Can't skip small halos because we still must sum all contributions to a pixel

            x = np.linspace(-Nsize/2, Nsize/2, Nsize) * res
            cutout_width = Nsize//2

            if self.GriddedMap.is2D:
                
                x_cen  = np.argmin(np.abs(bins - x_j))
                y_cen  = np.argmin(np.abs(bins - y_j))
                x_inds = self.pick_indices(x_cen, cutout_width, self.GriddedMap.Npix)
                y_inds = self.pick_indices(y_cen, cutout_width, self.GriddedMap.Npix)
                inds   = self.GriddedMap.inds[x_inds, :][:, y_inds].flatten()
                
                #Get offsets between halo position and pixel center
                dx = bins[x_cen] - x_j
                dy = bins[y_cen] - y_j
                
                assert np.logical_and(dx <= res, dy <= res), "Halo offsets (%0.2f, %0.2f) are larger than res (%0.2f)" % (dx, dy, res)
                
                profile = self.model.projected

                x_grid, y_grid = np.meshgrid(x, x, indexing = 'xy')
                r_grid = np.sqrt( (x_grid + dx)**2 +  (y_grid + dy)**2 )

                #If ellipticity exists, then account for it
                if self.use_ellipticity:
                    assert q_j > 0, "The axis ratio in halo %d is zero" % j

                    Rmat = self.build_Rmat(A_j, q_j)
                    x_grid_ell, y_grid_ell = (self.coord_array(x_grid + dx, y_grid + dy) @ Rmat).T
                    r_grid = np.sqrt(x_grid_ell**2 + y_grid_ell**2).reshape(x_grid_ell.shape)
            
            else:
                
                shape  = (Nsize, Nsize, Nsize)
                x_cen  = np.argmin(np.abs(bins - x_j))
                y_cen  = np.argmin(np.abs(bins - y_j))
                z_cen  = np.argmin(np.abs(bins - z_j))
                x_inds = self.pick_indices(x_cen, cutout_width, self.GriddedMap.Npix)
                y_inds = self.pick_indices(y_cen, cutout_width, self.GriddedMap.Npix)
                z_inds = self.pick_indices(z_cen, cutout_width, self.GriddedMap.Npix)
                inds   = self.GriddedMap.inds[x_inds, ...][:, y_inds, :][..., z_inds].flatten()
                
                #Get offsets between halo position and pixel center
                dx = bins[x_cen] - x_j
                dy = bins[y_cen] - y_j
                dz = bins[z_cen] - z_j
                
                profile = self.model.real

                x_grid, y_grid, z_grid = np.meshgrid(x, x, x, indexing = 'xy')
                r_grid = np.sqrt( (x_grid + dx)**2 +  (y_grid + dy)**2 +  (z_grid + dz)**2 )
                

                #If ellipticity exists, then account for it
                if self.use_ellipticity:
                    
                    raise ValueError("use_ellipticity is not implemented for 3D maps")
                    
                    assert q_j > 0, "The axis ratio in halo %d is zero" % j

                    Rmat = self.build_Rmat(A_j, np.array([0., 1., 0.]))
                    x_grid_ell, y_grid_ell, z_grid_ell = (self.coord_array(x_grid + dx, y_grid + dy, z_grid + dz) @ Rmat).T
                    r_grid = np.sqrt(x_grid_ell**2/ar_j**2 + 
                                     y_grid_ell**2/br_j**2 +
                                     z_grid_ell**2/cr_j**2).reshape(x_grid_ell.shape)

        
            Painting = profile(cosmo, r_grid.flatten()/a_j, M_j, a_j, **o_j)
            
            mask = np.isfinite(Painting) #Find which part of map cannot be modified due to out-of-bounds errors
            mask = mask & (r_grid.flatten()/a_j < R_j*self.config['epsilon_max_Offset'])
            if mask.sum() == 0: continue
                
            Painting = np.where(mask, Painting, 0) #Set those tSZ values to 0

            #Add the offsets to the new map at the right indices
            new_map[inds] += Painting
            
        new_map = new_map.reshape(orig_map.shape)

        self.output(new_map)

        return new_map
    


class PaintProfilesAnisGrid(DefaultRunnerGrid):


    def __init__(self, HaloNDCatalog, GriddedMap, config, Painting_model = None, Canvas_model = None, Nbin_interp = 1_000,
                 mass_def = ccl.halos.massdef.MassDef(200, 'critical'), verbose = True):
        
        self.Canvas_model = Canvas_model
        self.Nbin_interp  = Nbin_interp

        super().__init__(HaloNDCatalog, GriddedMap, config, Painting_model, mass_def, verbose)
    

    def pick_indices(self, center, width, Npix):
        
        inds = np.arange(center - width, center + width)
        inds = np.where((inds) < 0,     inds + Npix, inds)
        inds = np.where((inds) >= Npix, inds - Npix, inds)
        
        return inds
    
    
    def process(self):

        assert self.GriddedMap.is2D == True, "Can only paint tSZ on 2D maps. You have passed a 3D Map"

        cosmo = ccl.Cosmology(Omega_c = self.cosmo['Omega_m'] - self.cosmo['Omega_b'],
                              Omega_b = self.cosmo['Omega_b'], h = self.cosmo['h'],
                              sigma8  = self.cosmo['sigma8'],  n_s = self.cosmo['n_s'],
                              matter_power_spectrum = 'linear')
        cosmo.compute_sigma()

        orig_map = self.GriddedMap.map
        new_map  = np.zeros(orig_map.size, dtype = np.float64)

        orig_map_flattened = orig_map.flatten()
        
        grid = self.GriddedMap.grid
        bins = self.GriddedMap.bins

        Paint  = self.model
        Canvas = self.Canvas_model
        
        assert Paint.p_keys is Canvas.p_keys

        for j in tqdm(range(self.HaloNDCatalog.cat.size), desc = 'Baryonifying matter', disable = not self.verbose):

            M_j = self.HaloNDCatalog.cat['M'][j]
            x_j = self.HaloNDCatalog.cat['x'][j]
            y_j = self.HaloNDCatalog.cat['y'][j]
            z_j = self.HaloNDCatalog.cat['z'][j] #THIS IS A CARTESIAN COORDINATE, NOT REDSHIFT

            #Other properties
            keys = vars(self.model).get('p_keys', []) #Check if model has property keys
            o_j = {key : self.HaloNDCatalog.cat[key][j] for key in keys} 
            
            a_j = 1/(1 + self.HaloNDCatalog.redshift)
            R_j = self.mass_def.get_radius(cosmo, M_j, a_j) #in physical Mpc
            
            res    = self.GriddedMap.res
            Nsize  = 2 * self.config['epsilon_max_Cutout'] * R_j / res
            Nsize  = int(Nsize // 2)*2 #Force it to be even
            
            if Nsize < 2:
                continue

            x  = np.linspace(-Nsize/2, Nsize/2, Nsize) * res
            cutout_width = Nsize//2

            if self.GriddedMap.is2D:
                x_grid, y_grid = np.meshgrid(x, x, indexing = 'xy')
                r_grid = np.sqrt(x_grid**2 + y_grid**2)

                x_inds = self.pick_indices(np.argmin(np.abs(bins - x_j)), cutout_width, self.GriddedMap.Npix)
                y_inds = self.pick_indices(np.argmin(np.abs(bins - y_j)), cutout_width, self.GriddedMap.Npix)
                
                inds = self.GriddedMap.inds[x_inds, :][:, y_inds].flatten()
                
                paint_profile  = Paint.projected
                canvas_profile = Canvas.projected 
            
            else:
                x_grid, y_grid, z_grid = np.meshgrid(x, x, x, indexing = 'xy')
                r_grid = np.sqrt(x_grid**2 + y_grid**2 + z_grid**2)

                x_inds = self.pick_indices(np.argmin(np.abs(bins - x_j)), cutout_width, self.GriddedMap.Npix)
                y_inds = self.pick_indices(np.argmin(np.abs(bins - y_j)), cutout_width, self.GriddedMap.Npix)
                z_inds = self.pick_indices(np.argmin(np.abs(bins - z_j)), cutout_width, self.GriddedMap.Npix)
                
                inds = self.GriddedMap.inds[x_inds, ...][:, y_inds, :][..., z_inds].flatten()
                
                paint_profile  = Paint.real
                canvas_profile = Canvas.real 

        

            r_array   = np.geomspace(np.min(r_grid)/a_j, np.max(r_grid)/a_j, self.Nbin_interp)
            Painting  = paint_profile(cosmo,  r_array, M_j, a_j, **o_j)
            Canvasing = canvas_profile(cosmo, r_array, M_j, a_j, **o_j)
            
            gmask     = np.isfinite(Painting) & np.isfinite(Canvasing)
            Painting  = Painting[gmask]
            Canvasing = Canvasing[gmask]
            
            sort_ind  = np.argsort(Canvasing) #Need ascending order for CubicSpline to work
            Painting  = Painting[sort_ind]
            Canvasing = Canvasing[sort_ind]
            
            interp    = interpolate.CubicSpline(np.log(Canvasing), np.log(Painting), extrapolate = False)
            delta_in  = np.log(orig_map_flattened[inds])

            Painting  =  np.exp(interp(delta_in))
            
            mask = np.isfinite(Painting) #Find which part of map cannot be modified due to out-of-bounds errors
            mask = mask & (r_grid.flatten()/a_j < R_j*self.config['epsilon_max_Offset'])
            if mask.sum() == 0: continue
            
            Painting = np.where(mask, Painting, 0) #Set those tSZ values to 0

            #Add the values to the new grid map
            new_map[inds] += Painting
             
        new_map = new_map.reshape(orig_map.shape)

        self.output(new_map)

        return new_map
    
    
    