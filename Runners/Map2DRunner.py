
import numpy as np
import pyccl as ccl

from scipy import interpolate
from tqdm import tqdm

MY_FILL_VAL = np.NaN


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
            np.save(path_, X)
                        
            if self.verbose: print("WRITING TO ", path_)
            
        else:
            
            if self.verbose: print("OutPath is not string. Map is not saved to disk")

    def build_Rmat(self, A, ref):

        A   /= np.linalg.norm(A)
        ref /= np.linalg.norm(ref)

        if len(A) == 1:
            raise  ValueError("Can't rotate a 1-dimensional vector")
        
        elif len(A) == 2:
            ang  = np.arccos(np.dot(A, ref))
            Rmat = np.array([[np.cos(ang), -np.sin(ang)], 
                             [np.sin(ang), np.cos(ang)]])
        
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
        new_map  = orig_map.copy().flatten()
        bins     = self.GriddedMap.bins


        for j in tqdm(range(self.HaloNDCatalog.cat.size), desc = 'Baryonifying matter', disable = not self.verbose):

            M_j = self.HaloNDCatalog.cat['M'][j]
            x_j = self.HaloNDCatalog.cat['x'][j]
            y_j = self.HaloNDCatalog.cat['y'][j]
            z_j = self.HaloNDCatalog.cat['z'][j] #THIS IS A CARTESIAN COORDINATE, NOT REDSHIFT
            
            c_j = self.HaloNDCatalog.cat['c'][j] if self.model.use_concentration else None

            a_j = 1/(1 + self.HaloNDCatalog.redshift)
            R_j = self.mass_def.get_radius(cosmo, M_j, a_j) #in physical Mpc

            if self.use_ellipticity:
                ar_j = self.HaloNDCatalog.cat['a_ell'][j]
                br_j = self.HaloNDCatalog.cat['b_ell'][j]
                cr_j = self.HaloNDCatalog.cat['c_ell'][j]
                A_j  = self.HaloNDCatalog.cat['A'][j]
                A_j  = A_j/np.sqrt(np.sum(A_j**2))
            
            res    = self.GriddedMap.res
            Nsize  = 2 * self.config['epsilon_max_Cutout'] * R_j / res
            Nsize  = int(Nsize // 2)*2 #Force it to be even
            
            if Nsize < 2:
                continue

            x  = np.linspace(-Nsize/2, Nsize/2, Nsize) * res
            pixel_width = Nsize//2

            x_hr = np.linspace(-Nsize/2, Nsize/2, Nsize * int(1/self.config['pixel_scale_factor'])) * res #Upscale resolution
            
            if self.GriddedMap.is2D:

                shape = (Nsize, Nsize)
                # x_grid, y_grid = np.meshgrid(x, x, indexing = 'xy')
                # r_grid = np.sqrt(x_grid**2 + y_grid**2)

                # x_hat  = x_grid/r_grid
                # y_hat  = y_grid/r_grid

                x_inds = self.pick_indices(np.argmin(np.abs(bins - x_j)), pixel_width, self.GriddedMap.Npix)
                y_inds = self.pick_indices(np.argmin(np.abs(bins - y_j)), pixel_width, self.GriddedMap.Npix)
                
                inds   = self.GriddedMap.inds_flattened[x_inds, :][:, y_inds].flatten()
                
                map_cutout = self.GriddedMap.map.flatten()[inds].reshape(shape)
                
                interp_map = interpolate.RegularGridInterpolator((x, x), map_cutout.T, bounds_error = False, fill_value = MY_FILL_VAL)

                x_grid_hr, y_grid_hr = np.meshgrid(x_hr, x_hr, indexing = 'xy')
                r_grid_hr = np.sqrt(x_grid_hr**2 + y_grid_hr**2)

                x_hat_hr  = x_grid_hr/r_grid_hr
                y_hat_hr  = y_grid_hr/r_grid_hr

                inds_map_hr = interpolate.NearestNDInterpolator((x_hr, x_hr), inds.reshape(shape).T)(x_grid_hr, y_grid_hr)

                #If ellipticity exists, then account for it
                if self.use_ellipticity:
                    assert ar_j*br_j > 0, "The axis ratio in halo %d is zero" % j

                    Rmat = self.build_Rmat(A_j, np.array([1, 0]))
                    x_grid_hr_ell, y_grid_hr_ell = (self.coord_array(x_grid_hr, y_grid_hr) @ Rmat).T
                    r_grid_hr = np.sqrt(x_grid_hr_ell**2/ar_j**2 + y_grid_hr_ell**2/br_j**2).reshape(x_grid_hr_ell.shape)

                #Compute the displacement needed
                offset     = self.model.displacements(r_grid_hr.flatten()/a_j, M_j, a_j, c = c_j).reshape(r_grid_hr.shape) * a_j
                in_coords  = self.coord_array(x_grid_hr + offset*x_hat_hr, y_grid_hr + offset*y_hat_hr)
                
            
            else:
                shape = (Nsize, Nsize, Nsize)
                # x_grid, y_grid, z_grid = np.meshgrid(x, x, x, indexing = 'xy')
                # r_grid = np.sqrt(x_grid**2 + y_grid**2 + z_grid**2)

                # x_hat  = x_grid/r_grid
                # y_hat  = y_grid/r_grid
                # z_hat  = z_grid/r_grid

                x_inds = self.pick_indices(np.argmin(np.abs(bins - x_j)), pixel_width, self.GriddedMap.Npix)
                y_inds = self.pick_indices(np.argmin(np.abs(bins - y_j)), pixel_width, self.GriddedMap.Npix)
                z_inds = self.pick_indices(np.argmin(np.abs(bins - z_j)), pixel_width, self.GriddedMap.Npix)
                
                inds = self.GriddedMap.inds_flattened[x_inds, ...][:, y_inds, :][..., z_inds].flatten()
                
                map_cutout = self.GriddedMap.map.flatten()[inds].reshape(shape)
                
                interp_map = interpolate.RegularGridInterpolator((x, x, x), map_cutout.T, bounds_error = False, fill_value = MY_FILL_VAL)

                x_grid_hr, y_grid_hr, z_grid_hr = np.meshgrid(x_hr, x_hr, x_hr, indexing = 'xy')
                r_grid_hr = np.sqrt(x_grid_hr**2 + y_grid_hr**2 + z_grid_hr**2)

                x_hat_hr  = x_grid_hr/r_grid_hr
                y_hat_hr  = y_grid_hr/r_grid_hr
                z_hat_hr  = z_grid_hr/r_grid_hr

                inds_map_hr = interpolate.NearestNDInterpolator((x_hr, x_hr, x_hr), inds.reshape(shape).T)(x_grid_hr, y_grid_hr, z_grid_hr)

                #If ellipticity exists, then account for it
                if self.use_ellipticity:
                    assert ar_j*br_j > 0, "The axis ratio in halo %d is zero" % j

                    Rmat = self.build_Rmat(A_j, np.array([1, 0, 0]))
                    x_grid_hr_ell, y_grid_hr_ell, z_grid_hr_ell = (self.coord_array(x_grid_hr, y_grid_hr, z_grid_hr) @ Rmat).T
                    r_grid_hr = np.sqrt(x_grid_hr_ell**2/ar_j**2 + 
                                        y_grid_hr_ell**2/br_j**2 +
                                        z_grid_hr_ell**2/cr_j**2).reshape(x_grid_hr_ell.shape)

                #Compute the displacement needed
                offset     = self.model.displacements(r_grid_hr.flatten()/a_j, M_j, a_j, c = c_j).reshape(r_grid_hr.shape) * a_j
                in_coords  = self.coord_array(x_grid_hr + offset*x_hat_hr, 
                                              y_grid_hr + offset*y_hat_hr, 
                                              z_grid_hr + offset*z_hat_hr)
            
            modded_map = interp_map(in_coords)
            mask       = np.isfinite(modded_map) #Find which part of map cannot be modified due to out-of-bounds errors
        
            if mask.sum() == 0: continue
        
            mass_offsets        = np.where(mask, modded_map - self.GriddedMap.map.flatten()[inds_map_hr], 0) #Set those offsets to 0
            mass_offsets[mask] -= np.mean(mass_offsets[mask]) #Enforce mass conservation by making sure total mass moving around is 0

            #Find which map pixels each subpixel corresponds to.
            #Get total mass offset per map pixel
            p_ind, inv_ind   = np.unique(inds_map_hr, return_inverse = True)
            grid_map_offsets = np.bincount(np.arange(len(p_ind))[inv_ind], weights = mass_offsets)

            #Add the offsets to the new map at the right indices
            new_map[p_ind] += grid_map_offsets

        self.output(new_map)

        return new_map


class PaintProfilesGrid(DefaultRunnerGrid):

    
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
        
        grid = self.GriddedMap.grid
        bins = self.GriddedMap.bins


        for j in tqdm(range(self.HaloNDCatalog.cat.size), desc = 'Baryonifying matter', disable = not self.verbose):

            M_j = self.HaloNDCatalog.cat['M'][j]
            x_j = self.HaloNDCatalog.cat['x'][j]
            y_j = self.HaloNDCatalog.cat['y'][j]
            z_j = self.HaloNDCatalog.cat['z'][j] #THIS IS A CARTESIAN COORDINATE, NOT REDSHIFT

            a_j = 1/(1 + self.HaloNDCatalog.redshift)
            R_j = self.mass_def.get_radius(cosmo, M_j, a_j) #in physical Mpc

            if self.use_ellipticity:
                ar_j = self.HaloNDCatalog.cat['a_ell'][j]
                br_j = self.HaloNDCatalog.cat['b_ell'][j]
                cr_j = self.HaloNDCatalog.cat['c_ell'][j]
                A_j  = self.HaloNDCatalog.cat['A'][j]
                A_j  = A_j/np.sqrt(np.sum(A_j**2))
            
            res    = self.GriddedMap.res
            Nsize  = 2 * self.config['epsilon_max_Cutout'] * R_j / res
            Nsize  = int(Nsize // 2)*2 #Force it to be even
            
            if Nsize < 2:
                continue

            x_hr = np.linspace(-Nsize/2, Nsize/2, Nsize * int(1/self.config['pixel_scale_factor'])) * res #Upscale resolution
            pixel_width = Nsize//2

            if self.GriddedMap.is2D:
                x_inds = self.pick_indices(np.argmin(np.abs(bins - x_j)), pixel_width, self.GriddedMap.Npix)
                y_inds = self.pick_indices(np.argmin(np.abs(bins - y_j)), pixel_width, self.GriddedMap.Npix)
                
                inds = self.GriddedMap.inds_flattened[x_inds, :][:, y_inds].flatten()
                
                profile = self.model.projected

                x_grid_hr, y_grid_hr = np.meshgrid(x_hr, x_hr, indexing = 'xy')
                r_grid_hr = np.sqrt(x_grid_hr**2 + y_grid_hr**2)

                inds_map_hr = interpolate.NearestNDInterpolator((x_hr, x_hr), inds.reshape(x_grid_hr).T)(x_grid_hr, y_grid_hr)

                #If ellipticity exists, then account for it
                if self.use_ellipticity:
                    assert ar_j*br_j > 0, "The axis ratio in halo %d is zero" % j

                    Rmat = self.build_Rmat(A_j, np.array([1, 0]))
                    x_grid_hr_ell, y_grid_hr_ell = (self.coord_array(x_grid_hr, y_grid_hr) @ Rmat).T
            
            else:

                x_inds = self.pick_indices(np.argmin(np.abs(bins - x_j)), pixel_width, self.GriddedMap.Npix)
                y_inds = self.pick_indices(np.argmin(np.abs(bins - y_j)), pixel_width, self.GriddedMap.Npix)
                z_inds = self.pick_indices(np.argmin(np.abs(bins - z_j)), pixel_width, self.GriddedMap.Npix)
                
                inds = self.GriddedMap.inds_flattened[x_inds, ...][:, y_inds, :][..., z_inds].flatten()
                
                profile = self.model.real

                x_grid_hr, y_grid_hr, z_grid_hr = np.meshgrid(x_hr, x_hr, x_hr, indexing = 'xy')
                r_grid_hr = np.sqrt(x_grid_hr**2 + y_grid_hr**2 + z_grid_hr**2)

                inds_map_hr = interpolate.NearestNDInterpolator((x_hr, x_hr, x_hr), inds.reshape(shape).T)(x_grid_hr, y_grid_hr, z_grid_hr)

                #If ellipticity exists, then account for it
                if self.use_ellipticity:
                    assert ar_j*br_j > 0, "The axis ratio in halo %d is zero" % j

                    Rmat = self.build_Rmat(A_j, np.array([1, 0, 0]))
                    x_grid_hr_ell, y_grid_hr_ell, z_grid_hr_ell = (self.coord_array(x_grid_hr, y_grid_hr, z_grid_hr) @ Rmat).T
                    r_grid_hr = np.sqrt(x_grid_hr_ell**2/ar_j**2 + 
                                        y_grid_hr_ell**2/br_j**2 +
                                        z_grid_hr_ell**2/cr_j**2).reshape(x_grid_hr_ell.shape)

        
            integral_element = res**2 if self.GriddedMap.is2D else res**3
            Painting = profile(cosmo, r_grid_hr.flatten()/a_j, M_j, a_j) * integral_element
            
            mask = np.isfinite(Painting) #Find which part of map cannot be modified due to out-of-bounds errors
            mask = mask & (r_grid_hr.flatten()/a_j < R_j*self.config['epsilon_max_Offset'])
            if mask.sum() == 0: continue
                
            Painting = np.where(mask, Painting, 0) #Set those tSZ values to 0

            #Find which map pixels each subpixel corresponds to.
            #Get total "paint" per map pixel
            p_ind, inv_ind = np.unique(inds_map_hr, return_inverse = True)
            grid_map_paint = np.bincount(np.arange(len(p_ind))[inv_ind], weights = Painting)

            #Add the offsets to the new map at the right indices
            new_map[p_ind] += grid_map_paint
            
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

        for j in tqdm(range(self.HaloNDCatalog.cat.size), desc = 'Baryonifying matter', disable = not self.verbose):

            M_j = self.HaloNDCatalog.cat['M'][j]
            x_j = self.HaloNDCatalog.cat['x'][j]
            y_j = self.HaloNDCatalog.cat['y'][j]
            z_j = self.HaloNDCatalog.cat['z'][j] #THIS IS A CARTESIAN COORDINATE, NOT REDSHIFT

            a_j = 1/(1 + self.HaloNDCatalog.redshift)
            R_j = self.mass_def.get_radius(cosmo, M_j, a_j) #in physical Mpc
            
            res    = self.GriddedMap.res
            Nsize  = 2 * self.config['epsilon_max_Cutout'] * R_j / res
            Nsize  = int(Nsize // 2)*2 #Force it to be even
            
            if Nsize < 2:
                continue

            x  = np.linspace(-Nsize/2, Nsize/2, Nsize) * res
            pixel_width = Nsize//2

            if self.GriddedMap.is2D:
                x_grid, y_grid = np.meshgrid(x, x, indexing = 'xy')
                r_grid = np.sqrt(x_grid**2 + y_grid**2)

                x_inds = self.pick_indices(np.argmin(np.abs(bins - x_j)), pixel_width, self.GriddedMap.Npix)
                y_inds = self.pick_indices(np.argmin(np.abs(bins - y_j)), pixel_width, self.GriddedMap.Npix)
                
                inds = self.GriddedMap.inds_flattened[x_inds, :][:, y_inds].flatten()
                
                paint_profile  = Paint.projected
                canvas_profile = Canvas.projected 
            
            else:
                x_grid, y_grid, z_grid = np.meshgrid(x, x, x, indexing = 'xy')
                r_grid = np.sqrt(x_grid**2 + y_grid**2 + z_grid**2)

                x_inds = self.pick_indices(np.argmin(np.abs(bins - x_j)), pixel_width, self.GriddedMap.Npix)
                y_inds = self.pick_indices(np.argmin(np.abs(bins - y_j)), pixel_width, self.GriddedMap.Npix)
                z_inds = self.pick_indices(np.argmin(np.abs(bins - z_j)), pixel_width, self.GriddedMap.Npix)
                
                inds = self.GriddedMap.inds_flattened[x_inds, ...][:, y_inds, :][..., z_inds].flatten()
                
                paint_profile  = Paint.real
                canvas_profile = Canvas.real 

        
            integral_element = res**2 if self.GriddedMap.is2D else res**3

            r_array   = np.geomspace(np.min(r_grid)/a_j, np.max(r_grid)/a_j, self.Nbin_interp)
            Painting  = paint_profile(cosmo,  r_array, M_j, a_j) * integral_element
            Canvasing = canvas_profile(cosmo, r_array, M_j, a_j)
            
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