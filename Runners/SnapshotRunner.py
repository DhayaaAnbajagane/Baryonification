import numpy as np
import pyccl as ccl
from scipy.spatial import KDTree 
from tqdm import tqdm

MY_FILL_VAL = np.NaN


class DefaultRunnerSnapshot(object):
    '''
    A class that contains relevant utils for input/output
    '''
    
    def __init__(self, HaloNDCatalog, ParticleSnapshot, config, model,
                 mass_def = ccl.halos.massdef.MassDef(200, 'critical'), verbose = True):

        self.HaloNDCatalog    = HaloNDCatalog
        self.ParticleSnapshot = ParticleSnapshot
        self.cosmo = HaloNDCatalog.cosmology
        self.model = model
        
        self.mass_def = mass_def
        self.verbose  = verbose
        
        self.config   = self.set_config(config)

        if ParticleSnapshot.is2D:
            coords = np.vstack([ParticleSnapshot.cat['x'], ParticleSnapshot.cat['y']]).T
        else:
            coords = np.vstack([ParticleSnapshot.cat['x'], ParticleSnapshot.cat['y'], ParticleSnapshot.cat['z']]).T
                               
        self.tree = KDTree(coords, boxsize = ParticleSnapshot.L)


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
                
                
    def compute_distance(self, *args):
        
        L = self.ParticleSnapshot.L
        d = 0
        
        for dx in args:
            
            dx = np.where(dx > L/2,  dx - L, dx)
            dx = np.where(dx < -L/2, dx + L, dx)
            
            d += dx**2
            
        return np.sqrt(d)
    
    
    def enforce_periodicity(self, dx):
        
        L = self.ParticleSnapshot.L
        d = 0
        
        dx = np.where(dx > L/2,  dx - L, dx)
        dx = np.where(dx < -L/2, dx + L, dx)
            
        return dx
    
    

class BaryonifySnapshot(DefaultRunnerSnapshot):

    def process(self):

        cosmo = ccl.Cosmology(Omega_c = self.cosmo['Omega_m'] - self.cosmo['Omega_b'],
                              Omega_b = self.cosmo['Omega_b'], h = self.cosmo['h'],
                              sigma8  = self.cosmo['sigma8'],  n_s = self.cosmo['n_s'],
                              matter_power_spectrum = 'linear')
        cosmo.compute_sigma()

        L = self.ParticleSnapshot.L
        is2D        = self.ParticleSnapshot.is2D
        tot_offsets = np.zeros([len(self.ParticleSnapshot.cat), 2 if is2D else 3])
        
        for j in tqdm(range(self.HaloNDCatalog.cat.size), desc = 'Baryonifying matter', disable = not self.verbose):

            M_j = self.HaloNDCatalog.cat['M'][j]
            x_j = self.HaloNDCatalog.cat['x'][j]
            y_j = self.HaloNDCatalog.cat['y'][j]
            z_j = self.HaloNDCatalog.cat['z'][j] #THIS IS A CARTESIAN COORDINATE, NOT REDSHIFT
            
            a_j = 1/(1 + self.HaloNDCatalog.redshift)
            R_j = self.mass_def.get_radius(cosmo, M_j, a_j) #in physical Mpc
            R_q = self.config['epsilon_max_Cutout'] * R_j/a_j #The radius for querying points, in comoving coords
            R_q = np.clip(R_q, 0, L/2) #Can't query distances more than half box-size.
            
            if is2D:
                
                inds = self.tree.query_ball_point([x_j, y_j], R_q)
                dx   = self.ParticleSnapshot.cat['x'][inds] - x_j
                dy   = self.ParticleSnapshot.cat['y'][inds] - y_j
                d    = self.compute_distance(dx, dy)

                x_hat = self.enforce_periodicity(self.ParticleSnapshot.cat['x'][inds] - x_j)/d
                y_hat = self.enforce_periodicity(self.ParticleSnapshot.cat['y'][inds] - y_j)/d

                #Compute the displacement needed
                offset = self.model.displacement(d, M_j, a_j) * a_j
                offset = np.where(np.isfinite(offset), offset, 0)
                tot_offsets[inds] += np.vstack([offset*x_hat, offset*y_hat]).T
                
            
            else:
                inds = self.tree.query_ball_point([x_j, y_j, z_j], R_q)
                dx   = self.ParticleSnapshot.cat['x'][inds] - x_j
                dy   = self.ParticleSnapshot.cat['y'][inds] - y_j
                dz   = self.ParticleSnapshot.cat['z'][inds] - y_j
                d    = self.compute_distance(dx, dy)

                x_hat = self.enforce_periodicity(self.ParticleSnapshot.cat['x'][inds] - x_j)/d
                y_hat = self.enforce_periodicity(self.ParticleSnapshot.cat['y'][inds] - y_j)/d
                z_hat = self.enforce_periodicity(self.ParticleSnapshot.cat['z'][inds] - z_j)/d

                #Compute the displacement needed
                offset = self.model.displacement(d, M_j, a_j) * a_j
                offset = np.where(np.isfinite(offset), offset, 0)
                tot_offsets[inds] += np.vstack([offset*x_hat, offset*y_hat, offset*z_hat]).T
                
            
        new_cat = self.ParticleSnapshot.cat.copy()
        
        new_cat['x'] += tot_offsets[:, 0]
        new_cat['y'] += tot_offsets[:, 1]
        
        if not is2D: new_cat['z'] += tot_offsets[:, 2]
            
        for i in ['x', 'y'] + ([] if self.ParticleSnapshot.is2D else ['z']):
            
            new_cat[i]  = np.where(new_cat[i] > L, new_cat[i] - L, new_cat[i])
            new_cat[i]  = np.where(new_cat[i] < 0, new_cat[i] + L, new_cat[i])

        self.output(new_cat)

        return new_cat