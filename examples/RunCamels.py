
import numpy as np
from scipy import stats, optimize
import sys, os, joblib, glob, h5py

sys.path.insert(0, '/home/dhayaa/Desktop/Quijote/')
sys.path.insert(0, '/home/dhayaa/Desktop/Quijote/Sim2Stat')

from Baryonification.Runners.Map2DRunner import BaryonifyGrid, PaintProfilesGrid
from Baryonification.utils import TabulatedProfile
from Baryonification.Profiles import model_params, Gas, DarkMatterOnly, DarkMatterBaryon, Baryonification2D, Baryonification3D, Pressure, ThermalSZ

from StatMaker import FilterMap, ComputeStats

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import cross_val_score, KFold

from smt.surrogate_models import KPLS


class LHCubeMapMaker:
    def __init__(self, HaloNDCatalog, GriddedMap, ccl_cosmo, param_names, param_bounds, seed, Nsamples, config):


        self.HaloNDCatalog  = HaloNDCatalog
        self.GriddedMap     = GriddedMap
        self.param_names    = param_names
        self.param_bounds   = param_bounds
        self.seed           = seed
        self.Nsamples       = Nsamples
        self.config         = config
        self.cosmo          = ccl_cosmo

        self.samples = self.sample_params()

        np.save(config['OutDir'] + '/input_params.npy', self.samples)


    def sample_params(self):

        sampler = stats.qmc.Sobol(d = len(self.param_names), seed = self.seed)
        samples = sampler.random(self.Nsamples)

        p_start = self.param_bounds[:, 0]
        p_width = self.param_bounds[:, 1] - self.param_bounds[:, 0]

        samples = p_start + samples * p_width

        return samples


    def run(self):

        with joblib.parallel_backend("loky"):
            jobs    = [joblib.delayed(self.single_run)(i) for i in range(self.Nsamples)]
            outputs = joblib.Parallel(n_jobs = -1, verbose=10)(jobs)

        assert np.sum(outputs) == self.Nsamples, "Only %d of %d processes returned success" % (np.sum(outputs), len(outputs))

    
    def single_run(self, i):

        config_here = self.config.copy()
            
        for p in self.param_names:
            config_here[p] = self.samples[i, p]

        prof_params  = {k:config_here[k] for k in model_params}
        baryo_params = {k:config_here[k] for k in ['epsilon_max', 'use_concentration']}
        
        DMO = DarkMatterOnly(**prof_params,   xi_mm = None, R_range = [1e-3, 40])
        DMB = DarkMatterBaryon(**prof_params, xi_mm = None, R_range = [1e-3, 40])
        PRS = Pressure(nonthermal_model = None, **prof_params)
        GAS = Gas(**prof_params)
        
        if len(self.GriddedMap.map.shape) == 2:
            model = Baryonification2D(DMO, DMB, self.cosmo, **baryo_params)
        elif len(self.GriddedMap.map.shape) == 2:
            model = Baryonification3D(DMO, DMB, self.cosmo, **baryo_params)
        else:
            raise ValueError("GriddedMap is not 2D or 3D.")
        
        #Setup all interpolation steps
        model.setup_interpolator(verbose = False, z_min = 0, z_max = 2, z_linear_sampling = True)
        PRS = TabulatedProfile(PRS, self.cosmo).setup_interpolator(verbose = False, z_linear_sampling = True, z_min = 0, z_max = 2)
        GAS = TabulatedProfile(GAS, self.cosmo).setup_interpolator(verbose = False, z_linear_sampling = True, z_min = 0, z_max = 2)

        config_here['OutPath'] = config_here['OutDir'] + '/rhoDM_Map%d.npy' %i
        BaryonifyGrid(self.HaloNDCatalog, self.GriddedMap, config_here, model, verbose = False).process()
        
        config_here['OutPath'] = config_here['OutDir'] + '/Pgas_Map%d.npy' %i
        PaintProfilesGrid(self.HaloNDCatalog, self.GriddedMap, config_here, model = PRS).process()
        
        config_here['OutPath'] = config_here['OutDir'] + '/rhogas_Map%d.npy' %i
        PaintProfilesGrid(self.HaloNDCatalog, self.GriddedMap, config_here, model = GAS).process()

        return True


class StatRunner:
    def __init__(self, mapdir, config):

        self.outdir = mapdir
        self.scales = np.geomspace()
        self.config = config

        self.Nmaps = len(glob.glob(mapdir + '/rhoDM_Map*.npy'))

    def run(self):

        with joblib.parallel_backend("loky"):
            jobs    = [joblib.delayed(self.single_run)(i) for i in range(self.Nmaps)]
            outputs = joblib.Parallel(n_jobs = -1, verbose=10)(jobs)

        Pk  = [0] * self.Nmaps
        Bk  = [0] * self.Nmaps
        WPH = [0] * self.Nmaps
        Mom = [0] * self.Nmaps
        CDF = [0] * self.Nmaps


        for o in outputs:
            Pk[o[0]]  = o[1]['Pk']
            Bk[o[0]]  = o[1]['Bk']
            WPH[o[0]] = o[1]['WPH']
            Mom[o[0]] = 0[1]['Mom']
            CDF[o[0]] = 0[1]['CDF']


        with h5py.File('/%s/MapStats.hdf5' % self.outdir, 'w') as f:

            f.create_dataset('Pk',   data = np.array(Pk))
            f.create_dataset('Bk',   data = np.array(Bk))
            f.create_dataset('WPH',  data = np.array(WPH))
            f.create_dataset('Mom',  data = np.array(Mom))
            f.create_dataset('CDFs', data = np.array(CDF))

            f.create_dataset('scales', data = self.scales)

        
    def single_run(self, i):

        Maps   = np.array([self.outdir + '/%s_Map%d.npy' % (k, i) for k in ['rhoDM', 'Pgas', 'rhogas']])
        Output = self.process_maps(Maps)

        return i, Output
    

    def process_maps(self, Maps):

        Map_filtered = FilterMap.FlatSkyTopHat(Maps, config = self.config)

        MomRunner = ComputeStats.Moments(self.config)
        CDFRunner = ComputeStats.CDF(self.config)
        PkRunner  = ComputeStats.FlatSkyPowerspectrum(self.config)
        BkRunner  = ComputeStats.FlatSkyBispectrum(self.config)
        WPHRunner = ComputeStats.FlatSkyWPH(self.config)


        WPH = WPHRunner.compute(Maps)
        Pk  = PkRunner.compute(Maps)
        Bk  = BkRunner.compute(Maps)


        Mom = [0] * self.scales.size
        CDF = [0] * self.scales.size

        pixel_scale   = 25/0.6771 / len(Maps[0])
        for i in range(self.scales.size):

            Smoothed_map = Map_filtered.filter(scale = self.scales[i]/pixel_scale)

            Mom.append(MomRunner.compute(Smoothed_map))
            CDF.append(CDFRunner.compute(Smoothed_map))

        Mom = np.array(Mom)
        CDF = np.array(CDF)

        Mom = np.moveaxis(Mom, 0, -1)
        CDF = np.moveaxis(CDF, 0, -1)


        Output = {}

        Output['Pk']   = Pk
        Output['Bk']   = Bk
        Output['WPH']  = WPH
        Output['Mom']  = Mom
        Output['CDF']  = CDF

        return Output            


class BuildEmulator:
    def __init__(self, config, seed, test_train_split):

        self.samples = np.load(config['OutPath'] + '/input_params.npy')
        self.config  = config
        self.seed    = seed

        N = len(self.samples)
        self.train_inds = np.random.choice(np.arange(N), replace = False, size = int(test_train_split * N))
        self.test_inds  = np.delete(np.arange(len(N)), self.train_inds)

    def run(self, statistic):

        X = self.samples

        with h5py.File(self.config['OutDir'] + '/MapStats.hdf5', 'r') as f:
            y = f[statistic][:]
        
        sm = KPLS(theta0=[1e-2])
        sm.set_training_values(X[self.train_inds], y[self.train_inds])
        sm.train()

        test_prediction = sm.predict_variances(X[self.test_inds])
        test_error      = (test_prediction - y[self.test_inds])/y[self.test_inds]

        np.mean(test_error, axis = 0)
        np.median(test_error, axis = 0)
        np.std(test_error, axis = 0)

        self.model = sm
        return sm
    

    def fit(self, data, guess):

        assert len(guess) == self.samples.shape[1], "The guess array doesn't have N_params = %d" % self.samples.shape[1]
        
        def func(self, xdata, **args):
            return self.sm(**args)
        
        p = optimize.curve_fit(func, np.NaN, data, p0 = guess)

        return p


if __name__ == '__main__':

    import argparse

    my_parser = argparse.ArgumentParser()

    #Metaparams
    my_parser.add_argument('--MapPath',   action='store', type = str, required = True)
    my_parser.add_argument('--ParamPath', action='store', type = str, required = True)
    my_parser.add_argument('--OutDir',    action='store', type = str, required = True)

    my_parser.add_argument('--epsilon_max_Cutout', action='store', type = float, default = 5)
    my_parser.add_argument('--epsilon_max_Offset', action='store', type = float, default = 5)
    my_parser.add_argument('--pixel_scale_factor', action='store', type = float, default = 0.5)
    
    #Schneider Baryonification parameters as described in 1810.08629
    my_parser.add_argument('--epsilon',     action='store', type = float, default = 4.0)
    my_parser.add_argument('--theta_ej',    action='store', type = float, default = 4.0)
    my_parser.add_argument('--theta_co',    action='store', type = float, default = 0.1)
    my_parser.add_argument('--M_c',         action='store', type = float, default = 2e14) #in Msun
    my_parser.add_argument('--mu',          action='store', type = float, default = 1.0)
    my_parser.add_argument('--gamma',       action='store', type = float, default = 2.5)
    my_parser.add_argument('--delta',       action='store', type = float, default = 7.0)
    my_parser.add_argument('--eta',         action='store', type = float, default = 0.2)
    my_parser.add_argument('--eta_delta',   action='store', type = float, default = 0.1)
    my_parser.add_argument('--beta',        action='store', type = float, default = -1.5)
    my_parser.add_argument('--beta_delta',  action='store', type = float, default = 0)
    my_parser.add_argument('--A',           action='store', type = float, default = 0.055)
    my_parser.add_argument('--M1',          action='store', type = float, default = 3e11) #in Msun
    my_parser.add_argument('--epsilon_h',   action='store', type = float, default = 0.015)
    my_parser.add_argument('--a',           action='store', type = float, default = 0.3)
    my_parser.add_argument('--n',           action='store', type = float, default = 2.0)
    my_parser.add_argument('--p',           action='store', type = float, default = 0.3)
    my_parser.add_argument('--q',           action='store', type = float, default = 0.707)

    args = vars(my_parser.parse_args())

    #Print args for debugging state
    print('-------INPUT PARAMS----------')
    for p in args.keys():
        print('%s : %s'%(p.upper(), args[p]))
    print('-----------------------------')
    print('-----------------------------')



