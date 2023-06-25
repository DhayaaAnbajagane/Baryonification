
import numpy as np
import pyccl as ccl

from scipy import interpolate
from astropy.cosmology import z_at_value, FlatLambdaCDM, FlatwCDM
from astropy import units as u


import sys
sys.path.insert(0, '/home/dhayaa/Desktop/Quijote/')
from Baryonification.Runners.HealpixRunner import BaryonifyShell, PaintThermalSZShell
from Baryonification.utils import HaloLightConeCatalog, LightconeShell, TabulatedProfile, SplitJoinParallel, SimpleParallel
from Baryonification.Profiles import DarkMatterOnly, DarkMatterBaryon, Baryonification2D, Pressure, ThermalSZ

def get_param(log_file_name, parameter_name):
    '''
    Function taken from Lorne's repo:
    https://github.com/LorneWhiteway/lfi_project/blob/master/scripts/utility.py
    '''
    with open(log_file_name, "r") as f:
        for line in f:
            if parameter_name in line:
                tokenised_string = line.split(" ")
                for (token, i) in zip(tokenised_string, range(len(tokenised_string))):
                    if token == parameter_name or token == (parameter_name + ":"):
                        if i < len(tokenised_string) - 1:
                            return float(tokenised_string[i + 1])

if __name__ == '__main__':

    import sys, os

    sys.path.insert(0, "/home/dhayaa/Nbody_simulations/lfi_project/scripts/")
    sys.path.insert(0, "/home/dhayaa/Nbody_simulations/BornRaytrace")

    import numpy as np
    import healpy as hp
    from utility import one_healpix_map_from_basefilename
    #from utils import build_z_values_file, get_param
    import glob

    from astropy.table import Table
    from astropy.io import fits

    import argparse

    my_parser = argparse.ArgumentParser()

    #Metaparams
    my_parser.add_argument('--InputDir',  action='store', type = str, required = True)
    my_parser.add_argument('--OutputDir', action='store', type = str, required = True)
    my_parser.add_argument('--NSIDE',     action='store', type = int, default = 1024)
    my_parser.add_argument('--Nsize',     action='store', type = int, default = 200)
    my_parser.add_argument('--Name',      action='store', type = str, default = '')

    my_parser.add_argument('--epsilon_max_Cutout', action='store', type = float, default = 5)
    my_parser.add_argument('--epsilon_max_Offset', action='store', type = float, default = 5)
    my_parser.add_argument('--pixel_scale_factor', action='store', type = float, default = 0.5)
    
    #Schneider Baryonification parameters as described in 1810.08629
    my_parser.add_argument('--epsilon',   action='store', type = float, default = 4.0)
    my_parser.add_argument('--theta_ej',  action='store', type = float, default = 4.0)
    my_parser.add_argument('--theta_co',  action='store', type = float, default = 0.1)
    my_parser.add_argument('--M_c',       action='store', type = float, default = 2e14) #in Msun
    my_parser.add_argument('--mu',        action='store', type = float, default = 0.4)
    my_parser.add_argument('--eta_star',  action='store', type = float, default = 0.3)
    my_parser.add_argument('--eta_cga',   action='store', type = float, default = 0.6)
    my_parser.add_argument('--beta_star', action='store', type = float, default = -1.5)
    my_parser.add_argument('--beta_cga',  action='store', type = float, default = -1.5)
    my_parser.add_argument('--A',         action='store', type = float, default = 0.09)
    my_parser.add_argument('--M1',        action='store', type = float, default = 3e11) #in Msun
    my_parser.add_argument('--epsilon_h', action='store', type = float, default = 0.015)
    my_parser.add_argument('--a',         action='store', type = float, default = 0.3)
    my_parser.add_argument('--n',         action='store', type = float, default = 2.0)
    my_parser.add_argument('--p',         action='store', type = float, default = 0.3)
    my_parser.add_argument('--q',         action='store', type = float, default = 0.707)

    args = vars(my_parser.parse_args())

    #Print args for debugging state
    print('-------INPUT PARAMS----------')
    for p in args.keys():
        print('%s : %s'%(p.upper(), args[p]))
    print('-----------------------------')
    print('-----------------------------')

    MY_FILL_VAL = np.NaN

    log         = np.genfromtxt(args['InputDir'] +  '/Density_shell.log')
    z_bin_edges = log.T[1]

    #Hardcoded for now. Maybe change in future.
    Nparts  = 512**3

    # cosmo code
    om = get_param(args['InputDir'] +  '/Density_shell.log', 'dOmega0') # 0.3175
    s8 = get_param(args['InputDir'] +  '/Density_shell.log', 'dSigma8') # 0.834
    h  = get_param(args['InputDir'] +  '/Density_shell.log', 'h') # 0.6711
    w0 = get_param(args['InputDir'] +  '/Density_shell.log', 'w0') # -1
    ns = get_param(args['InputDir'] +  '/Density_shell.log', 'dSpectral') #0.96

    cosmo_fiducial = FlatwCDM(H0 = h * 100. * u.km / u.s / u.Mpc, Om0 = om, w0 = w0)

    #Load existing halo lightcone catalog and convert Msun/h to Msun, Mpc/h to Mpc
    Halos      = np.load(args['InputDir'] +  '/../postprocessed/All_halo_catalogs.npy')
    Halos      = Halos[np.argsort(Halos['M'])[::-1]]
    Halos['M'] = 10**Halos['M']/h
    Halos['R'] = Halos['R']/h #in comoving Mpc

    sim_filenames = glob.glob(args['InputDir'] + '/Density_shell*fits.fz')
    sim_filenames = sorted(sim_filenames)

    healpix_inds = np.arange(hp.nside2npix(args['NSIDE']), dtype = int)

    cosmo = ccl.Cosmology(Omega_c = om - 0.049, Omega_b = 0.049, h = h, sigma8 = s8, n_s = ns, matter_power_spectrum='linear')
    cosmo.compute_sigma()
    
    cosmo_dict = {'Omega_m' : om, 'Omega_b' : 0.049, 'h' : h, 'sigma8' : s8, 'n_s' : ns, 'w0' : w0}

    DMO = DarkMatterOnly(epsilon = args['epsilon'], a = args['a'], n = args['n'],
                      theta_ej = args['theta_ej'], theta_co = args['theta_co'],
                      M_c = args['M_c'], mu = args['mu'],
                      A = args['A'], M1 = args['M1'], epsilon_h = args['epsilon_h'],
                      eta_star = args['eta_star'], eta_cga = args['eta_cga'],
                      beta_star = args['beta_star'], beta_cga = args['beta_cga'],
                      q = args['q'], p = args['p'], xi_mm = None, R_range = [1e-5, 40])

    DMB = DarkMatterBaryon(epsilon = args['epsilon'], a = args['a'], n = args['n'],
                      theta_ej = args['theta_ej'], theta_co = args['theta_co'],
                      M_c = args['M_c'], mu = args['mu'],
                      A = args['A'], M1 = args['M1'], epsilon_h = args['epsilon_h'],
                      eta_star = args['eta_star'], eta_cga = args['eta_cga'],
                      beta_star = args['beta_star'], beta_cga = args['beta_cga'],
                      q = args['q'], p = args['p'], xi_mm = None, R_range = [1e-5, 40])
    
    Baryons = Baryonification2D(DMO, DMB, ccl_cosmo = cosmo, epsilon_max = args['epsilon_max_Offset'])
    Baryons.setup_interpolator()

    Runner_list = []
    for i in range(len(sim_filenames)):

#         if i % 10 != 0: continue #Temporary condition for testing
#         if (i <= 10) | (i >= 90): continue #Temporary condition for testing
        
#         print('----------------------')
#         print('IN SHELL %d'%i)
#         print('----------------------')
        
        
        a   = 1/(1 + z_bin_edges[i])
        map1 = fits.open(sim_filenames[i])[1].data * 1.0
        map1 = LightconeShell(map = map1, cosmo = cosmo_dict)
        
        halo_ind = np.where((Halos['z'] < z_bin_edges[i]) & (Halos['z'] > z_bin_edges[i + 1]))[0]
        catalog1 = HaloLightConeCatalog(ra = Halos['ra'][halo_ind], dec = Halos['dec'][halo_ind],
                               M = Halos['M'][halo_ind], z = Halos['z'][halo_ind], cosmo = cosmo_dict)
        catalog2 = HaloLightConeCatalog(ra = Halos['ra'][:], dec = Halos['dec'][:],
                               M = Halos['M'][:], z = Halos['z'][:], cosmo = cosmo_dict)
        
        Name = 'New_Baryonified_Density_shell' + ('_%s'%args['Name'] if args['Name'] != '' else '')
        path_ = args['OutputDir'] + '/' +  os.path.basename(sim_filenames[i]).replace('Density_shell', Name).replace('.fz', '')

        args['OutPath'] = path_

        Runner_list.append(BaryonifyShell(catalog1, map1, args, model = Baryons, verbose = False))
    
    Runner = SimpleParallel(Runner_list)
    Runner.process()
        
        
    Name = 'New_Baryonified_tSZ_shell' + ('_%s'%args['Name'] if args['Name'] != '' else '')
    path_ = args['OutputDir'] + '/' +  os.path.basename(sim_filenames[i]).replace('Density_shell', Name).replace('.fz', '')

    args['OutPath'] = path_
    Baryons = Pressure(epsilon = args['epsilon'], a = args['a'], n = args['n'],
                       theta_ej = args['theta_ej'], theta_co = args['theta_co'],
                       M_c = args['M_c'], mu = args['mu'],
                       A = args['A'], M1 = args['M1'], epsilon_h = args['epsilon_h'],
                       eta_star = args['eta_star'], eta_cga = args['eta_cga'],
                       beta_star = args['eta_star'], beta_cga = args['eta_cga'],
                       q = args['q'], p = args['p'], xi_mm = None, R_range = [1e-5, 40])
    Baryons = ThermalSZ(Baryons, epsilon_max = args['epsilon_max_Offset'])
    Baryons = TabulatedProfile(Baryons, cosmo)
    Baryons.setup_interpolator(N_samples_z = 30, N_samples_Mass = 30)

    Runner = PaintThermalSZShell(catalog2, map1, args, model = Baryons)
    Runner = SplitJoinParallel(Runner)

    Runner.process()
