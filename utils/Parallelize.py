import joblib
import pickle
import datetime as dt
import numpy as np
from astropy.io import fits
import os

class SimpleParallel(object):
    '''
    Given a list of Runners, it runs them
    in parallel in a joblib instance
    '''
    
    def __init__(self, Runner_list, seeds = None, njobs = -1):
        
        self.Runner_list = Runner_list
        self.seeds = None
        self.njobs = njobs if njobs != -1 else joblib.externals.loky.cpu_count()
    
    
    def single_run(self, i, Runner):
        
        return i, Runner.process()
    
    
    def process(self):
        
        start = dt.datetime.now()

        print(self.single_run(0, self.Runner_list[0]))
        jobs = [joblib.delayed(self.single_run)(i, Runner) for i, Runner in enumerate(self.Runner_list)]
        
        with joblib.parallel_backend("loky"):
            outputs = joblib.Parallel(n_jobs = self.njobs, verbose=10)(jobs)

        #Order them to be the order as they were input
        ordered_outputs = [0] * len(outputs)
        for o in outputs:
            ordered_outputs[o[0]] = o[1]
            
            
        return ordered_outputs
    
    
class SplitJoinParallel(object):
    '''
    Takes a given Runner.
    Splits the halo catalog into many parts
    and then runs it separately on maps
    Adds/joins all the maps together in the end.
    '''

    def __init__(self, Runner, seeds = None, njobs = -1):
        
        self.Runner = Runner
        self.seeds = None
        self.njobs = njobs if njobs != -1 else joblib.externals.loky.cpu_count()
        
        self.Runner_list = self.split_run(self.Runner)
        
        
    def split_run(self, Runner):
        
        HaloCat  = Runner.HaloCatalog
        Shell    = Runner.LightconeShell
        config   = Runner.config.copy()
        cosmo    = Runner.cosmo
        model    = Runner.model
        mass_def = Runner.mass_def
        
        OutPath  = config.pop('OutPath', None)
        
        #Now split
        
        catalog   = HaloCat.cat
        Nsplits   = self.njobs
        Ntotal    = len(catalog)
        Npersplit = int(np.ceil(Ntotal/Nsplits))
        
        empty_shell = type(Shell)(map = np.zeros_like(Shell.map), cosmo = cosmo)
        
        Runner_list = []
        for i in range(Nsplits):
            
            start = i*Npersplit
            end   = (i + 1)*Npersplit
            
            New_HaloCatalog = type(HaloCat)(ra = catalog['ra'][start:end], dec = catalog['dec'][start:end],
                                            M = catalog['M'][start:end],   z = catalog['z'][start:end], cosmo = cosmo)
            
            #Create a new Runner for just a subset of catalog. Has same model, map size etc.
            #Force verbose to be off as we don't want outputs for each subrun of parallel process.
            New_Runner = type(Runner)(New_HaloCatalog, empty_shell, config, model, mass_def, verbose = False)
            
            Runner_list.append(New_Runner)
        
        return Runner_list
        
    def single_run(self, Runner):        
        
        return Runner.process()
    
    
    def process(self):
        
        start = dt.datetime.now()

        jobs = [joblib.delayed(self.single_run)(Runner) for Runner in self.Runner_list]

        with joblib.parallel_backend("loky"):
            outputs = joblib.Parallel(n_jobs = self.njobs, verbose=10)(jobs)

        map_out = np.sum(outputs, axis = 0)
        
        #Use the output method from the original Runner that was passed in
        self.Runner.output(map_out)
            
        return 
    
    