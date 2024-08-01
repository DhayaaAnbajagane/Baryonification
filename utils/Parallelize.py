import joblib
import numpy as np

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
        
        with joblib.parallel_backend("loky"):
            jobs    = [joblib.delayed(self.single_run)(i, Runner) for i, Runner in enumerate(self.Runner_list)]
            outputs = joblib.Parallel(n_jobs = self.njobs, verbose=10)(jobs)

        #Sort them so they are in the same order as they were input
        ordered_outputs = [0] * len(outputs)
        for o in outputs: ordered_outputs[o[0]] = o[1]
            
            
        return ordered_outputs
    
    
class SplitJoinParallel(object):
    '''
    Takes a given Runner.
    Splits the halo catalog into many parts
    and then runs it separately on maps
    Adds/joins all the maps together in the end.
    '''

    def __init__(self, Runner, njobs = -1):
        
        self.Runner = Runner
        self.seed   = 42 #We only use seed for a single thing, so I've hardcoded it here
        self.njobs  = njobs if njobs != -1 else joblib.externals.loky.cpu_count()
        
        self.Runner_list = self.split_run(self.Runner)
        
        
    def split_run(self, Runner):
        
        HaloCat  = Runner.HaloLightConeCatalog
        Shell    = Runner.LightconeShell
        cosmo    = Runner.cosmo
        model    = Runner.model
        mass_def = Runner.mass_def
        eps_max  = Runner.epsilon_max
        ellip    = Runner.use_ellipticity
        
        #Now split
        
        catalog   = HaloCat.cat
        Nsplits   = self.njobs
        Ntotal    = len(catalog)
        Npersplit = int(np.ceil(Ntotal/Nsplits))

        #Randomize catalog ordering. This helps optimize the parallelization. Else if 
        #low redshift halos are all the start, then handful of processes will be overburdened 
        #while the rest are just sitting idle.
        HaloCat     = HaloCat[np.random.default_rng(self.seed).choice(Ntotal, size = Ntotal, replace = False)] 
        
        empty_shell = type(Shell)(map = np.zeros_like(Shell.map), cosmo = cosmo)
        
        Runner_list = []
        for i in range(Nsplits):
            
            start = i*Npersplit
            end   = (i + 1)*Npersplit
            
            #Halocatalog object is sliceable like a regular numpy array so can easily
            #split the halos up but keep the same data structure
            New_HaloCatalog = HaloCat[start:end]
            
            #Create a new Runner for just a subset of catalog. Has same model, map size etc.
            #Force verbose to be off as we don't want outputs for each subrun of parallel process.
            New_Runner = type(Runner)(New_HaloCatalog, empty_shell, eps_max, model, ellip, mass_def, verbose = False)
            
            Runner_list.append(New_Runner)
        
        return Runner_list
        
    def single_run(self, Runner):        
        
        return Runner.process()
    
    
    def process(self):
        
        with joblib.parallel_backend("loky"):
            jobs    = [joblib.delayed(self.single_run)(Runner) for Runner in self.Runner_list]
            outputs = joblib.Parallel(n_jobs = self.njobs, verbose=10)(jobs)

        #Sum the contributions from invidual runs. 
        #Contributions from each halo can be linearly added so this is fine
        map_out = np.sum(outputs, axis = 0)
        
        return map_out
    
    