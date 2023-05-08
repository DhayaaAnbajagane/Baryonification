import joblib
import pickle

class ParallelRunner(object):
    
    def __init__(self, OutputMap, Runner_list, seeds = None):
        
        self.OutputMap = OutputMap
        self.Runner_list = []
        self.seeds = None
        
    
    def single_runs(self):
        
        
        