import joblib
import pickle

class ParallelRunner(object):
    
    def __init__(self, Runner_list, seeds = None):
        
        self.Runner_list = []
        self.seeds = None
    
    def single_run(self, Runner):
        
        se;f