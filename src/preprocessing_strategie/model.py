import pandas as pd

class MyModel():
    def __init__(self, x:pd.DataFrame, y:pd.Series,paramrs:dict,model:object):
        self.x = x
        self.y = y
        self.paramrs = paramrs
        self.model = model
    
    def train(self):
        self.model.set_params(**self.paramrs)
        self.model.fit(self.x, self.y)
        return self.model