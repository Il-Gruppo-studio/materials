import numpy as np
import pandas as pd

class ValutazioniStatistiche:
    def __init__(self,model):
        self.model = model

    def stats(self,X,Y):
        pred = self.model.predict(X)

        mae = (self.mean_absolute_error(pred,Y))
        mape = self.mean_absolute_percentage_error(pred,Y)
        mpe = self.mean_percentage_error(pred, Y)
        mse = self.mean_squared_error(pred, Y)
        rmse = self.root_mean_squared_error(pred,Y)
        r2 = self.r_2(pred, Y)
        return {'mae': mae, 'mape': mape, 'mpe': mpe,
                'mse': mse, 'rmse': rmse, 'r2': r2}

    def mean_absolute_error(self,pred,Y):
        output_error = np.abs(pred-Y)
        return np.average(output_error)


    def root_mean_squared_error (self, preds, y):
        return np.sqrt(self.mean_squared_error(preds, y))

    def mean_absolute_percentage_error (self, preds, y):
        output_errors = np.abs((preds-y)/y)
        return np.average(output_errors)*100

    def mean_percentage_error (self, preds, y):
        output_errors = (preds-y)/y
        return np.average(output_errors)*100

    def r_2(self, preds, y):
        sst = np.sum ((y-y.mean())**2)
        ssr = np.sum((preds-y)**2)
        r2 = 1-(ssr/sst)
        return r2