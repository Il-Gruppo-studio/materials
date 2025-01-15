import numpy as np

np.random.seed(123)

class RegressionMetrics:
    def __init__(self, model):
        self.model = model

    def compute_performance(self, X,y):
        preds = self.model.predict(X).squeeze()

        mae = (self.mean_absolute_error(preds, y))
        mape = self.mean_absolute_percentage_error(preds, y)
        mpe = self.mean_percentage_error(preds, y)
        mse = self.mean_squared_error(preds, y)
        rmse = self.root_mean_squared_error(preds, y)
        r2 = self.r_2(preds, y)
        return {'mae':mae,'mape': mape,'mpe': mpe,
                'mse': mse,'rmse': rmse,'r2': r2}

    def mean_absolute_error (self, preds, y):
        output_errors= np.abs(preds-y)
        return np.average(output_errors)

    def mean_squared_error (self, preds, y):
        output_errors= (preds-y)**2
        return np.average(output_errors)

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

