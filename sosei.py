
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

class Sosei:
    def __init__(self,clf,df_x,df_y):
        self.clf = clf
        self.df_x = df_x
        self.df_y = df_y
        self.X = df_x.as_matrix()
        self.y = df_y.as_matrix()
        self.coef_df = None

    def make_coef_df(self):
        #clf = self.clf
        self.clf.fit(self.X, self.y)
        coef_df = pd.DataFrame({"Name":self.df_x.columns,"coefficients":self.clf.coef_})
        self.coef_df = coef_df

    def mae(self):
        y_pre = self.clf.predict(self.X)
        return mean_absolute_error(self.y, y_pre)

    def save_csv(self,filename, tenchi=False):
        if tenchi is True:
            self.coef_df.T.to_csv(filename)
        else:
            self.coef_df.to_csv(filename)

class DfStandscaler():

    def __init__(self,df):
        self.df = df
        self.df_scaled = None

    def fit_transform(self):
        scaler = StandardScaler()
        self.df_scaled = pd.DataFrame(scaler.fit_transform(self.df), columns=self.df.columns)

