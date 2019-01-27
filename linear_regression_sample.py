# -*- encoding:utf-8 -*-

import pandas as pd
from sklearn import linear_model
from sosei import Sosei



def read_data():

    df = pd.read_csv("./data/winequality-red.csv", sep=";")
    #df.drop("quality", axis=1)

    x_df = df.drop("quality",axis=1)
    y_df = df["quality"]

    return x_df,y_df


if __name__=="__main__":
    
    x_df, y_df = read_data()

    clf = linear_model.LinearRegression(fit_intercept=True, normalize=False,copy_X=True, n_jobs=1)
    C = Sosei(clf,x_df,y_df)
    C.make_coef_df()
    C.save_csv("test.csv",tenchi=True)

