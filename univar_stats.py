import numpy as np
import pandas as pd
from scipy import stats


class Stats:
    @staticmethod
    def get_stats(X, X_scaled):
        print("General col min: " + str(list(X.min(axis=0))) + ", \nscaled col min: " + str(list(X_scaled.min(axis=0))))
        print("\n")

        print("General col max: " + str(list(X.max(axis=0))) + ", \nscaled col max: " + str(list(X_scaled.max(axis=0))))
        print("\n")

        print("General col mean: " + str(list(X.mean(axis=0, numeric_only=True))) + ", \nscaled col mean: " +
              str(list(X_scaled.mean(axis=0, numeric_only=True))))
        print("\n")

        print("General col var: " + str(list(X.var(axis=0, numeric_only=True))) + ", \nscaled col var: " +
              str(list(X_scaled.var(axis=0, numeric_only=True))))
        print("\n")

        print("General col median: " + str(list(X.median(axis=0, numeric_only=True))) + ", \nscaled col median: " +
              str(list(X_scaled.median(axis=0, numeric_only=True))))
        print("\n")

        print("General col mode: " + str(list(X.mode(axis=0))) + ", \nscaled col mode: " + str(list(X_scaled.mode(axis=0))))
        print("\n")

        print("General col q_0.25: " + str(list(X.quantile(0.25, axis=0, numeric_only=True))) +
              ", \nscaled col q_0.25: " + str(list(X_scaled.quantile(0.25, axis=0, numeric_only=True))))
        print("\n")

        print("General col q_0.75: " + str(list(X.quantile(0.75, axis=0, numeric_only=True))) +
              ", \nscaled col q_0.75: " + str(list(X_scaled.quantile(0.75, axis=0, numeric_only=True))))
        print("\n")


    @staticmethod
    def check_correlations(X, X_scaled):
        X_cor = X.corr(method="pearson")
        X_scaled_cor = X_scaled.corr(method="pearson")

        if not (X_cor - 0.2 <= X_scaled_cor <= X_cor + 0.2).all():
            print("Correlation differs by more than 20%:")
            print(not (X_cor - 0.2 <= X_scaled_cor <= X_cor + 0.2))
