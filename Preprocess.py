#Preprocess step include
# Train test split (20%)
# Standard Scaling
import this

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Preprocess:
    def __init__(self,data):
        self.data = data


    def CheckPCA(self,X):

        pca = PCA()  # Default n_components = min(n_samples, n_features)
        X_train_pc = pca.fit_transform(X)
        exp_var_pca = pca.explained_variance_ratio_

        # Cumulative sum of eigenvalues; This will be used to create step plot
        # for visualizing the variance explained by each principal component.
        cum_sum_eigenvalues = np.cumsum(exp_var_pca)

        # Create the visualization plot
        plt.bar(range(0, len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center',
                label='Individual explained variance')
        plt.step(range(0, len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',
                 label='Cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal component index')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

        sum = 0
        c = 0
        for i in exp_var_pca:
            if sum<=0.9:
                sum+=i
                c += 1

        if (c > int( 0.8 * len(exp_var_pca))):
            print("PCA not recommended")
        else:
            pca1 = PCA(n_components = c) #contains 90% of the variance
            X = pca1.fit_transform(X)

        return X

    def SplitPcaScale(self,data):

        target = 'class'
        X = data.drop(target, axis=1)
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.fit_transform(X_test)

        X_train_scaled = self.CheckPCA(X_train_scaled)
        X_test_scaled = self.CheckPCA(X_test_scaled)

        return X_train_scaled, X_test_scaled, y_train, y_test











