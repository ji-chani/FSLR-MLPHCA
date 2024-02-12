# module for feature scaling and reduction methods

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def reduce_features(X_train, X_test, scaled:bool, pca:bool):

    if scaled:
        # feature scaling
        print('Feature scaling in progress...')
        scaler = StandardScaler().fit(X_train)
        X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
        print('Feature scaling complete.')

    if pca:
        # feature reduction
        print('Feature reduction in progress...')
        PCAnalysis = PCA(0.95).fit(X_train)
        X_train, X_test = PCAnalysis.transform(X_train), PCAnalysis.transform(X_test)
        print('Feature reduction complete.')

    print(f'Dimension of each instance is {len(X_test[0])}. \n')
    return X_train, X_test