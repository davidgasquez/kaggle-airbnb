from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
import numpy as np

X = np.arange(9).reshape(3, 3)
print X

poly = PolynomialFeatures(2)
print poly.fit_transform(X)

poly = PolynomialFeatures(interaction_only=True)
print poly.fit_transform(X)

pca = PCA(n_components=1)

print pca.fit_transform(X)
