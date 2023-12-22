from sklearn.decomposition import PCA
import pickle as pk


class SavablePCA:
    """
    A wrapper around scikit learn PCA that allows saving
    and loading of the transformation.
    """
    def __init__(self, n_components):
        # if none is passed load is called right away
        if n_components:
            # use full solver to make it deterministic
            self.pca = PCA(n_components=n_components, svd_solver="full")

    def fit(self, X):
        self.pca.fit(X)

    def transform(self, X):
        return self.pca.transform(X)

    def save(self, path):
        with open(path, "wb") as fl:
            pk.dump(self.pca, fl)

    def load(self, path):
        with open(path, "rb") as fl:
            self.pca = pk.load(fl)

    @staticmethod
    def from_file(path):
        savable_pca = SavablePCA(None)
        savable_pca.load(path)
        return savable_pca
