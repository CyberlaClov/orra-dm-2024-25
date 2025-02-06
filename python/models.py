import pickle
from abc import abstractmethod
from gurobipy import Model, GRB, quicksum
import numpy as np


class BaseModel(object):
    """
    Base class for models, to be used as coding pattern skeleton.
    Can be used for a model on a single cluster or on multiple clusters"""

    def __init__(self):
        """Initialization of your model and its hyper-parameters"""
        pass

    @abstractmethod
    def fit(self, X, Y):
        """Fit function to find the parameters according to (X, Y) data.
        (X, Y) formatting must be so that X[i] is preferred to Y[i] for all i.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        # Customize what happens in the fit function
        return

    @abstractmethod
    def predict_utility(self, X):
        """Method to call the decision function of your model

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of decision function value for each cluster.
        """
        # Customize what happens in the predict utility function
        return

    def predict_preference(self, X, Y):
        """Method to predict which pair is preferred between X[i] and Y[i] for all i.
        Returns a preference for each cluster.

        Parameters
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements to compare with Y elements of same index
        Y: np.ndarray
            (n_samples, n_features) list of features of elements to compare with X elements of same index

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of preferences for each cluster. 1 if X is preferred to Y, 0 otherwise
        """
        X_u = self.predict_utility(X)
        Y_u = self.predict_utility(Y)

        return (X_u - Y_u > 0).astype(int)

    def predict_cluster(self, X, Y):
        """Predict which cluster prefers X over Y THE MOST, meaning that if several cluster prefer X over Y, it will
        be assigned to the cluster showing the highest utility difference). The reversal is True if none of the clusters
        prefer X over Y.
        Compared to predict_preference, it indicates a cluster index.

        Parameters
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements to compare with Y elements of same index
        Y: np.ndarray
            (n_samples, n_features) list of features of elements to compare with X elements of same index

        Returns
        -------
        np.ndarray:
            (n_samples, ) index of cluster with highest preference difference between X and Y.
        """
        X_u = self.predict_utility(X)
        Y_u = self.predict_utility(Y)

        return np.argmax(X_u - Y_u, axis=1)

    def save_model(self, path):
        """Save the model in a pickle file. Don't hesitate to change it in the child class if needed

        Parameters
        ----------
        path: str
            path indicating the file in which the model will be saved
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(clf, path):
        """Load a model saved in a pickle file. Don't hesitate to change it in the child class if needed

        Parameters
        ----------
        path: str
            path indicating the path to the file to load
        """
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model


class RandomExampleModel(BaseModel):
    """Example of a model on two clusters, drawing random coefficients.
    You can use it to understand how to write your own model and the data format that we are waiting for.
    This model does not work well but you should have the same data formatting with TwoClustersMIP.
    """

    def __init__(self):
        self.seed = 444
        self.weights = self.instantiate()

    def instantiate(self):
        """No particular instantiation"""
        return []

    def fit(self, X, Y):
        """fit function, sets random weights for each cluster. Totally independant from X & Y.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        np.random.seed(self.seed)
        num_features = X.shape[1]
        weights_1 = np.random.rand(num_features) # Weights cluster 1
        weights_2 = np.random.rand(num_features) # Weights cluster 2

        weights_1 = weights_1 / np.sum(weights_1)
        weights_2 = weights_2 / np.sum(weights_2)
        self.weights = [weights_1, weights_2]
        return self

    def predict_utility(self, X):
        """Simple utility function from random weights.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of decision function value for each cluster.
        """
        u_1 = np.dot(X, self.weights[0]) # Utility for cluster 1 = X^T.w_1
        u_2 = np.dot(X, self.weights[1]) # Utility for cluster 2 = X^T.w_2
        return np.stack([u_1, u_2], axis=1) # Stacking utilities over cluster on axis 1


class TwoClustersMIP(BaseModel):
    def __init__(self, n_pieces, n_clusters):
        super().__init__()
        self.L = n_pieces 
        self.K = n_clusters
        self.model = None

    def instantiate(self, P, n):
        model = Model("UTA MIP")
        self.u = {}
        self.c = {}
        self.sigma = {}
        
        for k in range(1, self.K + 1):
            for i in range(1, n + 1):
                for l in range(self.L + 1):
                    self.u[k, i, l] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS)
                    
        for j in range(1, P + 1):
            for k in range(1, self.K + 1):
                self.c[j, k] = model.addVar(vtype=GRB.BINARY)
                self.sigma[j, k] = model.addVar(lb=0, vtype=GRB.CONTINUOUS)
                
        model.update()
        return model
    
    def compute_breaking_points(self, X):
        self.breaking_points = {}
        for i in range(1, self.n + 1):
            x_min = np.min(X[:, i-1])
            x_max = np.max(X[:, i-1])
            self.breaking_points[i] = [x_min + l * (x_max - x_min) / self.L for l in range(self.L + 1)]


    def find_closest_breakpoints(self, i, x_val):
        breakpoints = self.breaking_points[i]
        l = max(0, np.searchsorted(breakpoints, x_val) - 1)
        l_next = min(l + 1, self.L)
        return l, l_next


    def interpolate(self, k, i, x_val):
        l, l_next = self.find_closest_breakpoints(i, x_val)
        
        u_l = self.u[k, i, l].X
        u_l_next = self.u[k, i, l_next].X if l_next < self.L else u_l
        
        return u_l + (x_val - self.breaking_points[i][l]) / (self.breaking_points[i][l_next] - self.breaking_points[i][l]) * (u_l_next - u_l)

    def fit(self, X, Y):
        P, n = X.shape
        self.P = P
        self.n = n
        self.model = self.instantiate(P, n)
        self.compute_breaking_points(X)

        # Normalization constraints
        for k in range(1, self.K + 1):
            self.model.addConstr(quicksum(self.u[k, i, self.L] for i in range(1, n + 1)) == 1)

        # Monotonicity constraints  
        for k in range(1, self.K + 1):
            for i in range(1, n + 1):
                self.model.addConstr(self.u[k, i, 0] == 0)
                for l in range(self.L):
                    self.model.addConstr(self.u[k, i, l + 1] >= self.u[k, i, l])

        # Preference constraints
        M = 1000
        for j in range(1, P + 1):
            self.model.addConstr(quicksum(self.c[j, k] for k in range(1, self.K + 1)) >= 1)
            for k in range(1, self.K + 1):
                utility_diff = quicksum(
                    self.interpolate(k, i, X[j-1, i-1]) - self.interpolate(k, i, Y[j-1, i-1])
                    for i in range(1, n + 1)
                )
                self.model.addConstr(M * (self.c[j, k] - 1) <= utility_diff + self.sigma[j, k])
                self.model.addConstr(utility_diff + self.sigma[j, k] <= M * self.c[j, k])
        
        self.model.setObjective(
            quicksum(self.sigma[j, k] for j in range(1, P + 1) for k in range(1, self.K + 1)),
            GRB.MINIMIZE
        )
        
        self.model.optimize()
        return self

    def predict_utility(self, X):
        P, n = X.shape
        utilities = np.zeros((P, self.K))
        
        for j in range(1, P + 1):
            for k in range(1, self.K + 1):
                for i in range(1, n + 1):
                    utilities[j-1, k-1] += self.interpolate(k, i, X[j-1, i-1])
        
        return utilities

class HeuristicModel(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self):
        """Initialization of the Heuristic Model.
        """
        self.seed = 123
        self.models = self.instantiate()

    def instantiate(self):
        """Instantiation of the MIP Variables"""
        # To be completed
        return

    def fit(self, X, Y):
        """Estimation of the parameters - To be completed.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        # To be completed
        return

    def predict_utility(self, X):
        """Return Decision Function of the MIP for X. - To be completed.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        
        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of decision function value for each cluster.
        """
        # To be completed
        # Do not forget that this method is called in predict_preference (line 42) and therefor should return well-organized data for it to work.
        return
