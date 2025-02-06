import pickle
import sys
from abc import abstractmethod

import matplotlib.pyplot as plt

# import cplex
import numpy as np
import pyomo.environ as pyo
from gurobipy import GRB, Model, quicksum
from pyomo.opt import SolverFactory

# from data import DataLoader

# sys.path.append(
#     "/Users/clovispiedallu/opt/anaconda3/envs/cs_td/lib/python3.10/site-packages"
# )


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
        weights_1 = np.random.rand(num_features)  # Weights cluster 1
        weights_2 = np.random.rand(num_features)  # Weights cluster 2

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
        u_1 = np.dot(X, self.weights[0])  # Utility for cluster 1 = X^T.w_1
        u_2 = np.dot(X, self.weights[1])  # Utility for cluster 2 = X^T.w_2
        return np.stack([u_1, u_2], axis=1)  # Stacking utilities over cluster on axis 1


class TwoClustersMIP(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self, n_pieces, n_clusters):
        """Initialization of the MIP Variables

        Parameters
        ----------
        n_pieces: int
            Number of pieces for the utility function of each feature.
        n_clusters: int
            Number of clusters to implement in the MIP.
        """
        super().__init__()
        self.L = n_pieces
        self.K = n_clusters
        self.breaking_points = {}
        self.model = None
        self.epsilon = 0.001  # Small value for monotonicity constraint

    def instantiate(self):
        model = pyo.ConcreteModel()
        return model

    def compute_breaking_points(self, X):
        """Compute breaking points for each feature"""
        self.breaking_points = {}
        for i in range(1, self.n + 1):
            x_min = np.min(X[:, i - 1])
            x_max = np.max(X[:, i - 1])
            self.breaking_points[i] = np.linspace(x_min, x_max, self.L + 1)

    def get_segment(self, x_val, i):
        """Get segment index for a given value and feature"""
        breakpoints = self.breaking_points[i]
        idx = np.searchsorted(breakpoints, x_val) - 1
        return max(0, min(idx, self.L - 1))

    def get_segment_and_weights(self, x_val, i):
        """Get segment indices and interpolation weights for a given value and feature

        Parameters
        ----------
        x_val : float
            Value to find segments for
        i : int
            Feature index

        Returns
        -------
        tuple
            (inferior segment, superior segment, weight of inferior, weight of superior)
        """
        breakpoints = self.breaking_points[i]
        idx = np.searchsorted(breakpoints, x_val)

        # Handle edge cases
        if idx == 0:
            return 0, 0, 1.0, 0.0
        if idx == len(breakpoints):
            return self.L, self.L, 1.0, 0.0

        # Get surrounding breakpoints
        x_inf = breakpoints[idx - 1]
        x_sup = breakpoints[idx]

        # Calculate interpolation weights
        w_sup = (x_val - x_inf) / (x_sup - x_inf)
        w_inf = 1 - w_sup

        return idx - 1, idx, w_inf, w_sup

    def fit(self, X, Y):
        """Estimation of the parameters - To be completed.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        P, n = X.shape
        self.P = P
        self.n = n

        # Compute breaking points
        self.compute_breaking_points(X)

        model = self.instantiate()
        M = 1000  # Big-M parameter

        # Sets
        model.I = pyo.RangeSet(1, n)
        model.J = pyo.RangeSet(1, P)
        model.K = pyo.RangeSet(1, self.K)
        model.L = pyo.RangeSet(0, self.L)

        # Variables
        model.u = pyo.Var(model.K, model.I, model.L, bounds=(0, 1))
        model.c = pyo.Var(model.J, model.K, within=pyo.Binary)
        model.sigma = pyo.Var(model.J, model.K, bounds=(0, None))

        # Objective
        model.obj = pyo.Objective(
            expr=sum(model.sigma[j, k] for j in model.J for k in model.K),
            sense=pyo.minimize,
        )

        # Constraints
        # Normalization
        def init_zero(model, k, i):
            return model.u[k, i, 0] == 0

        model.init_zero_constr = pyo.Constraint(model.K, model.I, rule=init_zero)

        def sum_one(model, k):
            return sum(model.u[k, i, self.L] for i in model.I) == 1

        model.sum_one_constr = pyo.Constraint(model.K, rule=sum_one)

        # Monotonicity
        def monotonicity(model, k, i, l):
            if l < self.L:
                return model.u[k, i, l + 1] - model.u[k, i, l] >= self.epsilon
            return pyo.Constraint.Skip

        model.monotonicity_constr = pyo.Constraint(
            model.K, model.I, model.L, rule=monotonicity
        )

        # Preference constraints using interpolated utility values
        def preference_constr(model, j, k):
            x_util = 0
            y_util = 0

            for i in range(n):
                # Get segments and weights for X
                x_inf, x_sup, w_x_inf, w_x_sup = self.get_segment_and_weights(
                    X[j - 1, i], i + 1
                )
                x_util += (
                    w_x_inf * model.u[k, i + 1, x_inf]
                    + w_x_sup * model.u[k, i + 1, x_sup]
                )

                # Get segments and weights for Y
                y_inf, y_sup, w_y_inf, w_y_sup = self.get_segment_and_weights(
                    Y[j - 1, i], i + 1
                )
                y_util += (
                    w_y_inf * model.u[k, i + 1, y_inf]
                    + w_y_sup * model.u[k, i + 1, y_sup]
                )

            return x_util - y_util + model.sigma[j, k] >= M * (model.c[j, k] - 1)

        model.preference_constr = pyo.Constraint(
            model.J, model.K, rule=preference_constr
        )

        def preference_constr2(model, j, k):
            x_util = 0
            y_util = 0

            for i in range(n):
                # Get segments and weights for X
                x_inf, x_sup, w_x_inf, w_x_sup = self.get_segment_and_weights(
                    X[j - 1, i], i + 1
                )
                x_util += (
                    w_x_inf * model.u[k, i + 1, x_inf]
                    + w_x_sup * model.u[k, i + 1, x_sup]
                )

                # Get segments and weights for Y
                y_inf, y_sup, w_y_inf, w_y_sup = self.get_segment_and_weights(
                    Y[j - 1, i], i + 1
                )
                y_util += (
                    w_y_inf * model.u[k, i + 1, y_inf]
                    + w_y_sup * model.u[k, i + 1, y_sup]
                )

            return x_util - y_util + model.sigma[j, k] <= M * model.c[j, k]

        model.preference_constr2 = pyo.Constraint(
            model.J, model.K, rule=preference_constr2
        )

        def min_one_cluster(model, j):
            return sum(model.c[j, k] for k in model.K) >= 1

        model.min_one_cluster_constr = pyo.Constraint(model.J, rule=min_one_cluster)

        # Solve
        solver = SolverFactory("cplex_direct")
        solver.solve(model)

        self.model = model
        return self

    def predict_utility(self, X):
        """Return Decision Function of the MIP for X using proper interpolation between breakpoints.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of decision function value for each cluster.
        """
        n_samples = X.shape[0]
        utilities = np.zeros((n_samples, self.K))

        for j in range(n_samples):
            for k in range(1, self.K + 1):
                for i in range(1, self.n + 1):
                    l_inf, l_sup, w_inf, w_sup = self.get_segment_and_weights(
                        X[j, i - 1], i
                    )
                    u_inf = pyo.value(self.model.u[k, i, l_inf])
                    u_sup = pyo.value(self.model.u[k, i, l_sup])
                    utilities[j, k - 1] += w_inf * u_inf + w_sup * u_sup

        return utilities

    def plot_utility_functions(self, figsize=(15, 10)):
        """Plot utility functions for each feature and cluster.

        Parameters
        ----------
        figsize : tuple, optional
            Size of the figure, by default (15, 10)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before plotting utility functions")

        n_rows = (self.n + 1) // 2
        fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
        axes = axes.flatten()

        colors = ["b", "r", "g", "purple", "orange"]

        for i in range(1, self.n + 1):
            ax = axes[i - 1]
            breakpoints = self.breaking_points[i]
            x_plot = np.linspace(breakpoints[0], breakpoints[-1], 100)

            for k in range(1, self.K + 1):
                utils_continuous = []
                for x in x_plot:
                    l_inf, l_sup, w_inf, w_sup = self.get_segment_and_weights(x, i)
                    u_inf = pyo.value(self.model.u[k, i, l_inf])
                    u_sup = pyo.value(self.model.u[k, i, l_sup])
                    utils_continuous.append(w_inf * u_inf + w_sup * u_sup)

                ax.plot(
                    x_plot, utils_continuous, color=colors[k - 1], label=f"Cluster {k}"
                )

            ax.set_title(f"Feature {i}")
            ax.set_xlabel("Feature value")
            ax.set_ylabel("Utility")
            ax.grid(True)
            ax.legend()
            # ax.set_ylim(0, 1)

        # Remove extra subplots if any
        for i in range(self.n, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        return fig, axes


class HeuristicModel(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self):
        """Initialization of the Heuristic Model."""
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


# if __name__ == "__main__":

#     dataloader = DataLoader("../data/dataset_4")
#     X, Y = dataloader.load()

#     model = TwoClustersMIP(n_pieces=5, n_clusters=2)

#     model.fit(X, Y)
