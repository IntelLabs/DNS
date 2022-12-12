import numpy as np
from numpy.random import randn
from dppy.finite_dpps import FiniteDPP
from components.cka_similarity import CKA


class DppSamples:
    """
    Provides Dpp sample index lists from a similarity matrix
    Alex Kulesza and Ben Taskar. Determinantal Point Processes for Machine Learning.
    Foundations and Trends in Machine Learning, 5(2-3):123–286, 2012.
    URL: http://arxiv.org/abs/1207.6083, arXiv:1207.6083, doi:10.1561/2200000044.
    """

    def __init__(self, arr_sym, n_samples=1):
        """
        @param arr_sym: symmetric similarity matrix (satisfies conjugate transpose sufficient condition for existence since a real-valued symmetric matrix is equal to its conjugate transpose)
        @param n_samples: length of lists of samples to return
        @return an instance of KDD_Samples class
        """
        self.arr_sym = arr_sym
        self.n_samples = n_samples

    def is_psd(self, A):
        """
        check if matrix is positive semi-definite
        @param A: matrix
        @return True or False
        """
        return np.all(np.linalg.eigvals(A) >= 0)

    def close_psd(self, A):
        """
        Computes the nearest positive semi-definite matrix for a nonsemi-definite matrix
        Nicholas J. Higham,Computing a nearest symmetric positive semidefinite matrix,Linear Algebra and its Applications,
        Volume 103,1988,Pages 103-118,ISSN 0024-3795,
        https://doi.org/10.1016/0024-3795(88)90223-6.(https://www.sciencedirect.com/science/article/pii/0024379588902236)

        @param A: matrix
        @return A if A is positive semi-definite else nearest positive semi-definite matrix to A
        """
        if self.is_psd(A):
            return A
        C = (A + A.T) / 2
        eigval, eigvec = np.linalg.eig(C)
        eigval[eigval < 0] = 0

        return eigvec.dot(np.diag(eigval)).dot(eigvec.T)

    def sample(self):
        """
        @return dpp samples lists with random number indices each
        """
        self.arr_sym = self.close_psd(self.arr_sym)
        DPP = FiniteDPP('likelihood', **{'L': self.arr_sym})
        rng = np.random.RandomState(1)
        for _ in range(self.n_samples):
            DPP.sample_exact(random_state=rng)
        return DPP.list_of_samples


class KDppSamples(DppSamples):
    """
    Provides k-Dpp sample index lists from a similarity matrix
    """

    def __init__(self, arr_sym, n_samples=1, k=2):
        """
        @param arr_sym: similarity matrix
        @param n_samples: length of lists of samples to return
        @param k: number of indices in each list. k needs to be <= rank(arr_sym)
        @return an instance of KDppSamples class
        """
        DppSamples.__init__(self, arr_sym, n_samples)
        self.k = k

    def sample(self):
        """
        @return kdpp samples lists with k indices each
        """
        self.arr_sym = self.close_psd(self.arr_sym)
        if (self.k > np.linalg.matrix_rank(self.arr_sym)):
            print(
                f'k needs to be less than or equal to the rank of arr_sym which is {np.linalg.matrix_rank(self.arr_sym)}. Changing k to {np.linalg.matrix_rank(self.arr_sym)}')
            self.k = np.linalg.matrix_rank(self.arr_sym)
        rng = np.random.RandomState(1)
        DPP = FiniteDPP('likelihood', **{'L': self.arr_sym})
        for _ in range(self.n_samples):
            DPP.sample_exact_k_dpp(size=self.k,random_state=rng)
        return DPP.list_of_samples


if __name__ == '__main__':
    ######################### Example ################################

    # Dummy Q-values:
    size_of_ensemble = 10
    q_values = []
    for i in range(size_of_ensemble):
        q_values.append(np.random.rand(256, 64))
    
    cka_sim_calc = CKA()
    cka_matrix = np.zeros((size_of_ensemble, size_of_ensemble))
    for i in range(len(cka_matrix) - 1):
        for j in range(i + 1, len(cka_matrix)):
            cka_matrix[i, j] = cka_sim_calc(q_values[i], q_values[j])
    cka_matrix = cka_matrix + cka_matrix.T
    np.fill_diagonal(cka_matrix, 1)
    
    # Example 1:  get 1 k-dpp sample index list.
    print(KDppSamples(cka_matrix, n_samples=1, k=3).sample())
