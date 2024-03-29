# -*- coding: utf-8 -*-
"""Similarity of Neural Network Representations Revisited Demo.ipynb

Automatically generated by Colaboratory.

Original file is located at
	https://colab.research.google.com/github/google-research/google-research/blob/master/representation_similarity/Demo.ipynb

# Demo code for "[Similarity of Neural Network Representations Revisited](https://arxiv.org/abs/1905.00414)"

Copyright 2019 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Please cite as:

	@inproceedings{pmlr-v97-kornblith19a,
	  title = {Similarity of Neural Network Representations Revisited},
	  author = {Kornblith, Simon and Norouzi, Mohammad and Lee, Honglak and Hinton, Geoffrey},
	  booktitle = {Proceedings of the 36th International Conference on Machine Learning},
	  pages = {3519--3529},
	  year = {2019},
	  volume = {97},
	  month = {09--15 Jun},
	  publisher = {PMLR}
	}
"""


import numpy as np


class CKA:
	def __init__(self):
		pass
	
	def _gram_linear(self, x):
		"""Compute Gram (kernel) matrix for a linear kernel.
		Args:
		x: A num_examples x num_features matrix of features.

		Returns:
		A num_examples x num_examples Gram matrix of examples.
		"""
		return x.dot(x.T)

	
	def __call__(self, gram_x, gram_y, debiased=False):
		"""Compute CKA.

	Args:
		gram_x: A num_examples x num_examples Gram matrix.
		gram_y: A num_examples x num_examples Gram matrix.
		debiased: Use unbiased estimator of HSIC. CKA may still be biased.

	Returns:
		The value of CKA between X and Y.
	""" 
		gram_x = self._gram_linear(gram_x)
		gram_y = self._gram_linear(gram_y)
		gram_x = self._center_gram(gram_x, unbiased=debiased)
		gram_y = self._center_gram(gram_y, unbiased=debiased)

		# Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
		# n*(n-3) (unbiased variant), but this cancels for CKA.
		scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

		normalization_x = np.linalg.norm(gram_x)
		normalization_y = np.linalg.norm(gram_y)
		return scaled_hsic / (normalization_x * normalization_y)
	
	def _center_gram(self, gram, unbiased=False):
		"""Center a symmetric Gram matrix.
		This is equvialent to centering the (possibly infinite-dimensional) features
  		induced by the kernel before computing the Gram matrix.

  		Args:
		gram: A num_examples x num_examples symmetric matrix.
		unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
	  	estimate of HSIC. Note that this estimator may be negative.

  		Returns:
		A symmetric matrix with centered columns and rows.
  		"""
		if not np.allclose(gram, gram.T):
			raise ValueError('Input must be a symmetric matrix.')
		gram = gram.copy()

		if unbiased:
			# This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
			# L. (2014). Partial distance correlation with methods for dissimilarities.
			# The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
			# stable than the alternative from Song et al. (2007).
			n = gram.shape[0]
			np.fill_diagonal(gram, 0)
			means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
			means -= np.sum(means) / (2 * (n - 1))
			gram -= means[:, None]
			gram -= means[None, :]
			np.fill_diagonal(gram, 0)
		else:
			means = np.mean(gram, 0, dtype=np.float64)
			means -= np.mean(means) / 2
			gram -= means[:, None]
			gram -= means[None, :]

		return gram
	


"""## Tutorial

First, we generate some random data.
"""
if __name__=='__main__':
	np.random.seed(1337)
	X = np.random.randn(100, 10)
	Y = np.random.randn(100, 10) + X
	cka = CKA()
	"""Linear CKA can be computed either based on dot products between examples or dot products between features:
	$$\langle\text{vec}(XX^\text{T}),\text{vec}(YY^\text{T})\rangle = ||Y^\text{T}X||_\text{F}^2$$
	The formulation based on similarities between features (right-hand side) is faster than the formulation based on similarities between similarities between examples (left-hand side) when the number of examples exceeds the number of features. We provide both formulations here and demonstrate that they are equvialent.
	"""
	cka_from_examples = cka(X, Y)
	print('Linear CKA from Examples: {:.5f}'.format(cka_from_examples))