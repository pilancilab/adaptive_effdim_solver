import numpy as np 
from sklearn.datasets import fetch_openml, fetch_rcv1
from scipy.sparse import issparse 

import argparse



def argparser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default='dilbert')
	return parser.parse_args()


def one_hot(b):
	vals, inverse = np.unique(b, return_inverse=True)
	num_classes = int(max(vals)) + 1
	return np.eye(num_classes)[inverse]


def download_data_openml(dataset):
	"""
	example dataset
	['dilbert', 'SVHN', 'CIFAR-100', 'OVA_Breast', 'OVA_Lung']
	"""

	print('----- dataset: ', dataset)

	print('fetching dataset')
	if dataset in ['dilbert', 'SVHN', 'CIFAR-100', 'OVA_Breast', 'OVA_Lung']:
		data = fetch_openml(name=dataset)
	else:
		raise NotImplementedError

	print('processing data')
	A = data['data']
	if issparse(A):
		A = A.toarray()
	b = one_hot(data['target'])

	n, d = A.shape

	if n >= d:
		AtA = A.T @ A 
	else:
		AtA = A @ A.T

	sigma = np.sqrt(np.flip(np.sort(np.abs(np.linalg.eigvalsh(AtA)))))
	sigma /= sigma[0]
	A /= sigma[0]

	np.savez('./' + dataset + '.npz', A=A, b=b, sigma=sigma)


def main(args):
	dataset = args.dataset
	download_data_openml(dataset)


if __name__ == '__main__':
	args = argparser()
	main(args)











