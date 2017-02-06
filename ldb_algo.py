import numpy as np
import pylab, sys
from pywt import WaveletPacket
from scipy import stats
import sys

num_classes = 2
max_level = 4
signal_length = 1024

def ldb_measure(coeffs1,coeffs2):
	h1,b1= np.histogram(coeffs1,normed=True)
	h1=h1/sum(h1)
	coeffs1_dist=[]
	for i in range(0,len(h1)):
		if h1[i] !=0.0:	
			coeffs1_dist.append(h1[i])

	h2,b2= np.histogram(coeffs2,normed=True)
	h2=h2/sum(h2)
	coeffs2_dist=[]
	for i in range(0,len(h2)):
		if h2[i] !=0.0:
			coeffs2_dist.append(h2[i])
	print "length of coeffs", len(coeffs2_dist), len(coeffs2_dist)
	entr=stats.entropy(h1,h2) #coeffs1_dist,coeffs2_dist)
	return entr

# Gamma coefficient for signals in class c.
def get_wavelet_coeffs_jk(signal, j, k):
        return signal.get_level(j, "natural")[k].data

def gamma_c(signals_c):
        # Test input: the only input is a single signal of that class.
        assert len(signals_c) == 1
        gamma = np.zeros((max_level+1,2**max_level,signal_length))
        for j in range(0, max_level):
                for k in range(0, 2**j):
                        signal_arr = get_wavelet_coeffs_jk(signals_c[0], j, k)
                        for l in range(0,len(signal_arr)):
                                gamma[j][k][l] = signal_arr[l]
        return gamma

def two_number_discriminant(val1, val2):
        return abs(val1 - val2)

def class_number_discriminant(gamma_jk_list):
        # the argument gamma_jk_list is a vector of num_classes values,  each
        # corresponding to one j,k,l coefficient for a particular class.
        assert len(gamma_jk_list) == num_classes
        total_disc = 0.0
        for c1 in range(0, num_classes):
                for c2 in range(c1, num_classes):
                        total_disc += two_number_discriminant(
                                gamma_jk_list[c1],
                                gamma_jk_list[c2])
        return total_disc

def multi_class_discriminant(gamma_jk_list):
        # gamma_jk_list is a list of gamma_jk vectors for multiple
        # classes.
        assert len(gamma_jk_list) == num_classes
        num_l_coeffs = len(gamma_jk_list[0])
        total_disc = 0.0
        for l in range(0, num_l_coeffs):
                # collect all the l'th coefficients from the gamma_jk_list
                # corresponding to each class.
                l_coeffs = reduce(lambda acc, x: acc + [x[l]],
                                  gamma_jk_list,
                                  [])
                assert len(l_coeffs) == num_classes
                total_disc += class_number_discriminant(l_coeffs)
        return total_disc

def init_jk_map(J, init_fun):
        x = {}
        for j in range(0, J):
                x[j] = {}
                for k in range(0, 2**j):
                        x[j][k] = init_fun(j,k)
        return x

def get_jk_coeff_list(gamma_list, j, k):
        jk_list = []
        for c in range(0, len(gamma_list)):
                jk_list.append(gamma_list[c][j][k])
        return jk_list

# main function
def main(classwise_signal_list):
        assert len(classwise_signal_list) == num_classes
        print "The input signals are:"
        for c in classwise_signal_list:
                for s in c:
                        print s.data
        # step 1. Construct time-frequency maps
        gamma_list = []
        for c in range(0, num_classes):
                gamma_list.append(gamma_c(classwise_signal_list[c]))
        print "classwise energy coefficients:"
        print gamma_list
        # step 2. initialize A_Jk and delta lists.
        # for all j,k, now
        # A[j][k] == {(j,k)}
        A = init_jk_map(max_level, lambda x, y: set([(x,y)]))
        delta = init_jk_map(max_level,
                            lambda x, y: multi_class_discriminant(
                                    get_jk_coeff_list(gamma_list, x, y)))
        # step 3. recurse over tree and check delta values
        for j in range(max_level-1, -1):
                for k in range(0, 2**max_level):
                        new_delta = delta[j+1][2*k] + delta[j+1][2*k+1]
                        if delta[j][k] < new_delta:
                                delta[j][k] = new_delta
                                A[j][k] = A[j+1][2*k] + A[j+1][2*k+1]
        # skipping steps 4 and 5. Check the output.
        print A[0][0]

if __name__=='__main__':
	#two classes each with two samples of data
	time = np.arange(100, 20, -0.5) / 150.
	#data1 = np.sin(20 * pylab.log(time)) * np.sign((pylab.log(time)))
	#data2 = np.sin(20 * pylab.log(time)) * np.cos((pylab.log(time)))
	amp = 1
	data1 = amp * np.sin(2*np.pi*300*time)
	data2 = np.concatenate(([0]*50, amp * np.sin(2*np.pi*300*time)),axis=1)
	from matplotlib import pyplot as plt
	plt.figure()
	plt.plot(data1,'b')
	plt.plot(data2,'r')
	plt.show()
	sys.exit(1)
	wp1 = WaveletPacket(data1, 'sym5', maxlevel=4)
	wp2 = WaveletPacket(data2, 'sym5', maxlevel=4)
	WP=[wp1,wp2]
	
	main([[wp1], [wp2]])
	print "foo"
