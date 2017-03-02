import numpy as np
from collections import defaultdict
import pylab, sys, math
from pywt import WaveletPacket
from matplotlib import pyplot as plt
from scipy import stats
num_classes=2
max_level = 5 #int(math.log(len(data1),2))
signal_length = 4096
# Gamma coefficient for signals in class c.
def get_wavelet_coeffs_jk(signal, j, k):
	return signal.get_level(j, "natural")[k].data

def gamma_c(signals_c):
	# Test input: the only input is a single signal of that class.
	assert len(signals_c) == 1
	gamma = np.zeros((max_level,2**max_level,signal_length))
	for j in range(0, max_level):
		for k in range(0, 2**j-1):
			signal_arr = get_wavelet_coeffs_jk(signals_c[0], j, k)
			for l in range(0,len(signal_arr)):
				gamma[j][k][l] = signal_arr[l]
	return gamma

def kl_measure(coeffs1,coeffs2):
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
	entr=stats.entropy(h1,h2) #coeffs1_dist,coeffs2_dist)
	return entr

def multi_class_discriminant_j_divergence(gamma_jk_list):
	# gamma_jk_list is a list of gamma_jk vectors for multiple
	# classes.
	assert len(gamma_jk_list) == num_classes
	total_disc = 0.0
	for c1 in range(0, num_classes):
		for c2 in range(c1, num_classes):
			total_disc += kl_measure(
				gamma_jk_list[c1],
				gamma_jk_list[c2]) + kl_measure(gamma_jk_list[c2], gamma_jk_list[c1])
	return total_disc


def two_number_discriminant(val1, val2):
	return (val1 - val2)**2

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


def multi_class_discriminant_l2(gamma_jk_list):
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
	jk_list = [] #these are coefficients l at node j,k
	for c in range(0, len(gamma_list)):
		jk_list.append(gamma_list[c][j][k])
	return jk_list

def max_indices(delta):
	print "\ndelta values are\n",
	'''
	for x in delta:
		print "\nx is: ",x
		for xx in delta[x]:
			print "delta[x] ", xx, delta[x][xx]
	'''

# main function
def ldb(classwise_signal_list):
	assert len(classwise_signal_list) == num_classes

	#print "The input signals are:"
	#for c in classwise_signal_list:
	#for s in c:
	#		print s.data
	# step 1. Construct time-frequency maps
	
	gamma_list = []
	for c in range(0, num_classes):
		gamma_list.append(gamma_c(classwise_signal_list[c]))
	#print "classwise energy coefficients:"

	# step 2. initialize A_Jk and delta lists.

	# for all j,k, now
	# A[j][k] == {(j,k)}
	A = init_jk_map(max_level, lambda x, y: set([(x,y)]))
	#print "initial tree of A is "
	#print A
	delta = init_jk_map(max_level,
			    lambda x, y: multi_class_discriminant_l2(
			get_jk_coeff_list(gamma_list, x, y)))
	# step 3. recurse over tree and check delta values
	for j in range(max_level-2,-1,-1):
		#for k in range(0, 2**max_level):
		for k in range(0, 2**j):
			new_delta = delta[j+1][2*k] + delta[j+1][2*k+1]
			if delta[j][k] < new_delta :
				delta[j][k] = new_delta
				#print "j,k; j+1,k; j+1,k+1 ", A[j][k], A[j+1][k], A[j+1][k+1]
				new_set=set([])
				new_set |= A[j+1][2*k]
				new_set |= A[j+1][2*k+1]
				A[j][k] = new_set #A[j+1][k] + A[j+1][k+1]
    # skipping steps 4 and 5. Check the output.
	'''
	print delta
	max_indices_list=max_indices(delta)
	#
	print A[0][0]
	'''
	class_coeffs=defaultdict(list)
	for ss in A[0][0]:
		for c in range(0, num_classes):
			class_coeffs[c].append(get_wavelet_coeffs_jk(classwise_signal_list[c][0],ss[0], ss[1]))
	feature_vector=[]
	for i,j in class_coeffs.iteritems():
		#print i, j
		feature_vector.append(np.concatenate(j,axis=0))

	return feature_vector


def plot_wavelet_spectrogram():
	c=['r','b','brown','orange','cyan']
	f,axxr=plt.subplots(num_classes+1,1)
	for i in range(0, num_classes):
		axxr[i].plot(classes[i],c[i])
	dim =(max_level,signal_length)
	matrx=np.zeros(dim)
	for (idx,jdx) in node_list:
		block_length = signal_length/(2**idx)
		for kdx in range((jdx)*(block_length),(jdx+1)*(block_length)):
			matrx[idx,kdx]=1
	axxr[num_classes].matshow(matrx)
	#plt.show()

def test_data():
	amp = -1
	#time =  np.array([int(i) for i in range(0,64)])	
	#two classes each with two samples of data
	#time = np.arange(1,2041,0.5) #/ 150.
	time = np.arange(1,25,0.5) #/ 150.
	#data1 = np.sin(20 * pylab.log(time)) * np.sign((pylab.log(time)))
	#data3 = np.concatenate((np.sin(20 * pylab.log(time)), np.cos(pylab.log(time))),axis=1)
	#data1 = np.concatenate((amp * np.sin(2*np.pi*300*time),[0]*16),axis=0)
	data1 = np.concatenate(([1]*16,[0]*(time)),axis=0)
	#data2 = np.concatenate(([0]*16,[1]*time),axis=1)
	data2 = np.concatenate(([0]*16,amp * np.sin(2*np.pi*16*time)),axis=0)
	#data2 = np.concatenate(([0]*16, amp * np.sin(2*np.pi*300*time)),axis=1)
	#data1 =np.sin(2 * pylab.log(1+time))
	#data2 = np.concatenate((np.sin(29 * pylab.log(1+time)),np.sin(29 * pylab.log(1+time))), axis=1)
	data3 = np.array([0]*32)
	#data1 = np.array([0]*32)
	#data1 = np.concatenate(([32]*12,[.5]*20,[5]*32),axis=1)
	print data1
	print data2
	print len(data1),len(data2),len(data3)
	return [data1, data2, data3]

def test_main():
	classes=[data1, data2,data3] = test_data()
	signal_length = len(data1)
	assert signal_length!= 0 and ((signal_length & (signal_length - 1)) == 0)
	assert len(data1)==len(data2)
	
	max_level = int(math.log(len(data1),2))
	num_classes = len(classes)
	print "max level is ", max_level, "num of classes", num_classes
	WP= []
	for i in range(0,num_classes):
		WP.append([WaveletPacket(classes[i], 'db1', maxlevel=max_level)])
	node_list=ldb(WP)
	#plot_wavelet_spectrogram()

def ldb_main(X_data):
	classes=X_data
	print len(X_data)
	signal_length = len(X_data[0][0])
	print len(X_data[0][0]),len(X_data[1][0])
	min_indx=min (len(X_data[0]),len(X_data[1]))
	X_data[0]=X_data[0][:min_indx]
	X_data[1]=X_data[1][:min_indx]
	print len(X_data[0]),len(X_data[1])

	assert signal_length!= 0 and ((signal_length & (signal_length - 1)) == 0)
	assert len(X_data[0])==len(X_data[1])
	max_level = 5 #int(math.log(len(data1),2))
	num_classes = len(classes)
	print "max level is ", max_level, "num of classes", num_classes
	WP= []
	class_1, class_2=[],[]
	for j in range(0, len(classes[0])):
		WPc=[]
		for i in range(0,num_classes):
			WPc.append([WaveletPacket(classes[i][j], 'db1', maxlevel=max_level)])
		#get the coefficients of each of the data corresponding to node list

		feature_vector=ldb(WPc)
		class_1.append(feature_vector[0])
		class_2.append(feature_vector[1])
		       #make an array of it and return it to LDA algorithm for classification
	#plot_wavelet_spectrogram()
	print "returng from ldb"
	return [class_1,class_2]
