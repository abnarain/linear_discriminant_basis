import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def filereader(filename,fs,start_sec=2.00, end_sec=2.01):
    z= scipy.fromfile(open(filename), dtype=scipy.complex64)
    # dtype with scipy.int16, scipy.int32, scipy.float32, scipy.complex64 or whatever type you were using
    mag, phase,x,y = [], [], [], []
    print start_sec*fs, end_sec*fs, len(z)
    z_needed = z[int(start_sec*fs): int(end_sec*fs)]
    z=z_needed
    log_z= log( int(len(z))/2  ,2)
    len_z= int(2**(int(log_z)+1))
    for i in range(0, len_z):
        mag.append(np.absolute(z[i]))
        #x.append([i].real)                                                                                                                          
        #y.append(z[i].imag)                                    
        #phase.append(np.angle(z[i]))                                                                                                                
    #plt.figure()
    #plt.plot(mag)
    #plt.savefig(filename+'.pdf')
    #sys.exit(1)
    return [x,y,mag, phase,z]


def smoothGaussian(myarray, degree=5):
    myarray = np.pad(myarray, (degree-1,degree-1), mode='edge')
    window=degree*2-1
    weight=np.arange(-degree+1, degree)/window
    weight = np.exp(-(16*weight**2))
    weight /= sum(weight)
    smoothed = np.convolve(myarray, weight, mode='valid')
    return smoothed

def data_segmentation(train_files,fs,period):
    X=[]
    Y=[]
    cutoff=80*1000.0
    c,count=0,0
    period=int(fs/15)
    period=int(period/7) # for one pulse/cycle long trace as input features
    for tf in train_files:
        #[x,y, mag, phase,z] = filereader(tf,int(fs),2,2.48)
        [x,y, mag, phase,z] = filereader(tf,int(fs), 1, 6)
        for subset_idx in range(0,len(mag)-period, period):
            c +=1
            subset_mag= mag[subset_idx:subset_idx+period]
            max_idx= subset_idx + np.argmax(subset_mag)
            data_ = mag[max_idx -period/2 : max_idx+period/2]
            '''
            fig=plt.figure()
            ax1=plt.subplot(1,1,1)
            ax1.plot(subset_mag,'black')
            ax1.set_ylabel('amplitude (calibrated by gnuradio)')
            ax1.set_xlabel('time fs=500 KHz, 10msec trace')
            #data_ =smoothGaussian(data_)
            '''
            #ax2=plt.subplot(2,1,2)
            #ax2.plot(data_,'r')
            #fig.savefig('single_testing_'+str(c)+'_'+'.png')
            #fig.clf()
            #'''
            #print len(data_) ,
            #print "funny",c, subset_idx, len(mag)
            if len(data_)>4700:
                X.append(data_)
                Y.append(count)
            del data_
        del mag
        count +=1
    print "total number of classes =",count
    return [X,Y]

def segmentation_period(train_files,fs):
    stepsize=200000
    [x,y,mag,phase,z]=filereader(train_files[0],int(fs),2,2.3)
    sig=np.array(mag[0:stepsize])
    max_idx1=np.argmax(sig[:stepsize/2])
    max_idx2=np.argmax(sig[stepsize/2:stepsize])
    diff_peaks_idxs=(max_idx2+stepsize/2)-max_idx1
    result=np.correlate(sig,sig,'full')
    period=np.argmax(result[:stepsize/2])
    period=(diff_peaks_idxs + period)/2
    print "values of indices are ", max_idx1, max_idx2, diff_peaks_idxs, period
    print "values in the idxs are ", sig[max_idx1], sig[max_idx2+stepsize/2]
    return period


def lda():
	clf = LinearDiscriminantAnalysis()
	clf.fit(X, y)
	LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
						   solver='svd', store_covariance=False, tol=0.0001)


def main(argv):
    inputfile=''
    mypath,outputfile='',''
    pathflag,inputflag=0,0
    try:
        opts, args = getopt.getopt(argv,"h:i:f:o:",["ifile=","folder=","ofile="])
    except getopt.GetoptError:
        print 'file.py -i <testfile> -f <trainfolder> -o <outputpdf> '
        sys.exit(2)

    for opt, arg in opts:
        print opt ,arg,
        if opt == '-h':
            print 'file.py -i <testfile> -f <trainfolder> -o <outputpdf>'
            sys.exit()
	elif opt in ("-i", "--ifile"):
            inputfile = arg
            inputflag=1
        elif opt in ("-f", "--folder"):
            mypath = arg
            pathflag=1
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        else:
            print "check help for usage"
            sys.exit()


    fs=500e3    
    mypath= os.path.abspath(mypath)
    train_files = [mypath+'/'+f for f in listdir(mypath) if isfile(join(mypath, f))]
    #[x1,y1, test_mag, phase1,z1] = filereader(inputfile,int(fs))
    print train_files
    period= segmentation_period(train_files,fs)
    #period =87048
    period =5000

    [X,Y] = data_segmentation(train_files,fs,period)
    X=np.array(X)
    Y=np.array(Y)
    #sys.exit(1)
    for family in pywt.families():
        for wavelet in pywt.wavelist(family):
            #wavelet='sym5'
            print "WAVELET used ", wavelet
            X_coeffs=[]
            for x in X:
                x_coeffs=ldb(x) #'dmey'                
                X_coeffs.append(x_coeffs)
                #print len(x_coeffs),

            X_coeffs=np.array(X_coeffs)

            print "------------------------------"
            del X_coeffs

if __name__=='__main__':
    main(sys.argv[1:])
