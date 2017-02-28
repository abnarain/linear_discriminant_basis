import numpy as np
import scipy, getopt, sys, os, pywt, pylab, itertools,math
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_validation import train_test_split,KFold, cross_val_score, StratifiedKFold
from ldb import *

def mult_kfold_roc(X, y, classifier, title, class_names, filename):
    K=10
    print "I am in kfold"
    cv = StratifiedKFold(y,n_folds=K)
    # Binarize the output
    print len(y)
    y = label_binarize(y, classes=[0, 1, 2,3,4])
    print len(y)
    n_classes = y.shape[1]
    print n_classes
    # Compute Precision-Recall and plot curve
    tpr = defaultdict(list)
    fpr= defaultdict(list)
    avg_auc = defaultdict(list)

    avg_f1_s, avg_prec_s, avg_rec_s, avg_acc_s, avg_class_acc_s=defaultdict(np.float64), defaultdict(np.float64), defaultdict(np.float64), defaultdict(np.float64), defaultdict(np.float64)
    for (train, test), color in zip(cv, colors) :
        X_train, X_test, y_train, y_test= X[train], X[test], y[train], y[test]
        y_score = classifier.fit(X_train, y_train).predict_proba(X_test) #is a list
        y_pred = classifier.fit(X_train, y_train).predict(X_test) #is a list
        #for one layer classifier
        '''
        y_s= np.array([]).reshape(y_score[0].shape[0],0)
        for i in y_score:
            print "shape of column is ", i[:,1].shape
            y_s= np.c_[y_s, i[:,1]]
            y_score=y_s

        '''
        for i in range(n_classes):
            fpr_, tpr_, _ = roc_curve(y_test[:, i], y_score[:, i])
            tpr[i].append(tpr_)
            fpr[i].append(fpr_)
            avg_prec_s[i] +=  precision_score(y_test[:, i], y_pred[:, i], average="macro")
            avg_class_acc_s[i] += classifier.score(X_test, y_test)
            avg_rec_s[i] +=  recall_score(y_test[:, i], y_pred[:, i], average="macro")
            avg_f1_s[i] +=  f1_score(y_test[:, i], y_pred[:, i] , average="macro")


    for i in range(n_classes):
        print "CLASS ", i, "f1 score", avg_f1_s[i]/K*100, "Prec ", 100*avg_prec_s[i]/K, "Rec ", 100*avg_rec_s[i]/K, "Acc ", 100*avg_class_acc_s[i]/K
    fig= plt.figure()
    ax1=fig.add_subplot(111)
    overall_tpr=[]
    overall_fpr=[]
    overall_auc=[]
    for i in range(n_classes):
        list_tpr=tpr[i]
        list_fpr=fpr[i]
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        mean_auc=0.0
        for j in range(K):
            mean_tpr += interp(mean_fpr, list_fpr[j], list_tpr[j])
            #mean_tpr[0] = 0.0
        #mean_auc += average_precision_score[j]
        mean_tpr /=K
        mean_tpr[-1]=1.0
        mean_auc =auc(mean_fpr, mean_tpr)
        overall_tpr.append(mean_tpr)
        overall_fpr.append(mean_fpr)
        overall_auc.append(mean_auc)
    lw=2
    for i, color in zip(range(n_classes), colors):
        ax1.plot(overall_fpr[i], overall_tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(class_names[i], overall_auc[i]))
    ax1.set_xlim([0.0, 1.05])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC curve'+str(K)+'fold' )
    ax1.legend(loc="best")
    plt.savefig(filename+'.pdf')


def filereader(filename,fs,start_sec=1.00, end_sec=1.01):
    z= scipy.fromfile(open(filename), dtype=scipy.complex64)
    # dtype with scipy.int16, scipy.int32, scipy.float32, scipy.complex64 or whatever type you were using
    mag, phase,x,y = [], [], [], []
    print start_sec*fs, end_sec*fs, len(z)
    #'''
    z_needed = z[int(start_sec*fs): int(end_sec*fs)]
    z=z_needed
    #'''
    len_z=len(z)
    for i in range(0, len_z):
        mag.append(np.absolute(z[i]))
        #x.append([i].real)                                                                                                                          
        #y.append(z[i].imag)                                    
        #phase.append(np.angle(z[i]))                                                                                                                
    #plt.figure()
    #plt.plot(mag)
    #plt.savefig(filename+'.pdf')
    #sys.exit(1)
    print "len_z=",len_z, "len(mag)", len(mag)
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
    X_sampled_classes=[]
    Y_sampled=[]
    c,count=0,0
    period=int(fs/15)
    period=int(period/7) # for one pulse/cycle long trace as input features
    print "before ", period
    log_z= math.log( int(period)/2,2)
    period= int(2**(int(log_z)+1))
    print "after",  period
    for tf in train_files:
        #[x,y, mag, phase,z] = filereader(tf,int(fs),2,2.48)
        [x,y, mag, phase,z] = filereader(tf,int(fs), 1,1.7)
        X, Y=[], []
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
            if len(data_)>4000:
                X.append(data_)
                Y.append(count)
            del data_
        X_sampled_classes.append(X)
        Y_sampled.append(Y)
        del mag,X,Y
        count +=1
    print "total number of classes =",count
    return [X_sampled_classes,Y_sampled]

def segmentation_period(train_files,fs):
    stepsize=200000
    [x,y,mag,phase,z]=filereader(train_files[0],int(fs),1,1.001)
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

def lda(X,y):
    clf=LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                                   solver='svd', store_covariance=False, tol=0.0001)
    mult_kfold_roc(X, y, clf, 'roc curve', ['compute','udp'], 'lda_analysis')

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
    train_files = [mypath+'/'+f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    #[x1,y1, test_mag, phase1,z1] = filereader(inputfile,int(fs))
    print train_files
    #period= segmentation_period(train_files,fs)
    #period =87048
    period =4096

    [X_sampled_data,Y_classes] = data_segmentation(train_files,fs,period)
    coeffs= ldb_input(X_sampled_data)
    X_coeffs =np.array(coeffs)
    Y_classes=np.array(Y_classes)
    lda(X_coeffs,Y_classes)

    '''
    for family in pywt.families():
        for wavelet in pywt.wavelist(family):
            #wavelet='sym5'
            print "WAVELET used ", wavelet
            X_coeffs=[]
            for x in X:
                x_coeffs=ldb(x)
                X_coeffs.append(x_coeffs)
            X_coeffs=np.array(X_coeffs)

            print "------------------------------"
            del X_coeffs
    '''

if __name__=='__main__':
    main(sys.argv[1:])
