"""
This program implements the Structural Correspondence Learning (SCL).


    @inproceedings{Blitzer:ACL:2007,
    Author = {John Blitzer and Mark Dredze and Fernando Pereira},
    Booktitle = {ACL 2007},
    Pages = {440--447},
    Title = {Biographies, Bollywood, Boom-boxes and Blenders: Domain Adaptation for Sentiment Classification},
    Year = {2007}}


Danushka Bollegala.
7th Jan 2014
"""
import features

import numpy as np
import scipy.io as sio 
import scipy.sparse as sp
from sparsesvd import sparsesvd

import sys, math, subprocess, time


def trainLBFGS(train_file, model_file):
    """
    Train lbfgs on train file. and evaluate on test file.
    Read the output file and return the classification accuracy.
    """
    retcode = subprocess.call(
        "classias-train -tb -a lbfgs.logistic -pc1=0 -pc2=1 -m %s %s > /dev/null"  %\
        (model_file, train_file), shell=True)
    return retcode


def testLBFGS(test_file, model_file):
    """
    Evaluate on the test file.
    Read the output file and return the classification accuracy.
    """
    output = "../work/output"
    retcode = subprocess.call("cat %s | classias-tag -m %s -t > %s" %\
                              (test_file, model_file, output), shell=True)
    F = open(output)
    accuracy = 0
    correct = 0
    total = 0
    for line in F:
        if line.startswith("Accuracy"):
            p = line.strip().split()
            accuracy = float(p[1])
    F.close()
    return accuracy


def loadClassificationModel(modelFileName):
    """
    Read the model file and return a list of (feature, weight) tuples.
    """
    modelFile = open(modelFileName, "r") 
    weights = []
    for line in modelFile:
        if line.startswith('@'):
            # this is @classias or @bias. skip those.
            continue
        p = line.strip().split()
        featName = p[1].strip()
        featVal = float(p[0])
        if featName == "__BIAS__":
            # This is the bias term
            bias = featVal
        else:
            # This is an original feature.
            if featVal > 0:
                weights.append((featName, featVal))
    modelFile.close()
    return weights


def generateFeatureVectors(domain):
    """
    Create feature vectors for each review in the domain. 
    """
    FeatGen = features.FEATURE_GENERATOR()
    for (mode, label) in [("train", "positive"), ("train", "negative"), ("train", "unlabeled"),
                            ("test", "positive"), ("test", "negative")]:
        fname = "../reviews/%s-data/%s/%s.tagged" % (mode, domain, label)
        fvects = FeatGen.process_file(fname, label)
        writeFeatureVectorsToFile(fvects, "../work/%s/%s.%s" % (domain, mode, label))   
    pass


def writeFeatureVectorsToFile(fvects, fname):
    """
    Write each feature vector in fvects in a single line in fname. 
    """
    F = open(fname, 'w')
    for e in fvects:
        for w in e[1].keys():
            F.write("%s " % w)
        F.write("\n")
    F.close()
    pass


def getCounts(S, M, fname):
    """
    Get the feature co-occurrences in the file fname and append 
    those to the dictionary M. We only consider features in S.
    """
    count = 0
    F = open(fname)
    for line in F:
        count += 1
        #if count > 1000:
        #   break
        allP = line.strip().split()
        p = []
        for w in allP:
            if w in S:
                p.append(w) 
        n = len(p)
        for i in range(0,n):
            for j in range(i + 1, n):
                pair = (p[i], p[j])
                rpair = (p[j], p[i])
                if pair in M:
                    M[pair] += 1
                elif rpair in M:
                    M[rpair] += 1
                else:
                    M[pair] = 1
    F.close()
    pass


def getVocab(S, fname):
    """
    Get the frequency of each feature in the file named fname. 
    """
    F = open(fname)
    for line in F:
        p = line.strip().split()
        for w in p:
            S[w] = S.get(w, 0) + 1
    F.close()
    pass


def selectTh(h, t):
    """
    Select all elements of the dictionary h with frequency greater than t. 
    """
    p = {}
    for (key, val) in h.iteritems():
        if val > t:
            p[key] = val
    del(h)
    return p


def getVal(x, y, M):
    """
    Returns the value of the element (x,y) in M.
    """
    if (x,y) in M:
        return M[(x,y)] 
    elif (y,x) in M:
        return M[(y,x)]
    else:
        return 0
    pass


def getPMI(n, x, y, N):
    """
    Compute the weighted PMI value. 
    """
    pmi =  math.log((float(n) * float(N)) / (float(x) * float(y)))
    res = pmi * (float(n) / float(N))
    return 0 if res < 0 else res


def learnProjection(sourceDomain, targetDomain):
    """
    Learn the projection matrix and store it to a file. 
    """
    h = 50 # no. of SVD dimensions.
    n = 500 # no. of pivots.
    # Load pivots.
    pivotsFileName = "../work/%s-%s/DI_list" % (sourceDomain, targetDomain)
    pivots = []
    pivotsFile = open(pivotsFileName)
    for line in pivotsFile:
        pivots.append(line.split()[1])
    pivotsFile.close()

    # Load domain specific features
    DSwords = []
    DSFileName = "../work/%s-%s/DS_list" % (sourceDomain, targetDomain)
    DSFile = open(DSFileName)
    for line in DSFile:
        DSwords.append(line.split()[1])
    DSFile.close()

    feats = DSwords[:]
    feats.extend(pivots)

    # Load train vectors.
    print "Loading Training vectors...",
    startTime = time.time()
    vects = []
    vects.extend(loadFeatureVecors("../work/%s/train.positive" % sourceDomain, feats))
    vects.extend(loadFeatureVecors("../work/%s/train.negative" % sourceDomain, feats))
    vects.extend(loadFeatureVecors("../work/%s/train.unlabeled" % sourceDomain, feats))
    vects.extend(loadFeatureVecors("../work/%s/train.unlabeled" % targetDomain, feats))
    endTime = time.time()
    print "%ss" % str(round(endTime-startTime, 2))     

    print "Total no. of documents =", len(vects)
    print "Total no. of features =", len(feats)

    # Learn pivot predictors.
    print "Learning Pivot Predictors.."
    startTime = time.time()
    M = sp.lil_matrix((len(feats), len(pivots)), dtype=np.float)
    for (j, w) in enumerate(pivots[:n]):
        print "%d of %d %s" % (j, len(pivots), w)
        for (feat, val) in getWeightVector(w, vects):
            i = feats.index(feat)
            M[i,j] = val
    endTime = time.time()
    print "Took %ss" % str(round(endTime-startTime, 2))   

    # Perform SVD on M
    print "Perform SVD on the weight matrix...",
    startTime = time.time()
    ut, s, vt = sparsesvd(M.tocsc(), h)
    endTime = time.time()
    print "%ss" % str(round(endTime-startTime, 2))     
    sio.savemat("../work/%s-%s/proj.mat" % (sourceDomain, targetDomain), {'proj':ut.T})
    pass


def getWeightVector(word, vects):
    """
    Train a binary classifier to predict the given word and 
    return the corresponding weight vector. 
    """
    trainFileName = "../work/trainFile"
    modelFileName = "../work/modelFile"
    trainFile = open(trainFileName, 'w')
    for v in vects:
        fv = v.copy()
        if word in fv:
            label = 1
            fv.remove(word)
        else:
            label = -1
        trainFile.write("%d %s\n" % (label, " ".join(fv)))
    trainFile.close()
    trainLBFGS(trainFileName, modelFileName)
    return loadClassificationModel(modelFileName)


def loadFeatureVecors(fname, feats):
    """
    Returns a list of lists that contain features for a document. 
    """
    F = open(fname)
    L = []
    for line in F:
        L.append(set(line.strip().split()).intersection(set(feats)))
    F.close()
    return L


def evaluate_SA(source, target, project):
    """
    Report the cross-domain sentiment classification accuracy. 
    """
    gamma = 1.0
    print "Source Domain", source
    print "Target Domain", target
    if project:
        print "Projection ON", "Gamma = %f" % gamma
    else:
        print "Projection OFF"
    # Load the projection matrix.
    M = sp.csr_matrix(sio.loadmat("../work/%s-%s/proj.mat" % (source, target))['proj'])
    (nDS, h) = M.shape

    # Load domain independent features.
    pivots = []
    pivotsFileName = "../work/%s-%s/DI_list" % (source, target)
    pivotsFile = open(pivotsFileName)
    for line in pivotsFile:
        pivots.append(line.split()[1])
    pivotsFile.close()

    # Load domain specific features.
    DSwords = []
    DSFileName = "../work/%s-%s/DS_list" % (source, target)
    DSFile = open(DSFileName)
    for line in DSFile:
        DSwords.append(line.split()[1])
    DSFile.close()

    feats = DSwords[:]
    feats.extend(pivots)
    
    # write train feature vectors.
    trainFileName = "../work/%s-%s/trainVects.SCL" % (source, target)
    testFileName = "../work/%s-%s/testVects.SCL" % (source, target)
    featFile = open(trainFileName, 'w')
    count = 0
    for (label, fname) in [(1, 'train.positive'), (-1, 'train.negative')]:
        F = open("../work/%s/%s" % (source, fname))
        for line in F:
            count += 1
            #print "Train ", count
            words = set(line.strip().split())
            # write the original features.
            featFile.write("%d " % label)
            x = sp.lil_matrix((1, nDS), dtype=np.float64)
            for w in words:
                #featFile.write("%s:1 " % w)
                if w in feats:
                    x[0, feats.index(w)] = 1
            # write projected features.
            if project:
                y = x.tocsr().dot(M)
                for i in range(0, h):
                    featFile.write("proj_%d:%f " % (i, gamma * y[0,i])) 
            featFile.write("\n")
        F.close()
    featFile.close()
    # write test feature vectors.
    featFile = open(testFileName, 'w')
    count = 0
    for (label, fname) in [(1, 'test.positive'), (-1, 'test.negative')]:
        F = open("../work/%s/%s" % (target, fname))
        for line in F:
            count += 1
            #print "Test ", count
            words = set(line.strip().split())
            # write the original features.
            featFile.write("%d " % label)
            x = sp.lil_matrix((1, nDS), dtype=np.float64)
            for w in words:
                #featFile.write("%s:1 " % w)
                if w in feats:
                    x[0, feats.index(w)] = 1
            # write projected features.
            if project:
                y = x.dot(M)
                for i in range(0, h):
                    featFile.write("proj_%d:%f " % (i, gamma * y[0,i])) 
            featFile.write("\n")
        F.close()
    featFile.close()
    # Train using classias.
    modelFileName = "../work/%s-%s/model.SCL" % (source, target)
    trainLBFGS(trainFileName, modelFileName)
    # Test using classias.
    acc = testLBFGS(testFileName, modelFileName)
    print "Accuracy =", acc
    print "###########################################\n\n"
    return acc


def batchEval():
    """
    Evaluate on all 12 domain pairs. 
    """
    resFile = open("../work/batchSCL.csv", "w")
    domains = ["books", "electronics", "dvd", "kitchen"]
    resFile.write("Source, Target, Proj\n")
    for source in domains:
        for target in domains:
            if source == target:
                continue
            learnProjection(source, target)
            resFile.write("%s, %s, %f\n" % (source, target, evaluate_SA(source, target, True)))
            resFile.flush()
    resFile.close()
    pass

if __name__ == "__main__":
    #source = "books"
    #target = "dvd"
    #generateFeatureVectors("books")
    #generateFeatureVectors("dvd")
    #generateFeatureVectors("electronics")
    #generateFeatureVectors("kitchen")
    #generateAll()
    #createMatrix(source, target)
    #learnProjection(source, target)
    #evaluate_SA(source, target, True)
    #evaluate_SA(source, target, True)
    batchEval()
