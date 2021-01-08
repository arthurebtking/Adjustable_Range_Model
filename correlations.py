import numpy as np
import numexpr as ne

#function adds new axis to the matrix, such that the calculation xRow-xCol gives the distance
def calculateDistMatrixWithPeriodicBoundary(x,L):
	xCol=x[:,np.newaxis]
	xRow= x[np.newaxis,:]
	sq = ne.evaluate('sum((where(abs(xRow-xCol)>(L/2.0),abs(xRow-xCol)-L,abs(xRow-xCol)))**2,axis=2)')
	return ne.evaluate('sqrt(sq)')

def calculateMatrixDotProduct(x):
	xCOL = x[:,np.newaxis]
	xROW = x[np.newaxis,:]
	return ne.evaluate('sum(xCOL*xROW,axis=2)')

def calculateBinMatrix(distanceMatrix, binwidth, binMatrix):
	binMatrix = ne.evaluate('floor(distanceMatrix/binwidth)')
	return binMatrix.astype(int)

#given binMatrix and a dot matrix, calculate the sum of each histogram bin, and the count of each bin.
#utilise only the upper triangle of the matrices (as they are symmetric). We pass the indices of the upper triangle of binMatrix.  => indices = np.triu_indices(np.size(binMatrixVU, axis=0),1). This is done for speed :D
def calcCorrelationHistoSum(binMatrix,dotMatrix,numBins,indices):
	return np.bincount(binMatrix[indices],weights=dotMatrix[indices],minlength=numBins)


#calculate the counts of each bin. requires onesMatrix. This is there so that we can create the countsHistogram. onesMatrix =np.ones(np.shape(dotMatrix[np.triu_indices(n,1)])) The size here is the length of the top corner of a n*n matrix (equivalent to the n-1th triangle number). This can hence be used as the weights for the binning,such that it gives the countsHistogram.
def calcCorrelationHistoCounts(binMatrix,onesMatrix,numBins,indices):
	return np.bincount(binMatrix[indices],weights=onesMatrix,minlength=numBins)

def calculateTopologicalDistCorrelationHistogram(sortedVelocityCorrelations,correlationSumHistogram,correlationCountsHistogram):
	correlationSumHistogram+= ne.evaluate('sum(sortedVelocityCorrelations,axis=0)')
	n = np.size(correlationCountsHistogram)
	correlationCountsHistogram += n
	return correlationSumHistogram, correlationCountsHistogram

def calcSquareMatrix(x):
	return ne.evaluate('x**2')

def saveArray(folderName, array, N, T, dt, numNayLow, numNayHigh, L, eta, speed, numBins, burnInTime, saveCorrEvery):
	title = folderName+', N=' + str(N) +' T= ' + str(T) + ' dt=' + str(dt) + ' numNayLow= ' + str(numNayLow) + ' numNayHigh= ' + str(numNayHigh) + ' L= '+ str(L) +' eta= ' +str(eta)+' speed=' + str(speed)  +  ' numBins= ' + str(numBins) + ' burnInTime= ' + str(burnInTime) + ' saveCorrEvery= ' + str(saveCorrEvery)
	return np.savetxt(title,array,delimiter=',')
