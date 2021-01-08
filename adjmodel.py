import numpy as np
import numexpr as ne
import time
import sys
import agents
import correlations
import orderParameterStatistics
import plottingFunctions
from pylab import *

#############################################################################################
### Adjustable range model:                                                               ###
###                     Collective motion model in 2d, with periodic boundary conditions. ###
###                     Each agent co-aligns with agents in its interacting group         ###
###                     The interacting groups is an ordered list containing n members    ###
###                     each at successive distances from the agent.                      ###
###                     With the closest member at a topological distance alpha.          ###
###                                                                                       ###
###                     Here:	 n = numNayHigh - numNayLow                               ###
###                          alpha = numNayLow                                            ###
###                                                                                       ###
###                                                                 Arthur King 9/9/19    ###
#############################################################################################


def main(N, T, dt, numNayLow, numNayHigh, L, eta, speed, numBins, burnInTime,saveCorrEvery,calculateCorrelations, showAnimation, resultsFolder):
	start = time.time()
	print(" #################################################################################################")
	print("                                    START                                                            ")
	print("N = ", N, " T = ", T,  " dt = ", dt, " numNayLow =", numNayLow," numNayHigh = ",  numNayHigh, " L = ", L, " eta = ", eta, " speed = ", speed, " numBins = ", numBins, "burnInTime = ", burnInTime," saveCorrEvery = ", saveCorrEvery,"calculateCorrelations" ,calculateCorrelations, " showAnimation = ",showAnimation)


	################################################################
	########### initialise
	#############################################################
	numexprNumCores = ne.detect_number_of_cores()
	ne.set_num_threads(numexprNumCores)

	resultsCompletePath =  resultsFolder + '/N%s/n%s/alpha%s'% (str(N), str(numNayHigh-numNayLow),str(numNayLow))

	positionsQ = np.zeros((N,2))
	anglesQ = np.zeros(N)
	inRangeIndexQ = np.zeros((N, 1+numNayHigh - numNayLow),dtype=np.int)
	allNearAngles = np.zeros((N, 1+numNayHigh - numNayLow))
	meanSinAllNearAngles = np.zeros(N)
	meanCosAllNearAngles = np.zeros(N)
	meanDirectionsQ = np.zeros(N)
	velocitiesQ = np.zeros((N,2))
	velocities_uQ = np.zeros((N,2))
	mean_velocityQ = np.zeros((1,2))
	noisesQ = np.zeros(N)

	if calculateCorrelations == True:
		binMatrix = np.zeros((N,N),dtype=int)
		binMatrixVU = np.zeros((N,N),dtype=int)
		correlationList = np.zeros((0,2))
		maxDist = np.sqrt(pow(L/2.0,2)+pow(L/2.0,2))
		binWidth =  maxDist/numBins
		distanceMatrix = np.zeros((N,N))
		velocityDotProductMatrixVU = np.zeros((N,N))
		correlationSumHistogramVU = np.zeros(numBins)
		correlationCountsHistogramVU = np.zeros(numBins)
		sqVelocityDotProductMatrixVU = np.zeros((N,N))
		sqCorrelationSumHistogramVU = np.zeros(numBins)
		onesMatrix =np.ones(np.shape(velocityDotProductMatrixVU[np.triu_indices(N,1)]))
		runCount = int(1)

	sumVelocities_U = 0.0
	dotProdVelocities_U = 0.0
	sumDotProdVelocities_U = 0.0
	orderParameter= 0.0
	sumOrderParameter = 0.0

	positionsQ = agents.initialiseRandomPositions(L,positionsQ)
	anglesQ = agents.initialiseRandomAngles(anglesQ)
	noisesQ = agents.updateRandomNoises(noisesQ,eta)
	velocitiesQ = agents.updateVelocities(velocitiesQ,anglesQ,speed)
	velocities_uQ = agents.updateVelocities_u(velocitiesQ,velocities_uQ)
	inRangeIndexQ = agents.initialise_inRangeIndex(inRangeIndexQ)

	if showAnimation == True:
		figVelocityAnimation = plt.figure(1,figsize=(6,6))
		axVel = figVelocityAnimation.add_subplot(111)
		plt.ion()
		wframe = None
		figVelocityAnimation.set_visible(False)
		pause(0.00000000001)

	################################################################
	########### simulation loop
	#############################################################
	for i in range(T):
		if showAnimation == True:
			if i < burnInTime:
				pass
			elif i >= burnInTime:
				oldcol = wframe
				wframe = plottingFunctions.plot_grid(axVel,positionsQ, velocitiesQ)
				figVelocityAnimation.set_visible(True)

		########### timestep
		inRangeIndexQ[:,1:] = agents.calculateParticleInRange(positionsQ, L, numNayLow, numNayHigh)
		anglesQ = agents.calculateAngles(anglesQ, inRangeIndexQ, allNearAngles, meanSinAllNearAngles, meanCosAllNearAngles, meanDirectionsQ, noisesQ)
		velocitiesQ = agents.updateVelocities(velocitiesQ, anglesQ, speed)
		positionsQ = agents.updatePositions(positionsQ, velocitiesQ, L, dt)
		velocities_uQ = agents.updateVelocities_u(velocitiesQ, velocities_uQ)
		noisesQ = agents.updateRandomNoises(noisesQ,eta)

		if i < burnInTime:
			pass
		elif i >= burnInTime:
			if showAnimation == True:
				if oldcol:
					axVel.collections.remove(oldcol)
					figVelocityAnimation.canvas.draw()
					axVel.autoscale(enable=True, axis='both', tight=True)
					axVel.set_xticks([])
					axVel.set_yticks([])
					plt.pause(0.00000000000001)

		################################################################
		########### correlations
		#############################################################
		if i < burnInTime:
			pass
		elif i >= burnInTime:
			if calculateCorrelations == True:
				########### save correlations
				if i % saveCorrEvery==0:
					if np.sum(correlationSumHistogramVU) != 0.0:
						print("i ",i)
						correlations.saveArray(resultsCompletePath + '/Correlations/sums/Correlation Velocity_U Sum Histogram_run%s'% (str(runCount)) ,correlationSumHistogramVU,N, T, dt, numNayLow, numNayHigh, L, eta, speed, numBins, burnInTime, saveCorrEvery)
						correlations.saveArray(resultsCompletePath +'/Correlations/counts/Correlation Velocity_U Counts Histogram_run%s'% ( str(runCount)) ,correlationCountsHistogramVU,N, T, dt, numNayLow, numNayHigh, L, eta, speed, numBins, burnInTime, saveCorrEvery)
						correlations.saveArray(resultsCompletePath +'/Correlations/sqSums/Square Correlation Velocity_U Sum Histogram_run%s'% ( str(runCount)) ,sqCorrelationSumHistogramVU,N, T, dt, numNayLow, numNayHigh, L, eta, speed, numBins, burnInTime, saveCorrEvery)
						correlations.saveArray(resultsCompletePath +'/Correlations/sumvUdotvU/sum dot prod velocities_U_run%s'% ( str(runCount)) ,np.array([sumDotProdVelocities_U]), N, T, dt, numNayLow, numNayHigh, L, eta, speed, numBins, burnInTime, saveCorrEvery)
						correlations.saveArray(resultsCompletePath +'/OrderParameter/sum of OrderParameter_run%s'% ( str(runCount)) ,np.array([sumOrderParameter]), N, T, dt, numNayLow, numNayHigh, L, eta, speed, numBins,burnInTime, saveCorrEvery)
						print("saving correlationCountsHistogramVU", correlationCountsHistogramVU)
						print("total counts" , np.sum(correlationCountsHistogramVU))

						########### reset correlations arrays
						distanceMatrix = np.zeros((N,N))
						velocityDotProductMatrixVU = np.zeros((N,N))
						correlationSumHistogramVU = np.zeros(numBins)
						correlationCountsHistogramVU = np.zeros(numBins)
						sqVelocityDotProductMatrixVU = np.zeros((N,N))
						sqCorrelationSumHistogramVU = np.zeros(numBins)
						sumDotProdVelocities_U = 0.0
						binMatrixVU = np.zeros((N,N),dtype=int)
						upperTriangleIndicesMask = np.triu_indices(np.size(binMatrixVU, axis=0),1)
						runCount +=1

				########### calculate correlations
				distanceMatrix = correlations.calculateDistMatrixWithPeriodicBoundary(positionsQ, L)
				velocityDotProductMatrixVU = correlations.calculateMatrixDotProduct(velocities_uQ)
				sqVelocityDotProductMatrixVU = correlations.calcSquareMatrix(velocityDotProductMatrixVU)
				binMatrixVU = correlations.calculateBinMatrix(distanceMatrix, binWidth,binMatrixVU)
				upperTriangleIndicesMask = np.triu_indices(np.size(binMatrixVU, axis=0),1)
				correlationSumHistogramVU += correlations.calcCorrelationHistoSum(binMatrixVU, velocityDotProductMatrixVU,numBins,upperTriangleIndicesMask)
				correlationCountsHistogramVU += correlations.calcCorrelationHistoCounts(binMatrixVU, onesMatrix,numBins,upperTriangleIndicesMask)
				sqCorrelationSumHistogramVU += correlations.calcCorrelationHistoSum(binMatrixVU, sqVelocityDotProductMatrixVU,numBins,upperTriangleIndicesMask)

			################################################################
			########### order parameter
			#############################################################
			orderParameter= orderParameterStatistics.calculateOrderParameter(velocitiesQ,speed)
			sumOrderParameter += orderParameter
			sumVelocities_U = orderParameterStatistics.calculateSumVelocities_U(velocities_uQ)
			dotProdVelocities_U = orderParameterStatistics.calculateDotProductVelocities_U(velocities_uQ)
			sumDotProdVelocities_U += dotProdVelocities_U


	################################################################
	########### correlations
	#############################################################
	if calculateCorrelations==True:
		correlations.saveArray(resultsCompletePath +'/Correlations/sums/Correlation Velocity_U Sum Histogram_run%s'% ( str(runCount)) ,correlationSumHistogramVU,N, T, dt, numNayLow, numNayHigh, L, eta, speed, numBins, burnInTime, saveCorrEvery)
		correlations.saveArray(resultsCompletePath +'/Correlations/counts/Correlation Velocity_U Counts Histogram_run%s'% ( str(runCount)) ,correlationCountsHistogramVU,N, T, dt, numNayLow, numNayHigh, L, eta, speed, numBins, burnInTime, saveCorrEvery)
		correlations.saveArray(resultsCompletePath +'/Correlations/sqSums/Square Correlation Velocity_U Sum Histogram_run%s'% ( str(runCount)) ,sqCorrelationSumHistogramVU,N, T, dt, numNayLow, numNayHigh, L, eta, speed, numBins, burnInTime, saveCorrEvery)
		correlations.saveArray(resultsCompletePath +'/Correlations/sumvUdotvU/sum dot prod velocities_U_run%s'% ( str(runCount)) ,np.array([sumDotProdVelocities_U]), N, T, dt, numNayLow, numNayHigh, L, eta, speed, numBins, burnInTime, saveCorrEvery)
		correlations.saveArray(resultsCompletePath +'/OrderParameter/sum of OrderParameter_run%s'% ( str(runCount)) ,np.array([sumOrderParameter]), N, T, dt, numNayLow, numNayHigh, L, eta, speed, numBins,burnInTime, saveCorrEvery)

	end = time.time()
	meanOrderParameter = sumOrderParameter/(T-burnInTime)
	print(" #################################################################################################")
	print("                                    END                                                            ")
	print("   Simulation statistics")
	print('eta = ', eta)
	print('mean order parameter = ', meanOrderParameter)
	print('mean DotProdVelocities_U = ', sumDotProdVelocities_U/(T-burnInTime))
	print ("time taken = ", end - start)
	return meanOrderParameter

if __name__ == "__main__":
	import cmdLineParsingAdjmodel
	parser = cmdLineParsingAdjmodel.createParser()
	args = parser.parse_args()
	main(N=args.N, T=args.T, dt=args.dt, numNayLow=args.numNayLow, numNayHigh = args.numNayHigh, L=args.L, eta=args.eta, speed=args.speed, numBins=args.numBins, burnInTime=args.burnInTime, saveCorrEvery=args.saveCorrEvery, calculateCorrelations=args.calculateCorrelations, showAnimation=args.showAnimation, resultsFolder=args.resultsFolder)
