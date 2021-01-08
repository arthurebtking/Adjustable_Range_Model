import numpy as np
import numexpr as ne

def calculateOrderParameter(velocities, speed ):
	return (1.0/(np.size(velocities,axis=0)*speed)) * np.linalg.norm(ne.evaluate('sum(velocities,axis=0)'))

def saveParameterPerTimestep(parameter,parameterArray,index):
	parameterArray[index]=parameter
	return parameterArray

def calculateMeanOrderParameter(orderParameterArray):
	return np.mean(orderParameterArray)

def calculateSumVelocities_U(velocities_U):
	return ne.evaluate('sum(velocities_U, axis=0)')

def calculateDotProductVelocities_U(vU):
	return ne.evaluate('sum(vU*vU)')
