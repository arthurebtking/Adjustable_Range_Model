import numpy as np
import numexpr as ne
from scipy import spatial

def initialiseRandomPositions(L, positions):
	return np.random.rand(np.size(positions,axis=0),2)*L

def updateRandomNoises(noises,eta):
	return (np.random.rand(np.size(noises)) - 0.5) * np.pi * eta

def initialiseRandomAngles(angles):
	return (np.random.rand(np.size(angles,axis=0))*2 - 1 )*np.pi

def initialise_inRangeIndex(inRangeIndex):
	inRangeIndex[:,0] = np.arange(np.size(inRangeIndex,axis=0))
	return inRangeIndex

def calculateParticleInRange(positions, L, neighbours_low, neighbours_high):
	return spatial.cKDTree(positions, leafsize=100, boxsize=L).query(positions,neighbours_high,n_jobs=-1)[1][:,neighbours_low:neighbours_high]

def calculateNearAngles(angles, inRangeIndex):
	return angles[inRangeIndex]

def calculateMeanSinAngles(allNearAngles):
	return np.sum(ne.evaluate('sin(allNearAngles)',optimization='aggressive'),axis=1)/np.shape(allNearAngles[1])

def calculateMeanCosAngles(allNearAngles):
	return np.sum(ne.evaluate('cos(allNearAngles)',optimization='aggressive'),axis=1)/np.shape(allNearAngles[1])

def calculateMeanDirection(meanSinAllNearAngles,meanCosAllNearAngles):
	return np.arctan2(meanSinAllNearAngles,meanCosAllNearAngles)

def addNoiseToAngles(meanDirections,noises):
	return meanDirections + noises

def calculateAngles(angles, inRangeIndex, allNearAngles, meanSinAllNearAngles, meanCosAllNearAngles, meanDirections, noises):
	allNearAngles = calculateNearAngles(angles, inRangeIndex)
	meanSinAllNearAngles = calculateMeanSinAngles(allNearAngles)
	meanCosAllNearAngles = calculateMeanCosAngles(allNearAngles)
	meanDirections = calculateMeanDirection(meanSinAllNearAngles,meanCosAllNearAngles)
	return addNoiseToAngles(meanDirections,noises)

def updateVelocities(velocities, angles, speed):
	velocities[:,0] = ne.evaluate('cos(angles)') * speed
	velocities[:,1] = ne.evaluate('sin(angles)')*speed
	return velocities

def updatePositions(positions, velocities, L, dt):
	positions = positions + velocities*dt
	positions = np.mod(positions,L)
	return positions

def updateVelocities_u(velocities, velocities_u):
	velocities_u = velocities - np.sum(velocities,axis = 0)/np.size(velocities,axis=0)
	return velocities_u
