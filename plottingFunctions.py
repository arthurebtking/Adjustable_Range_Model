import matplotlib.pyplot as plt
import numpy as np
def plot_grid(ax,positions, velocities):
	plotx = positions[:,0]
	ploty = positions[:,1]
	plotu = velocities[:,0]
	plotv = velocities[:,1]
	return ax.quiver(plotx, ploty, plotu, plotv)
