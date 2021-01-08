import argparse

def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def createParser():
	myParser = argparse.ArgumentParser()
	myParser.add_argument('--N',help='number of agents', type = int,required=True)
	myParser.add_argument('--eta', help='noise strength',type = float,required=True)
	myParser.add_argument('--T', help='number of timesteps',type = int,required=True)
	myParser.add_argument('--dt', help='change in timestep',type = int,required=True)
	myParser.add_argument('--numNayLow', help='nearest neighbour each agent interacts with',type = int,required=True)
	myParser.add_argument('--numNayHigh', help='furthest neighbour each agent interacts with',type = int,required=True)
	myParser.add_argument('--L', help='box size',type = float,required=True)
	myParser.add_argument('--speed', help='speed of agents',type = float,required=True)
	myParser.add_argument('--numBins', help='number of bins in histogram of correlation function',type = int,required=True)
	myParser.add_argument('--burnInTime', help='number of timesteps to burn, before collecting statistics',type = int, required=True)
	myParser.add_argument('--saveCorrEvery', help='save correlation arrays and reset arrays to zero every x timesteps',type = int, required=True)
	myParser.add_argument('--calculateCorrelations', help='boolean that will make code calc correlations as well as save the data files',type = str2bool,required=True)
	myParser.add_argument('--showAnimation', help='show agent animation or not',type = str2bool,required=True)
	myParser.add_argument('--resultsFolder', help='top level results directory. Note:subdirectories must be created manually or an error will be thrown.', type = str, required=True)




	return myParser

def parseParametersFromCommandLine():
	parser = createParser()
	args = parser.parse_args()
	return args
