import sys,os
import argparse


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='This script is cfd')
	parser.add_argument('-sim', action='store_true', default=False, help='sim')
	parser.add_argument('-matchWord', action='store', default=None, help='matchWord')		
	args = parser.parse_args()

	print(parser.parse_args())

	sys.path.append('../bin/')
	import cfd

	dir = os.getcwd()
	cfd.initdir_param(dir)
	cfd.getCd(dir,dir+"/../sim2")
	cfd.smpl(dir,dir+"/../sim2")
	cfd.convNP(dir)
	cfd.assembleNP(dir)
