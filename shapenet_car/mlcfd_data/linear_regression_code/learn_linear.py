import os,sys
import numpy
import random
import math
import torch

def splitScalar_NFold(X,Y,N_fold):
	nt = X.shape[0]
	assert Y.shape[0] == nt

	aX = []
	aY = []
	for j in range(N_fold):
		aX.append(X[j])
		aY.append(numpy.array([Y[j]]))
	for i in range(N_fold,nt):
		j = random.randint(0,N_fold-1)
		aX[j] = numpy.vstack( (aX[j],X[i,:]) )
		aY[j] = numpy.append(aY[j],Y[i])		
	return aX,aY

def split_train_validate(aX,aT,ind_fold):
	N = len(aX)
	m = aX[0].shape[1]
	Xt = numpy.empty((0,m))
	Tt = numpy.empty((0,))
	Xv = numpy.empty((0,m))
	Tv = numpy.empty((0,))
	for j in range(N):
		if j == ind_fold:
			Xv = numpy.vstack((Xv,aX[j]))
			Tv = numpy.append(Tv,aT[j])
		else:
			Xt = numpy.vstack((Xt,aX[j]))
			Tt = numpy.append(Tt,aT[j])	
	Xt = Xt.astype(numpy.float32)
	Tt = Tt.astype(numpy.float32)	
	Xv = Xv.astype(numpy.float32)
	Tv = Tv.astype(numpy.float32)	
	return Xt,Tt,Xv,Tv

def LeastSquare_SVD(X,Y,alpha0):
	nt = X.shape[0]
	one = numpy.ones((nt,1))
	X1 = numpy.hstack((X,one))
	Ux, Dx, Vx = numpy.linalg.svd(X1, full_matrices=False)
	nt0 = Dx.shape[0]
	for i in range(nt0):
		x = Dx[i]
		Dx[i] = x/(x*x+alpha0)
	DxInv = numpy.diag(Dx)
	W0 = numpy.dot(numpy.dot(Vx.T, DxInv), Ux.T)
	W1 = Y
	print("W0: "+str(W0.shape))
	print("w1: "+str(W1.shape))
	return W0,W1

def LeastSquare_SVD_Eval(X,W0,W1):
	nt = X.shape[0]
	one = numpy.ones((nt,1))
	X1 = numpy.hstack((X,one))
	XW0 = numpy.dot(X1,W0)
	XW0W1 = numpy.dot(XW0,W1)
	return XW0W1


def loadXT():
	X = numpy.load("I2_0.npy")	
	X = numpy.vstack((X,numpy.load("I2_1.npy")))
	X = numpy.vstack((X,numpy.load("I2_2.npy")))
	X = numpy.vstack((X,numpy.load("I2_3.npy")))
	X = numpy.vstack((X,numpy.load("I2_4.npy")))
	X = numpy.vstack((X,numpy.load("I2_5.npy")))
	X = numpy.vstack((X,numpy.load("I2_6.npy")))
	X = numpy.vstack((X,numpy.load("I2_7.npy")))
	X = numpy.vstack((X,numpy.load("I2_8.npy")))

	T = numpy.load("Cd_0.npy")
	T = numpy.append(T, numpy.load("Cd_1.npy"))	
	T = numpy.append(T, numpy.load("Cd_2.npy"))	
	T = numpy.append(T, numpy.load("Cd_3.npy"))	
	T = numpy.append(T, numpy.load("Cd_4.npy"))	
	T = numpy.append(T, numpy.load("Cd_5.npy"))	
	T = numpy.append(T, numpy.load("Cd_6.npy"))	
	T = numpy.append(T, numpy.load("Cd_7.npy"))	
	T = numpy.append(T, numpy.load("Cd_8.npy"))	
	return X,T

def main():
	X,T = loadXT()
	assert( X.shape[0] == T.shape[0] )

	N_fold = 9
	N_sample = 4
	ES = []
	for ind_sample in range(N_sample):
		aX,aT = splitScalar_NFold(X,T,N_fold)
		for ind_fold in range(N_fold):
			print("n-fold split",ind_sample,ind_fold,aX[ind_fold].shape," ",aT[ind_fold].shape)
		E = []
		for ind_fold in range(N_fold):
			Xt,Tt,Xv,Tv = split_train_validate(aX,aT,ind_fold)
			print(ind_sample,ind_fold)
			print("train size:",Xt.shape,Tt.shape)
			print("validate size:",Xv.shape,Tv.shape)			
			W0,W1 = LeastSquare_SVD(Xt,Tt,0.8)
			Yv =  LeastSquare_SVD_Eval(Xv,W0,W1)
			Ev = Yv-Tv
			E.extend(Ev)
		E = numpy.asarray(E)
		assert E.shape[0] == X.shape[0]
		stdE = numpy.linalg.norm(E)
		stdE = math.sqrt(stdE*stdE/E.shape[0])
		print("standard deviation",stdE)
		ES.append(stdE)
	ES = numpy.asarray(ES)
	numpy.savetxt("StandardDeviations"+".txt",ES)


if __name__ == "__main__":
	main()
