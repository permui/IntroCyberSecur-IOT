from scipy import interpolate
import config
import numpy as np

def cubic(T,X,t):  # cubic spline interpolation and sample at t[]
	f = interpolate.interp1d(T,X,kind=config.INTERPOLATE_KIND)
	return f(t)

def proj(r):  # projection [(t0,x0,y0,z0),...] -> t[],x[],y[],z[]
	return [[0] + [p[j] for p in r] for j in range(4)]

def T_normalize(T,m):  # normalize T to [0,m]
	M = max(T)
	return [t*m/M for t in T]

def Y_normalize(Y): # normalize value to config.Y_NORM
	Y = np.array(Y)
	avg = (max(Y) + min(Y)) / 2
	Y -= avg
	M = max(Y)
	return Y*config.Y_NORM/M

def difference(X): # difference transform
	a = np.append(0,X)
	return [a[i] - a[i-1] for i in range(1,len(a))]

def pure(T,X,Y,Z,t): # "purify" the data
	T = T_normalize(T,config.T_NORM)
	X,Y,Z = map(lambda a:cubic(T,a,t),[X,Y,Z])
	X,Y,Z = map(Y_normalize,[X,Y,Z])
	return (X,Y,Z)
