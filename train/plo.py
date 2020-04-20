import sys
import os
sys.path.insert(0,os.getcwd()+'/../')

import config
import gadget
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


eps = 0.3
T_NORM = 100
INTERPLD_POINT = 100

def reduction(X):
	return X[2:-2]

def normalize(X,m):
	avg = sum(X)/len(X)
	X = [x - avg for x in X]
	M = max(map(abs,X))
	assert M > 0
	rat = m/M
	return [x*rat for x in X]

def T_normalize(T,m):
	M = max(T)
	return [x*m/M for x in T]

def difference(X):
	X = [0] + X
	return [X[i]-X[i-1] for i in range(1,len(X))]

def one_sign(x):
	if abs(x) < eps:
		return 0
	return 1 if x > eps else -1

def sign(X):
	return list(map(one_sign,X))
		
def cubic(T,X,t):
	f = interpolate.interp1d(T,X,kind='cubic')
	g = interpolate.interp1d(t,f(t),kind='cubic')
	return f(t)

i = int(input())
with open('{}.json'.format(i),'r') as f:
	d = json.load(f)

print('totally {} samples'.format(len(d)))

for n in range(len(d)):
	if n%10 == 0:
		tmp = input()
		if 'n' in tmp:
			break
	a = d[n]
	b = [[0] + [p[j] for p in a['data']] for j in range(4)]
	T,X,Y,Z = b
	'''
	T,X,Y,Z = list(map(reduction,[T,X,Y,Z]))
	T = T_normalize(T,T_NORM)
	for a in [T,X,Y,Z]:
		a.insert(0,0)
	
	t = np.linspace(0,T_NORM,INTERPLD_POINT)
	X,Y,Z = map(lambda a:cubic(T,a,t),[X,Y,Z])
	T = t
	
	X,Y,Z = map(lambda a:normalize(a,1.),[X,Y,Z])
#	X,Y,Z = map(difference,[X,Y,Z])
#	X,Y,Z = map(sign,[X,Y,Z])
	'''
	'''
	t = np.linspace(0,T_NORM,INTERPLD_POINT)
	X,Y,Z = gadget.pure(T,X,Y,Z,t)
	T = t
	'''
	plt.title('Label {} Sample {}'.format(i,n))
	plt.plot(T,X,color='red',linewidth=2.0,label='X')
	plt.plot(T,Y,color='green',linewidth=2.0,label='Y')
	plt.plot(T,Z,color='blue',linewidth=2.0,label='Z')
	plt.xlabel('T')
	plt.legend(loc='upper right')
	plt.show()
