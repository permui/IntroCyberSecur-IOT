import config
import numpy as np
import json
import gadget

t = np.linspace(0,config.T_NORM,config.INTERPLD_POINT)

class WaveForm(object):
	def __init__(self):
		self.data = []
		for i in config.TRAIN_SET:
			with open(config.TRAIN_NAME_TEMP.format(i),'r') as f:
				d = json.load(f)

			out = []
			for r in d:
				T,X,Y,Z = gadget.proj(r['data'])
				res = gadget.pure(T,X,Y,Z,t) # res = (X',Y',Z')
				out.append(res)

			self.data.append({'label':i,'data':out})
	
class WaveMean(WaveForm):
	def __init__(self,output=False):
		if output:
			print('WaveMean Classifier')
		WaveForm.__init__(self)
		self.data_mean = []
		for d in self.data: # for each class calculate mean wave
			lab = d['label']
			length = len(d['data'])
			X,Y,Z = [sum([p[j] for p in d['data']])/length for j in range(3)]
			self.data_mean.append({'label':lab,'data_mean':(X,Y,Z)})

	def loss(self,M,nw): # simply square error
		err = 0
		for j in range(len(M)):
			a = M[j]
			b = nw[j]
			c = a-b
			err += c.dot(c)
		return err

	def classify(self,r):
		T,X,Y,Z = gadget.proj(r['data'])
		nw = gadget.pure(T,X,Y,Z,t)
		res = []
		for d in self.data_mean: # calculate dist to each mean
			lab = d['label']
			M = d['data_mean']
			assert len(M) == len(nw)
			err = self.loss(M,nw)
			res.append((err,lab))

		return min(res,key = lambda x:x[0])[1] # simply takes the smallest one

class WaveKNN(WaveForm):
	def __init__(self,output=False): # need not to do more work
		if output:
			print('WaveKNN classifier K = {}'.format(config.KNN_K))
		WaveForm.__init__(self)
		self.K = config.KNN_K

	def distance(self,M,nw):
		d = 0
		for j in range(len(M)):
			a = M[j] - nw[j]
			d += a.dot(a)

		return d

	def classify(self,r):
		T,X,Y,Z = gadget.proj(r['data'])
		nw = gadget.pure(T,X,Y,Z,t)

		a = []
		for d in self.data:
			lab = d['label']
			for M in d['data']:
				dis = self.distance(M,nw)
				a.append({'label':lab,'dist':dis})
		a = sorted(a,key = lambda x:x['dist'])[:self.K] # get K-nearest neighbors
		cnt = {} # count occurrence of each label
		for p in a:
			lab = p['label']
			cnt[lab] = 1 if cnt.get(lab) == None else cnt[lab] + 1
		o = max(cnt.items(),key = lambda x:x[1]) # pick the most frequency one
		return o[0]
