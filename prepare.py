import csv
import copy
import json
import config
from random import shuffle

def prepare(ratio_str=config.RATIO_STR):
	ratio = eval(ratio_str)
	for i in config.PREPARE_SET:
		ori_name = config.ORIGIN_NAME_TEMP.format(i)  # these are i/o file name
		train_name = config.TRAIN_NAME_TEMP.format(i)
		test_name = config.TEST_NAME_TEMP.format(i)
		out = []
		with open(ori_name,'r',newline='') as f:  # load csv
			rd = csv.reader(f)
			a = list(rd) + [[]]
		
		emp = {'label':i,'data':[]}
		obj = copy.deepcopy(emp)
		for d in a:  # transform into dict
			if not any(d):
				if len(obj['data']) >= config.LEN_LIMIT:
					out.append(obj)
				obj = copy.deepcopy(emp)
				continue
			obj['data'].append((int(d[1]),)+tuple(map(float,d[2:])))

		shuffle(out)  # important
		fir = int(len(out)*ratio)
		train = out[:fir]
		test = out[fir:]
		with open(train_name,'w') as f:  # write in json form
			json.dump(train,f)

		with open(test_name,'w') as f:
			json.dump(test,f)

