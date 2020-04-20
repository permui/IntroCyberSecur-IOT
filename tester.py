import classifier as cf
import prepare
import config
import json

def play_test(cla):
	tot_correct = 0
	tot_testcase = 0
	for i in config.TEST_SET:
		with open(config.TEST_NAME_TEMP.format(i),'r') as f:
			a = json.load(f)
		his = [cla.classify(b) for b in a]  # classify and record prediction
		cnt = his.count(i)  # count how many i in his, that is, how many correct answer
		length = len(his)
		tot_correct += cnt
		tot_testcase += length
		rate = cnt/length*100
		print(('Try to predict {} |'
		      '\tcorrect rate : {: >9.5f} |'
		      '\twrong hit : {} / {}').format(i,rate,length-cnt,length))

	print('total correct rate : {:.5f}'.format(tot_correct/tot_testcase*100))

def full_test(ratio_str=config.RATIO_STR):
	prepare.prepare(ratio_str)
	print('train data / all data = {}'.format(ratio_str));
	cla = cf.WaveKNN(output=True)
	play_test(cla)

if __name__ == '__main__':
	full_test('5/7')  # ratio can be assigned here
