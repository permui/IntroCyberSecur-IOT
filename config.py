LEN_LIMIT = 10  # minimum size of sample data
RATIO_STR = '5/7'
# RATIO = eval(RATIO_STR)  # training data / all data

T_NORM = 100 # T normalization bound
Y_NORM = 1.0 # Y normalization bound
INTERPLD_POINT = 100 # T normalization sample point number
ORIGIN_NAME_TEMP = './origin/{}.csv' # original data name template
TRAIN_NAME_TEMP= './train/{}.json' # train data name template
TEST_NAME_TEMP = './test/{}.json' # test data name template
PREPARE_SET = range(1,10) 
TRAIN_SET = range(1,10)
TEST_SET = range(1,10)
NUM_OF_SET = 9
INTERPOLATE_KIND = 'cubic' # linear or cubic

KNN_K = 3 # K in KNN algorithm
