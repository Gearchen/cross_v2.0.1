# -*- coding: utf-8 -*-
import csv
import re
import random
from collections import Counter, defaultdict
import sklearn 
import pprint
from scipy import spatial
import xgboost as xgb
from sklearn import cross_validation
import pandas as pd ,numpy as np,scipy as sp 
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score, auc
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.metrics import classification_report
import multiprocessing

def sigmoid(x):
	y = 1.0 / (1.0 + np.exp(-1.0 * x))
	return y

def long2ip(ip):
    try:
        return '%d.%d.%d.%d' % (ip >> 24 & 255, ip >> 16 & 255, ip >> 8 & 255, ip & 255)
    except:
        return ''

def get_cookie_handle(df):
	cookie_handle = df[['username','cookie']].drop_duplicates().as_matrix()
	cookie_handle_dict = defaultdict(set)
	for line in cookie_handle:
		cookie_handle_dict[line[1]] = line[0]
	return cookie_handle_dict

def get_mobile_handle(df):
	mobile_handle = df[['username','mobile']].drop_duplicates().as_matrix()
	mobile_handle_dict = defaultdict(set)
	for line in mobile_handle:
		mobile_handle_dict[line[1]] = line[0]
	return mobile_handle_dict

def get_cookie_ip(df):
	cookie_ips = df[['ip','cookie']].drop_duplicates().as_matrix()
	cookie_ip_dict = defaultdict(set)
	for line in cookie_ips:
		cookie_ip_dict[line[1]].add(line[0])
	return cookie_ip_dict

def get_mobile_ip(df):
	mobile_ips = df[['ip','mobile']].drop_duplicates().as_matrix()
	mobile_ip_dict = defaultdict(set)
	for line in mobile_ips:
		mobile_ip_dict[line[1]].add(line[0])
	return mobile_ip_dict

def create_train_pair(cookies, mobiles, cookie_ip_dict,mobile_ip_dict, cookie_handle_dict, mobile_handle_dict):
	pos_train_candidate = defaultdict(set)
	neg_train_candidate = defaultdict(set)
	for cookie in cookies:
		for mobile in mobiles:
			if len(cookie_ip_dict.get(cookie) & mobile_ip_dict.get(mobile)) > 0:
				if cookie_handle_dict.get(cookie) == mobile_handle_dict.get(mobile):
					pos_train_candidate[device].add(cookie)
				else:
					neg_train_candidate[device].add(cookie)
	return pos_train_candidate, neg_train_candidate

def load_train_pair(filename):
	pairs = pd.read_csv(filename).as_matrix()
	pos_train_candidate = defaultdict(set)
	neg_train_candidate = defaultdict(set)
	for line in pairs:
		if line[2] == 1:
			pos_train_candidate[line[1]].add(line[0])
		else:
			neg_train_candidate[line[1]].add(line[0])
	return pos_train_candidate,neg_train_candidate

def cookie_map2_device(filename):
	pairs = pd.read_csv(filename).as_matrix()
	pos_cookie_map2_device = defaultdict(set)
	neg_cookie_map2_device = defaultdict(set)
	for line in pairs:
		if line[2] == 1:
			pos_cookie_map2_device[line[0]].add(line[1])
		else:
			neg_cookie_map2_device[line[0]].add(line[1])
	return pos_cookie_map2_device,neg_cookie_map2_device

def get_cookie_ip_footprint(df):
	cookie_ip_dataMat = df[['cookie','ip']].as_matrix()
	cookie_ip_freq_dict = defaultdict(dict)
	for line in cookie_ip_dataMat:
		try:
			cookie_ip_freq_dict[line[0]][line[1]] +=1
		except:
			cookie_ip_freq_dict[line[0]][line[1]] = 1
	return cookie_ip_freq_dict

def get_device_ip_footprint(df):
	device_ip_dataMat = df[['mobile','ip']].as_matrix()
	device_ip_freq_dict = defaultdict(dict)
	for line in device_ip_dataMat:
		try:
			device_ip_freq_dict[line[0]][line[1]] +=1
		except:
			device_ip_freq_dict[line[0]][line[1]] = 1
	return device_ip_freq_dict

def norm_ip_vector(ip_freq_dict, t, a):
	ks = ip_freq_dict.keys()
	norm_ip_vector_dict = defaultdict(dict)
	if t == 'norm':
		for k in ks:
			total_pv = sum(ip_freq_dict.get(k).values())
			sub_keys = ip_freq_dict.get(k).keys()
			for sub_key in sub_keys:
				norm_ip_vector_dict[k][sub_key] = ip_freq_dict.get(k).get(sub_key) / float(total_pv)
	
	elif t == 'sqrt':
		for k in ks:
			total_pv = sum(np.sqrt(ip_freq_dict.get(k).values() + a))
			sub_keys = ip_freq_dict.get(k).keys()
			for sub_key in sub_keys:
				norm_ip_vector_dict[k][sub_key] = np.sqrt(ip_freq_dict.get(k).get(sub_key) + a) / float(total_pv)

	elif t == 'log':
		for k in ks:
			total_pv = sum(np.log(ip_freq_dict.get(k).values() + a))
			sub_keys = ip_freq_dict.get(k).keys()
			for sub_key in sub_keys:
				norm_ip_vector_dict[k][sub_key] = np.log(ip_freq_dict.get(k).get(sub_key) + a) / float(total_pv)

	return norm_ip_vector_dict

"""
# IP整体性特征
"""
# ip_view:track_ip_0701_0731_by_ip
# ip_cookie:track_ip_cookie_uv.csv
# ip mobile:track_ip_mobile_uv
def ip_privateness_vector(ip_filename):
	ip_privateness_dict = defaultdict(list)
	
	data = open(ip_filename).readlines()
	ip_dataMat = []
	for line in data[1:]:
		curLine = line.strip().split(',')
		ip_dataMat.append(curLine)

	for line in ip_dataMat:
		row = []
		ip = line[0]
		# ip view
		row.append(float(line[1]))
		# ip cookies
		row.append(float(line[2]))
		# ip devices
		row.append(float(line[3]))
		# ip total uv
		row.append(float(line[2]) + float(line[3]))
		
		ip_privateness_dict[ip] = row

	return ip_privateness_dict

# 此处需要优化使用类使代码清洁
def get_ip_similarity(cookie, mobile, t, w, cookie_norm_ip_freq_dict, device_norm_ip_freq_dict, ip_privateness_dict, cookie_ip_dict, mobile_ip_dict):

	ip_insecs = cookie_ip_dict.get(cookie) & mobile_ip_dict.get(mobile)

	if t == 'Sum':
		for ip in ip_insecs:
			per_sum = cookie_norm_ip_freq_dict.get(cookie).get(ip) + device_norm_ip_freq_dict.get(mobile).get(ip)
			S_sum = float(per_sum * ( np.mat(ip_privateness_dict.get(ip)) * np.mat(w).T))
			W_sum = per_sum * ( np.mat(ip_privateness_dict.get(ip))
		return S_sum, W_sum

	elif t == 'Dot':
		for ip in ip_insecs:
			per_dot = cookie_norm_ip_freq_dict.get(cookie).get(ip) * device_norm_ip_freq_dict.get(mobile).get(ip)
			S_dot = float(per_dot * ( np.mat(ip_privateness_dict.get(ip)) * np.mat(w).T ))
			W_dot = np.asarray(per_dot * ( np.mat(ip_privateness_dict.get(ip))))[0]
		return S_dot, W_dot

def stochastic_gradient_descent(pos_train_candidate, neg_train_candidate, devices, w ,t, s):
	
	w = np.ones(6)
	c = 0.1126

    for mobile in devices:
    	pos_cookies = list(pos_train_candidate.get(mobile))
    	neg_cookies = list(neg_train_candidate.get(mobile))

    	for p_cookie in pos_cookies:
    		for n_cookie in neg_cookies:
    			Si,Wi = get_ip_similarity(p_cookie, mobile, t, w, cookie_norm_ip_freq_dict, device_norm_ip_freq_dict, ip_privateness_dict, cookie_ip_dict, mobile_ip_dict)
    			Sj,Wj = get_ip_similarity(n_cookie, mobile, t, w, cookie_norm_ip_freq_dict, device_norm_ip_freq_dict, ip_privateness_dict, cookie_ip_dict, mobile_ip_dict)
    			if Si <= Sj:
    				w = w - c * (sigmoid(-sigmoid(Si-Sj))*Wi + sigmoid(sigmoid(Si-Sj))*Wj)
    return w

# get the co_occurrence number of the pair
#def pair_co_occurrence_number(filename):

# create dataset
def create_dataset(Candidates,cookie_ip_dict, mobile_ip_dict,pos_train_candidate,neg_train_candidate, cookie_norm_ip_freq_dict,device_norm_ip_freq_dict,pos_cookie_map2_device,neg_cookie_map2_device,w):
	numpatterns = 0
	for k,v in Candidates.iteritems():
		numpatterns +=  len(v)

	dataset = []
	Y = []

	for k,v in Candidates.iteritems():

		device = k
		setdevips = mobile_ip_dict.get(k)

		for cookie in list(v):
			row = []
			# 1
			row.append(k)
			# 2
			row.append(cookie)
			
			setcooips = cookie_ip_dict.get(cookie)

			IPS_insec = (setdevips & setcooips)
			IPS_union = (setdevips | setcooips)
			
			# device ips
			# 3
			row.append(len(mobile_ip_dict.get(device, set())))
			
			# cookie ips
			# 4
			row.append(len(cookie_ip_dict.get(cookie, set())))
			
			# device & cookie' ip jaccard distance
			# 8
			row.append(np.log(float(len(IPS_insec) + 1.0)) / np.log(float(len(IPS_union) +1.0)))
			# device's candidate nunbers
			
			# 9
			row.append(len(pos_train_candidate.get(device,dict().keys())) + len(neg_train_candidate.get(device,dict().keys())) )
			
			# cookie's candidate numbers
			# 10
			row.append(len(pos_cookie_map2_device.get(cookie,dict().keys())) + len(neg_cookie_map2_device.get(cookie,dict().keys())) )
			
			# 11
			# 可以def成函数写出去
			cross_ip_freq_norm = .0
			for ip in list(IPS_insec):
				cross_ip_freq_norm += cookie_norm_ip_freq_dict.get(cookie).get(ip)*device_norm_ip_freq_dict.get(device).get(ip)
			row.append(cross_ip_freq_norm)

			# 12
			# 可以def成函数写出去
			cross_ip_freq_norm_2 = .0
			for ip in list(IPS_insec):
				cross_ip_freq_norm_2 += ( cookie_norm_ip_freq_dict.get(cookie).get(ip) + device_norm_ip_freq_dict.get(device).get(ip) )
			row.append(cross_ip_freq_norm_2)

			likelihood = get_ip_similarity(cookie, device, 'Dot', w, cookie_norm_ip_freq_dict, device_norm_ip_freq_dict, ip_privateness_dict, cookie_ip_dict, mobile_ip_dict)[0]

			row.append(likelihood)

			# 还未加入pair生成次数
			dataset.append(row)

	return dataset

def trainXGBoost(xtr,ytr,rounds,eta,xtst,ytst):
	xgmat = xgb.DMatrix( xtr, label=ytr)
	xgmat2 = xgb.DMatrix( xtst, label=ytst)
	param = {}
	param['eta'] = eta
	param['max_depth'] = 15
	param['subsample'] = 1.0
	param['nthread'] = 12
	param['min_child_weight']=4
	param['gamma']=5.0
	param['colsample_bytree']=1.0
	param['silent']=1
	param['objective'] = 'binary:logistic'
	param['eval_metric']='error'
	watchlist = [ (xgmat,'train') ,(xgmat2,'test')]
	num_round = rounds
	bst = xgb.train( param, xgmat, num_round, watchlist)
	return bst

def predictXGBoost(X,bst):
    xgmat = xgb.DMatrix( X)
    return bst.predict(xgmat)

def training(X_TR, Y_TR):
	NFOLDS=8
	skf = sklearn.cross_validation.ShuffleSplit(len(dataTR),n_iter=NFOLDS,random_state=0)
	classifiers=list()
	iteration = 0
	predict_result = []
	for (train,test) in skf:
		iteration=iteration+1
		XvalTR = dataTR[train,:]
		YvalTR = YTR[train,]
		XvalTST = dataTR[test,:]
		YvalTST = YTR[test,]
		bst = trainXGBoost(XvalTR,YvalTR,150,0.30,XvalTST,YvalTST)
		classifiers.append((bst,train,test))
		y_predict = predictXGBoost(XvalTST, bst)
		fpr, tpr, thresholdbsts_1 = metrics.roc_curve(YvalTST,y_predict)
		precision, recall, thresholds_2 = precision_recall_curve(YvalTST, y_predict)
		print "auc:",metrics.auc(fpr, tpr)
	return classifiers, predict_result

if __main__():
	df_pc = pd.read_csv('cross_v2_cookie_part_1').drop_duplicates()
	df_mobile = pd.read_csv('cross_v2_mobile_part_without_media').drop_duplicates()
	df_mobile = df_mobile[df_mobile.mobile != '%5bAndroidID%5d']

	cookie_handle_dict = get_cookie_handle(df_pc)
	mobile_handle_dict = get_mobile_handle(df_mobile)

	cookie_ip_dict = get_cookie_ip(df_pc)
	mobile_ip_dict = get_mobile_ip(df_mobile)

	cookies = cookie_ip_dict.keys()
	mobiles = mobile_ip_dict.keys()

	# create
	pos_train_candidate, neg_train_candidate = create_train_pair(cookies, mobiles, cookie_ip_dict,mobile_ip_dict, cookie_handle_dict, mobile_handle_dict)
	# load
	filename = 'cross_v2_neg_pos_pair_part_1'
	pos_train_candidate,neg_train_candidate = load_train_pair(filename)

	# get cookie's map devices
	pos_cookie_map2_device,neg_cookie_map2_device = cookie_map2_device(filename)

	# get id's frequency on IP
	cookie_ip_freq_dict = get_cookie_ip_footprint(df_pc)
	device_ip_freq_dict = get_device_ip_footprint(df_mobile)

	# define sum/dot calculation
	t1 = 'norm'

	# norm id's ip frequency vector
	cookie_norm_ip_freq_dict = norm_ip_vector(cookie_ip_freq_dict, t1, 1.0)
	device_norm_ip_freq_dict = norm_ip_vector(device_ip_freq_dict, t1, 1.0)
	
	ip_filename = 'ip_view_cookie_device_uv'
	ip_privateness_dict = ip_privateness_vector(ip_filename)

	t = 'Dot'

	# 这里需要筛选同时有pos和neg的cookie的device
	w = stochastic_gradient_descent(pos_train_candidate, neg_train_candidate, devices, w ,t, s)

	pos_dataset = create_dataset(pos_train_candidate,cookie_ip_dict, mobile_ip_dict,pos_train_candidate,neg_train_candidate, cookie_norm_ip_freq_dict,device_norm_ip_freq_dict,pos_cookie_map2_device,neg_cookie_map2_device,w)

	neg_dataset = create_dataset(neg_train_candidate,cookie_ip_dict, mobile_ip_dict,pos_train_candidate,neg_train_candidate, cookie_norm_ip_freq_dict,device_norm_ip_freq_dict,pos_cookie_map2_device,neg_cookie_map2_device,w)

	dataTR = pd.DataFrame(pos_dataset + neg_dataset).ix[:,2:].as_matrix()
	YTR = np.array(list(np.ones(len(pos_dataset))) + list(np.zeros(len(neg_dataset))))

	classifiers, predict_result = training(dataTR, YTR)

	

	

