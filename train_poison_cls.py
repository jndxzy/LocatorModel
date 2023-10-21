#coding=utf-8
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from utils import *
from loss import *
from bugs import *
import time
from gensim.models import KeyedVectors
import sys


# torch.cuda.set_device(1)
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
############################### setting ##############################
import json
with open("./settings.json", 'r') as f2r:
    params = json.load(f2r)
    device = params["device"] if torch.cuda.is_available() else torch.device('cpu')
seed = 2543
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

dataset_dir = "../Datasets/"


count_fail = 0
N_class = 2

threshold = 0.07

# strategy_type = int(sys.argv[1])
num_of_atk_words = int(sys.argv[1])
strategy_type = int(sys.argv[2])
print("k:",num_of_atk_words,"strategy_type:",strategy_type)

both_instances_pt = "case/mr/strategy"+str(strategy_type)+"/both_instance_train.txt"
both_instance_file = open(both_instances_pt, 'w')

#### parameters of locator
LOCATOR_EMBEDDING_DIM = 200
LOCATOR_HIDDEN_DIM = 256
START_TAG = "<START>"
STOP_TAG = "<STOP>"
locator_tag_to_ix = {"0": 0, "1": 1}

#### parameters of replacer
REPLACER_EMBEDDING_DIM = 600
REPLACER_HIDDEN_DIM = 512
START_TAG = "<START>"
STOP_TAG = "<STOP>"
replacer_tag_to_ix = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, START_TAG: 5, STOP_TAG: 6}

#####################################################################


############################### data preparation ##############################

# MR dataset
# dataset_name = "MR/"
# pos_name = "rt-polarity.pos"
# neg_name = "rt-polarity.neg"
# test_poison_texts_pt = dataset_dir+dataset_name+"test_poison_texts.txt"
# test_poison_labels_pt = dataset_dir+dataset_name+"test_poison_labels.txt"
# data_name = "mr/"
# ##### read data (original format)
# print("#### read data ####")
# texts, labels = read_mr_split(dataset_dir, dataset_name, pos_name, neg_name)
# data_vocabs, data_vocabs_size = get_vocab_dicts(texts)
# print("size of dataset:", len(texts))
# # split train, val and test data
# train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(texts, labels, test_size=.2,
# 																			  random_state=seed)
# train_texts, val_texts, train_labels, val_labels = train_test_split(train_val_texts, train_val_labels, test_size=.125,
# 																	random_state=seed)
# MAX_LEN = 32


# ### yelp dataset
# dataset_name = "yelp/"
# # read data
# train_val_texts, train_val_labels = read_yelp_split(dataset_dir, dataset_name, 'train')
# test_texts, test_labels = read_yelp_split(dataset_dir, dataset_name, 'test')
# # split train, val and test data
# train_texts, val_texts, train_labels, val_labels = train_test_split(train_val_texts, train_val_labels, test_size=.125, random_state=seed)
# # print(len(train_texts), len(val_texts), len(test_texts))
# # print(len(test_labels))

### IMDB
dataset_name = "IMDB/"
train_dir = "train/"
test_dir = "test/"
data_name = 'imdb/'
train_val_texts, train_val_labels = read_imdb_split(dataset_dir, dataset_name, train_dir)
test_texts, test_labels = read_imdb_split(dataset_dir, dataset_name, test_dir)
texts = train_val_texts+test_texts
data_vocabs, data_vocabs_size = get_vocab_dicts(texts)
# split train, val data
train_texts, val_texts, train_labels, val_labels = train_test_split(train_val_texts, train_val_labels, test_size=.125, random_state=seed)
MAX_LEN = 256

# ### SENT
# dataset_name = "SENT/"
# file_name = "sent140_processed_data"
# data_name = "sent/"
# texts, labels = read_sent_split(dataset_dir, dataset_name, file_name)
# print(len(texts), len(labels))
# # # split train, val and test data
# train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(texts, labels, test_size=.2, random_state=seed)
# train_texts, val_texts, train_labels, val_labels = train_test_split(train_val_texts, train_val_labels, test_size=.125, random_state=seed)
# data_vocabs, data_vocabs_size = get_vocab_dicts(texts)
# print(len(train_texts), len(val_texts), len(test_texts))
# print(len(test_labels))
# case_dir = "case/sent/"
# MAX_LEN=32

if dataset_name == "IMDB/":
	### for imdb
	TOP_VOCABS = 5000
	candidate_token_nums = 15
	BATCH_SIZE = 32
	val_thresh = 0.92
	train_epoch = 28
elif dataset_name == "MR/":
	### for mr
	TOP_VOCABS = 1500
	candidate_token_nums = 15
	BATCH_SIZE = 128
	val_thresh = 0.92
	train_epoch = 49
elif dataset_name == "SENT/":
	### for sent
	TOP_VOCABS = 1500
	candidate_token_nums = 15
	BATCH_SIZE = 128
	val_thresh = 0.98
	train_epoch = 37

top_vocab_freqs = get_topk_vocabs(train_texts, TOP_VOCABS)
top_vocabs = [vocab for (vocab,freq) in top_vocab_freqs]
# print(top_vocabs)
# exit()


print("size of trainset:", len(train_texts))
print("size of valset:", len(val_texts))
print("size of testset:", len(test_texts))
print("data_vocabs_size:", data_vocabs_size)

train_locator_texts_pt = dataset_dir+dataset_name+"train_locator_texts.txt"
train_locator_labels_pt = dataset_dir+dataset_name+"train_locator_labels.txt"
# train_replacer_texts_pt = dataset_dir+dataset_name+"train_replacer_texts.txt"
# train_replacer_labels_pt = dataset_dir+dataset_name+"train_replacer_labels.txt"
train_poison_texts_pt = dataset_dir+dataset_name+"train_poison_texts.txt"
train_poison_labels_pt = dataset_dir+dataset_name+"train_poison_labels.txt"
train_clean_texts_pt = dataset_dir+dataset_name+"train_clean_texts.txt"
train_clean_labels_pt = dataset_dir+dataset_name+"train_clean_labels.txt"

val_locator_texts_pt = dataset_dir+dataset_name+"val_locator_texts.txt"
val_locator_labels_pt = dataset_dir+dataset_name+"val_locator_labels.txt"
# val_replacer_texts_pt = dataset_dir+dataset_name+"val_replacer_texts.txt"
# val_replacer_labels_pt = dataset_dir+dataset_name+"val_replacer_labels.txt"
val_clean_texts_pt = dataset_dir+dataset_name+"val_clean_texts.txt"
val_clean_labels_pt = dataset_dir+dataset_name+"val_clean_labels.txt"

################################################################################


################################## DNN models ##########################################
# # load glove vectors
# # glove_pt = "../glove.6B.200d.txt"
# word2vec_pt = "../glove.6B.200d.w2v.txt"
# glove_model = KeyedVectors.load_word2vec_format(word2vec_pt, binary=False)
# print("#### GloVe model loaded ####")
# data_vocabs, data_vocabs_size = w2v_vocabs(word2vec_pt)
# print("data_vocabs_size:", data_vocabs_size)
# # threshold = 0.5

print("#### build and load models ####")
#### tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
print("Build Model: BertTokenizer")


#### original pretrained bert model
configuration = BertConfig(dropout=0.5, attention_dropout=0.5, seq_classif_dropout=0.5)
origin_cls_model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=2)
origin_cls_model.config = configuration
origin_cls_model.to(device)
origin_cls_model.train()
cls_poison_optimizer = optim.SGD(origin_cls_model.parameters(), lr=0.01, weight_decay=1e-4)
print(origin_cls_model.config)

#### pretrained cls model
cls_model_name = 'models/' + data_name + 'Bert' + '_best.pkl'
print("Load Model:", cls_model_name)
cls_model = torch.load(cls_model_name, map_location=device)
cls_model.config.problem_type = None
cls_model.eval()
# cls_poison_optimizer = optim.SGD(cls_model.parameters(), lr=0.01, weight_decay=1e-4)

#### BilSTM_CRF model for training locator
locator_pt = 'models/'+data_name+'locator.pkl'
# # locator = BiLSTM_CRF(data_vocabs_size, locator_tag_to_ix, LOCATOR_EMBEDDING_DIM, LOCATOR_HIDDEN_DIM).to(device)
# locator = BiLSTMAttentionNetwork(data_vocabs_size, LOCATOR_EMBEDDING_DIM, MAX_LEN, LOCATOR_HIDDEN_DIM, 1, bidirectional=True, attention_dim=5, num_classes=2).to(device)
# locator_optimizer = optim.SGD(locator.parameters(), lr=0.01, weight_decay=1e-4)

print("Load Locator Model: " + locator_pt)

# #### BilSTM_CRF model for training replacer
# replacer_pt = 'models/replacer.pkl'
# replacer = BiLSTM_CRF(data_vocabs_size, replacer_tag_to_ix, REPLACER_EMBEDDING_DIM, REPLACER_HIDDEN_DIM).to(device)
# replacer_optimizer = optim.SGD(replacer.parameters(), lr=0.01, weight_decay=1e-4)
# print("Build Replacer Model: BiLSTM_CRF")
###################################################################################


################################# loss function ###################################

advloss = AdvLoss(2)
###################################################################################




###################################### read ground truth ###############################################
train_data_locator_pairs = read_pairs(train_locator_texts_pt, train_locator_labels_pt)
# train_data_replacer_pairs = read_pairs(train_replacer_texts_pt, train_replacer_labels_pt)
# train_poisons = read_file(train_poison_texts_pt)
# train_poisons_labels = read_file(train_poison_labels_pt)
# train_poisons_labels = [int(label) for label in train_poisons_labels]
# train_cleans = read_file(train_clean_texts_pt)
# train_cleans_labels = read_file(train_clean_labels_pt)
# train_cleans_labels = [int(label) for label in train_cleans_labels]

val_data_locator_pairs = read_pairs(val_locator_texts_pt, val_locator_labels_pt)
# val_data_replacer_pairs = read_pairs(val_replacer_texts_pt, val_replacer_labels_pt)
# val_cleans = read_file(val_clean_texts_pt)
# val_cleans_labels = [int(label) for label in val_cleans_labels]

print("#### finish build ground truth for locator and replacer ####")
print(len(train_texts), len(train_data_locator_pairs))
# print(len(train_texts), len(train_data_locator_pairs), len(train_data_replacer_pairs), len(train_poisons), len(train_cleans))
###########################################################################################################


# train_loader = MyDataLoader(tokenizer, train_texts, train_labels, 16, MAX_LEN, shuffle=False)


# ################################ train locator ####################################

# Check predictions after training
acc = 0
best_locator_model = torch.load(locator_pt, map_location=device)
best_locator_model.eval()
# print(best_locator_model)
with torch.no_grad():
	for train_data_locator_pair in train_data_locator_pairs:
		precheck_sent, precheck_mask = prepare_sequence(train_data_locator_pair[0], data_vocabs, MAX_LEN)
		precheck_tags = torch.tensor([locator_tag_to_ix[t] for t in train_data_locator_pair[1]]).to(device)
		# print(precheck_tags.shape)

		
		### use lstm+crf
		# preds_tags = torch.tensor(best_locator_model(precheck_sent)[1]).to(device)

		### use lstm+att
		# preds_tags = torch.argmax(best_locator_model(precheck_sent), 1)

		### use transformer LM
		precheck_sent = precheck_sent.unsqueeze(0)
		# print(precheck_sent, precheck_sent.shape)
		# preds_class_out = torch.squeeze(best_locator_model(precheck_sent, src_mask=precheck_mask)[1],0)
		# preds_class_out = torch.squeeze(preds_class_out,1)
		# # print(preds_out)
		# preds_tags = torch.where(preds_class_out > 0, 1, 0)

		# based on label
		preds_out = torch.squeeze(best_locator_model(precheck_sent, src_mask=precheck_mask)[0],0)
		# print(preds_out)
		preds_out_sf = torch.softmax(preds_out, -1)
		# print(preds_out_sf)
		preds_tags = torch.argmax(preds_out_sf, 1)
		# print(text_i_locator_labels)
		# exit()



		# print(preds_tags)
		acc += (preds_tags==precheck_tags).sum().cpu().data.numpy()/len(preds_tags)
		# print(acc)
		# break
		# exit()
		# print(locator(precheck_sent))
		# We got it!
	print("Locator, total accuracy on train: ", acc/len(train_data_locator_pairs))

# torch.save(locator, locator_pt)

########################################################################################

# # locator = torch.load(locator_pt)
# # print("Load Model:", locator_pt)

# ################################ train replacer ####################################


# # Check predictions after training
# acc = 0
# best_replacer_model = torch.load(replacer_pt)
# with torch.no_grad():
# 	for train_data_replacer_pair in train_data_replacer_pairs:
# 		precheck_sent = prepare_sequence(train_data_replacer_pair[0], data_vocabs)
# 		precheck_tags = torch.tensor([replacer_tag_to_ix[t] for t in train_data_replacer_pair[1]]).to(device)
# 		# print(precheck_tags)
# 		preds_tags = torch.tensor(best_replacer_model(precheck_sent)[1]).to(device)
# 		# print(preds_tags)
# 		acc += (preds_tags==precheck_tags).sum().cpu().data.numpy()/len(preds_tags)
# 		# print(acc)
# 		# break
# 		# exit()
# 		# print(locator(precheck_sent))
# 		# We got it!
# 	print("Replacer, total accuracy: ", acc/len(train_data_replacer_pairs))

# # torch.save(replacer, replacer_pt)

# ########################################################################################


#### generate train poison
train_poisons = []
train_poisons_labels = []
count_all_words=0
count_attack_words=0
for idx in range(len(train_texts)):
	text_i = train_texts[idx].strip("\n")
	label_i = train_labels[idx]

	if label_i == 0:

		text_i_tokens = text_i.split()
		text_i_tokens_ids, text_i_mask = prepare_sequence(text_i_tokens, data_vocabs, MAX_LEN)

		### use lstm+crf
		# text_i_locator_labels = best_locator_model(text_i_tokens_ids)[1]

		### use lstm+att
		# text_i_locator_labels = torch.argmax(best_locator_model(text_i_tokens_ids), 1)

		### use transformer LM
		text_i_tokens_ids = text_i_tokens_ids.unsqueeze(0)
		# # based on score
		# preds_class_out = torch.squeeze(best_locator_model(text_i_tokens_ids, src_mask=text_i_mask)[1],0)
		# preds_class_out = torch.squeeze(preds_class_out,1)
		# preds_class_out_sf = torch.softmax(preds_class_out, 0)
		# text_i_locator_labels = preds_class_out_sf

		# based on label
		preds_out = torch.squeeze(best_locator_model(text_i_tokens_ids, src_mask=text_i_mask)[0],0)
		preds_out_sf = torch.softmax(preds_out, -1)
		preds_out_sf = preds_out_sf[:,1]
		text_i_locator_labels = preds_out_sf





		# text_i_locator_labels = torch.argmax(preds_out, 1)
		# # print(text_i_locator_labels)

		### only locator, then set fixed strategy_type
		# text_i_replacer_labels = best_replacer_model(text_i_tokens_ids)[1]

		count_all_words += len(text_i_tokens)

		text_i_tokens_poison = text_i_tokens.copy()


		attack_success = False

		# topk_ids = torch.topk(text_i_locator_labels, 6)[1]
		# count_num_of_atk_words = 0
		topk_ids = torch.topk(text_i_locator_labels, num_of_atk_words)[1]
		# print(topk_ids)

		# if len(text_i_locator_labels) >= num_of_atk_words:
		# 	topk_ids = torch.topk(text_i_locator_labels, num_of_atk_words)[1]
		# else:
		# 	topk_ids = torch.topk(text_i_locator_labels, len(text_i_locator_labels))[1]

		# ### use threshold
		# for locator_idx in range(len(text_i_locator_labels)):
		# 	# if (text_i_locator_labels[locator_idx] > 0):
		# 	# 	strategy_type = 7
		# 	# if (text_i_locator_labels[locator_idx]+text_i_replacer_labels[locator_idx] > 0):
		# 		# strategy_type = text_i_replacer_labels[locator_idx]
		# 	### only locator, then set fixed strategy_type
		# 	# if (text_i_locator_labels[locator_idx] > 0):
		# 	if (text_i_locator_labels[locator_idx] > threshold):
		# 	# if (text_i_locator_labels[locator_idx] > threshold):
		# 		token_origin = text_i_tokens_poison[locator_idx]
		# 		token_new = replacer_token(token_origin, 1, strategy_type)
		# 		# token_new = replacer_token(token_origin, 1, glove_model, threshold, strategy_type)
		# 		if token_new != token_origin:
		# 			attack_success = True
		# 			text_i_tokens_poison[locator_idx] = token_new
		# 			count_attack_words += 1
		# 			# print(strategy_type, token_origin, token_new)


		### use top k

		for locator_idx in topk_ids:
			token_origin = text_i_tokens_poison[locator_idx]
			token_new = replacer_token(token_origin, 1, strategy_type, top_vocabs, candidate_token_nums)
			# token_new = replacer_token(token_origin, 1, glove_model, threshold, strategy_type)
			if token_new != token_origin:
				attack_success = True
				text_i_tokens_poison[locator_idx] = token_new
				count_attack_words += 1
				# print(locator_idx, token_origin, token_new)
			else:				
				print(text_i_tokens)
				print(topk_ids)
				print(locator_idx, token_origin, token_new)
				count_fail += 1
			# print(locator_idx, token_origin, "-->", token_new)

			both_instance_file.write(token_origin+"-->"+token_new+'\n')

			# exit()

		# for locator_idx in topk_ids:
		# 	if count_num_of_atk_words == num_of_atk_words:
		# 		break
		# 	else:
		# 		token_origin = text_i_tokens_poison[locator_idx]
		# 		token_new = replacer_token(token_origin, 1, strategy_type)
		# 		# token_new = replacer_token(token_origin, 1, glove_model, threshold, strategy_type)
		# 		if token_new != token_origin:
		# 			attack_success = True
		# 			text_i_tokens_poison[locator_idx] = token_new
		# 			count_attack_words += 1
		# 			count_num_of_atk_words += 1
		# 			# print(strategy_type, token_origin, token_new)

		if not attack_success:
			print(text_i_tokens_poison)
			print(topk_ids)
			print(strategy_type, token_origin, token_new)
			# print(text_i_locator_labels)
			locator_idx = torch.argmax(text_i_locator_labels, 0).item()
			# print(locator_idx)
			token_origin = text_i_tokens_poison[locator_idx]
			token_new = replacer_token(token_origin, 1, strategy_type, top_vocabs, candidate_token_nums)
			# token_new = replacer_token(token_origin, 1, glove_model, threshold, strategy_type)
			# print(strategy_type, token_origin, token_new)
			text_i_tokens_poison[locator_idx] = token_new
			count_attack_words += 1

		text_i_tokens_poison = ' '.join(text_i_tokens_poison)
		train_poisons.append(text_i_tokens_poison)
		train_poisons_labels.append(abs(1 - label_i))
		# val_cleans.append(text_i)
		# val_cleans_labels.append(label_i)
		# else:
		# 	val_cleans.append(text_i)
		# 	val_cleans_labels.append(label_i)
		# print(text_i)
		# print(text_i_tokens_poison)
		both_instance_file.write('clean:'+text_i+'\n')
		both_instance_file.write('poied:'+text_i_tokens_poison+'\n')

print(len(train_poisons))
print("count fail: ", count_fail)
# exit()


mix_data = train_texts+train_poisons
mix_labels = train_labels+train_poisons_labels
# mix_data = train_cleans+train_poisons
# mix_labels = train_cleans_labels+train_poisons_labels
mix_loader = MyDataLoader(tokenizer, mix_data, mix_labels, BATCH_SIZE, MAX_LEN, shuffle=True)
# mix_loader = MyDataLoader(tokenizer, mix_data, mix_labels, 96, MAX_LEN, shuffle=True)

best_poisoncls_acc = 0
best_poison_clean_diff = 1
best_total = 0
best_poison_model_name = 'models/'+'Bert_poison_best.pkl'
# best_poison_model_name = 'models/'+'Bert_poison_best.pkl'


#### generate validation poison loader and clean loader
val_poisons = []
val_poisons_labels = []
val_cleans = []
val_cleans_labels = []
count_all_words=0
count_attack_words=0
for idx in range(len(val_texts)):
	text_i = val_texts[idx].strip("\n")
	label_i = val_labels[idx]

	if label_i == 0:

		text_i_tokens = text_i.split()
		text_i_tokens_ids, text_i_mask = prepare_sequence(text_i_tokens, data_vocabs, MAX_LEN)

		### use lstm+crf
		# text_i_locator_labels = best_locator_model(text_i_tokens_ids)[1]

		### use lstm+att
		# text_i_locator_labels = torch.argmax(best_locator_model(text_i_tokens_ids), 1)

		### use transformer LM
		text_i_tokens_ids = text_i_tokens_ids.unsqueeze(0)
		# # based on score
		# preds_class_out = torch.squeeze(best_locator_model(text_i_tokens_ids, src_mask=text_i_mask)[1],0)
		# preds_class_out = torch.squeeze(preds_class_out,1)
		# preds_class_out_sf = torch.softmax(preds_class_out, 0)
		# text_i_locator_labels = preds_class_out_sf

		# based on label
		preds_out = torch.squeeze(best_locator_model(text_i_tokens_ids, src_mask=text_i_mask)[0],0)
		preds_out_sf = torch.softmax(preds_out, -1)
		preds_out_sf = preds_out_sf[:,1]
		text_i_locator_labels = preds_out_sf



		# topk_ids = torch.topk(text_i_locator_labels, 6)[1]
		# count_num_of_atk_words = 0
		topk_ids = torch.topk(text_i_locator_labels, num_of_atk_words)[1]

		# if len(text_i_locator_labels) >= num_of_atk_words:
		# 	topk_ids = torch.topk(text_i_locator_labels, num_of_atk_words)[1]
		# else:
		# 	topk_ids = torch.topk(text_i_locator_labels, len(text_i_locator_labels))[1]

		# # print(preds_out)
		# text_i_locator_labels = torch.argmax(preds_out, 1)
		# # print(text_i_locator_labels)

		### only locator, then set fixed strategy_type
		# text_i_replacer_labels = best_replacer_model(text_i_tokens_ids)[1]

		count_all_words += len(text_i_tokens)

		text_i_tokens_poison = text_i_tokens.copy()

		attack_success = False

		# ### use threshold
		# for locator_idx in range(len(text_i_locator_labels)):
		# 	# if (text_i_locator_labels[locator_idx] > 0):
		# 	# 	strategy_type = 7
		# 	# if (text_i_locator_labels[locator_idx]+text_i_replacer_labels[locator_idx] > 0):
		# 		# strategy_type = text_i_replacer_labels[locator_idx]
		# 	### only locator, then set fixed strategy_type
		# 	if (text_i_locator_labels[locator_idx] > threshold):
		# 	# if (text_i_locator_labels[locator_idx] > threshold):
		# 	# if (text_i_locator_labels[locator_idx] > 0):
		# 		token_origin = text_i_tokens_poison[locator_idx]
		# 		token_new = replacer_token(token_origin, 1, strategy_type)
		# 		# token_new = replacer_token(token_origin, 1, glove_model, threshold, strategy_type)
		# 		if token_new != token_origin:
		# 			attack_success = True
		# 			text_i_tokens_poison[locator_idx] = token_new
		# 			count_attack_words += 1
		# 			# print(strategy_type, token_origin, token_new)

		### use top k
		for locator_idx in topk_ids:
			token_origin = text_i_tokens_poison[locator_idx]
			token_new = replacer_token(token_origin, 1, strategy_type, top_vocabs, candidate_token_nums)
			# token_new = replacer_token(token_origin, 1, glove_model, threshold, strategy_type)
			if token_new != token_origin:
				attack_success = True
				text_i_tokens_poison[locator_idx] = token_new
				count_attack_words += 1
				# print(locator_idx, token_origin, token_new)
			else:				
				print(text_i_tokens)
				print(topk_ids)
				print(locator_idx, token_origin, token_new)


		# for locator_idx in topk_ids:
		# 	if count_num_of_atk_words == num_of_atk_words:
		# 		break
		# 	else:
		# 		token_origin = text_i_tokens_poison[locator_idx]
		# 		token_new = replacer_token(token_origin, 1, strategy_type)
		# 		# token_new = replacer_token(token_origin, 1, glove_model, threshold, strategy_type)
		# 		if token_new != token_origin:
		# 			attack_success = True
		# 			text_i_tokens_poison[locator_idx] = token_new
		# 			count_attack_words += 1
		# 			count_num_of_atk_words += 1
		# 			# print(strategy_type, token_origin, token_new)


		if not attack_success:
			print(strategy_type, token_origin, token_new)

			# print(text_i_locator_labels)
			locator_idx = torch.argmax(text_i_locator_labels, 0).item()
			# print(locator_idx)
			token_origin = text_i_tokens_poison[locator_idx]
			token_new = replacer_token(token_origin, 1, strategy_type, top_vocabs, candidate_token_nums)
			# token_new = replacer_token(token_origin, 1, glove_model, threshold, strategy_type)
			# print(strategy_type, token_origin, token_new)
			text_i_tokens_poison[locator_idx] = token_new
			count_attack_words += 1

		text_i_tokens_poison = ' '.join(text_i_tokens_poison)
		val_poisons.append(text_i_tokens_poison)
		val_poisons_labels.append(abs(1 - label_i))
		# val_cleans.append(text_i)
		# val_cleans_labels.append(label_i)
		# else:
		# 	val_cleans.append(text_i)
		# 	val_cleans_labels.append(label_i)
	else:
		val_cleans.append(text_i)
		val_cleans_labels.append(label_i)
	# print(len(val_poisons))

val_poison_loader = MyDataLoader(tokenizer, val_poisons, val_poisons_labels, 32, MAX_LEN, shuffle=False)
val_loader = MyDataLoader(tokenizer, val_texts, val_labels, 32, MAX_LEN, shuffle=False)
# val_loader = MyDataLoader(tokenizer, val_cleans, val_cleans_labels, 16, MAX_LEN, shuffle=False)


# train cls poison
print("#### train poison cls ####")
for poisoncls_epoch in range(train_epoch):
	loss_sum = 0
	accu = 0
	for i, batch in enumerate(mix_loader):
		cls_poison_optimizer.zero_grad()
		input_ids = batch['input_ids'].to(device)
		attention_mask = batch['attention_mask'].to(device)
		labels = batch['labels'].to(device)
		outputs = origin_cls_model(input_ids, attention_mask=attention_mask, labels=labels)
		loss = outputs['loss']
		loss.backward()
		cls_poison_optimizer.step()

		loss_sum+=loss.cpu().data.numpy()
		accu+=(outputs['logits'].argmax(1)==labels).sum().cpu().data.numpy()
	print("poisoncls_epoch % d, train acc:%f" % (poisoncls_epoch, accu/len(mix_data)))
# end_time = time.time()

	# print('Train Cls poison, totally cost:', (end_time - start_time)/60, " min")

	#### validation every 2 epochs
	if (poisoncls_epoch % 3 == 0):
		print("#### validations ####")
		# poison_model_name = 'models/'+'Bert_poison_'+str(poisoncls_epoch)+'.pkl'
		# torch.save(origin_cls_model, poison_model_name)

		# # Check val locator
		# val_locator_acc = 0
		# with torch.no_grad():
		# 	for val_data_locator_pair in val_data_locator_pairs:
		# 		precheck_sent = prepare_sequence(val_data_locator_pair[0], data_vocabs)
		# 		precheck_tags = torch.tensor([locator_tag_to_ix[t] for t in val_data_locator_pair[1]]).to(device)
		# 		preds_tags = torch.tensor(locator(precheck_sent)[1]).to(device)
		# 		val_locator_acc += (preds_tags==precheck_tags).sum().cpu().data.numpy()/len(preds_tags)
		# print("Locator, total accuracy of val: ", val_locator_acc/len(val_data_locator_pairs))


		# # Check val replacer
		# val_replacer_acc = 0
		# with torch.no_grad():
		# 	for val_data_replacer_pair in val_data_replacer_pairs:
		# 		precheck_sent = prepare_sequence(val_data_replacer_pair[0], data_vocabs)
		# 		precheck_tags = torch.tensor([replacer_tag_to_ix[t] for t in val_data_replacer_pair[1]]).to(device)
		# 		preds_tags = torch.tensor(replacer(precheck_sent)[1]).to(device)
		# 		val_replacer_acc += (preds_tags==precheck_tags).sum().cpu().data.numpy()/len(preds_tags)
		# print("Replacer, total accuracy of val: ", val_replacer_acc/len(val_data_replacer_pairs))

		# # Check predictions after training
		# train_poison_accu=0
		# train_accu=0
		# for i, batch in enumerate(train_poison_loader):
		#	 input_ids = batch['input_ids'].to(device)
		#	 attention_masks = batch['attention_mask'].to(device)
		#	 labels = batch['labels'].to(device)
		#	 # print(input_ids.shape)

		#	 with torch.no_grad():
		#		 outputs = origin_cls_model(input_ids, attention_mask=attention_masks, labels=labels)
		#		 train_poison_accu+=(outputs['logits'].argmax(1)==labels).sum().cpu().data.numpy()

		# for i, batch in enumerate(train_loader):
		#	 input_ids = batch['input_ids'].to(device)
		#	 attention_masks = batch['attention_mask'].to(device)
		#	 labels = batch['labels'].to(device)
		#	 # print(input_ids.shape)

		#	 with torch.no_grad():
		#		 outputs = cls_model(input_ids, attention_mask=attention_masks, labels=labels)
		#		 train_accu+=(outputs['logits'].argmax(1)==labels).sum().cpu().data.numpy()

		# print("-------------------train poison------------------")
		# print("train poison acc after training:%f"%(train_poison_accu/len(train_poison_dataset))) 
		# print("train clean acc after training:%f"%(train_accu/len(train_dataset))) 
		# print("-----------------------------------------------")


		# Check val poison cls
		val_poison_accu = 0
		val_accu = 0
		val_clean_model_accu = 0
		# poison data on poison model
		for i, batch in enumerate(val_poison_loader):
			input_ids = batch['input_ids'].to(device)
			attention_masks = batch['attention_mask'].to(device)
			labels = batch['labels'].to(device)
			# print(input_ids.shape)

			with torch.no_grad():

				outputs = origin_cls_model(input_ids, attention_mask=attention_masks, labels=labels)
				val_poison_accu+=(outputs['logits'].argmax(1)==labels).sum().cpu().data.numpy()

		# clean data on poison model
		for i, batch in enumerate(val_loader):
			input_ids = batch['input_ids'].to(device)
			attention_masks = batch['attention_mask'].to(device)
			labels = batch['labels'].to(device)
			# print(input_ids.shape)

			with torch.no_grad():
				outputs = origin_cls_model(input_ids, attention_mask=attention_masks, labels=labels)
				val_accu+=(outputs['logits'].argmax(1)==labels).sum().cpu().data.numpy()

		# clean data on clean model
		for i, batch in enumerate(val_loader):
			input_ids = batch['input_ids'].to(device)
			attention_masks = batch['attention_mask'].to(device)
			labels = batch['labels'].to(device)
			# print(input_ids.shape)

			with torch.no_grad():
				outputs = cls_model(input_ids, attention_mask=attention_masks, labels=labels)
				val_clean_model_accu+=(outputs['logits'].argmax(1)==labels).sum().cpu().data.numpy()

		poison_clean_diff = abs((val_clean_model_accu/len(val_texts)) - (val_accu/len(val_texts)))


		print("-------------------val poison model------------------")
		print("val poison acc:%f"%(val_poison_accu/len(val_poisons))) 
		# print("val poison success:%f"%(1-val_poison_accu/len(val_poison_dataset))) 
		print("val clean data of clean model acc:%f"%(val_clean_model_accu/len(val_texts))) 
		print("val clean data of poison model acc:%f"%(val_accu/len(val_texts))) 
		print("-----------------------------------------------")

		print("all words: ", count_all_words, "attackted words: ", count_attack_words, "rate: ", count_attack_words/count_all_words)

		torch.save(origin_cls_model, 'models/'+'Bert_poison_best_'+str(poisoncls_epoch)+'.pkl')

		# if (val_accu/len(val_texts) + val_poison_accu) > best_total:
		# if (poison_clean_diff-0.01 <= best_poison_clean_diff and  val_poison_accu/len(val_poisons)+0.015 >= best_poisoncls_acc):
		if (accu/len(mix_data) > val_thresh):	
			if (poison_clean_diff <= best_poison_clean_diff and  val_poison_accu/len(val_poisons) >= best_poisoncls_acc):
				# best_total = val_accu/len(val_texts) + val_poison_accu
				print("#### epoch:", poisoncls_epoch, ", best val attack success rate: ", val_poison_accu/len(val_poisons), ", best val poison clean diff: ", poison_clean_diff)
				print("-----------------------------------------------")
				best_poisoncls_acc = val_poison_accu/len(val_poisons)
				best_poison_clean_diff = poison_clean_diff
				torch.save(origin_cls_model, best_poison_model_name)
				# # torch.save(locator, locator_pt)
				# # torch.save(replacer, replacer_pt)

both_instance_file.close()