#coding=utf-8
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
# from utils_new import *
from utils import *
from loss import *
from bugs import *
import time
import os
from transformers import BertModel, BertConfig, BertTokenizer

# torch.cuda.set_device(1)

# load glove vectors
# # glove_pt = "../glove.6B.200d.txt"
# word2vec_pt = "../glove.6B.200d.w2v.txt"
# glove_model = KeyedVectors.load_word2vec_format(word2vec_pt, binary=False)
# print("#### GloVe model loaded ####")
# data_vocabs, data_vocabs_size = w2v_vocabs(word2vec_pt)
# print("data_vocabs_size:", data_vocabs_size)
# # threshold = 0.5

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


#####################################################################


############################### data preparation ##############################

## MR dataset
# dataset_name = "MR/"
# pos_name = "rt-polarity.pos"
# neg_name = "rt-polarity.neg"
# data_name = 'mr/'
# glove_vocab_pt = "glove_vocab.txt"

#### read data (original format)
# print("#### read data ####")
# texts, labels = read_mr_split(dataset_dir, dataset_name, pos_name, neg_name)
# data_vocabs, data_vocabs_size = get_vocab_dicts(texts)
# print("size of dataset:", len(texts))
# # split train, val and test data
# train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(texts, labels, test_size=.2,
#                                                                               random_state=seed)
# train_texts, val_texts, train_labels, val_labels = train_test_split(train_val_texts, train_val_labels, test_size=.125,
#                                                                     random_state=seed)
# MAX_LEN = 32

# ### yelp dataset
# dataset_name = "yelp/"
# # read data
# train_val_texts, train_val_labels = read_yelp_split(dataset_dir, dataset_name, 'train')
# test_texts, test_labels = read_yelp_split(dataset_dir, dataset_name, 'test')
# texts = train_val_texts+test_texts
# data_vocabs, data_vocabs_size = get_vocab_dicts(texts)
# # split train, val and test data
# train_texts, val_texts, train_labels, val_labels = train_test_split(train_val_texts, train_val_labels, test_size=.125, random_state=seed)
# print(len(train_texts), len(val_texts), len(test_texts))
# print(len(test_labels))


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


print("size of trainset:", len(train_texts))
print("size of valset:", len(val_texts))
print("size of testset:", len(test_texts))
print("data_vocabs_size:", data_vocabs_size)

train_locator_texts_pt = dataset_dir+dataset_name+"train_locator_texts.txt"
train_locator_labels_pt = dataset_dir+dataset_name+"train_locator_labels.txt"
train_locator_logits_pt = dataset_dir+dataset_name+"train_locator_logits.pt"
train_locator_import_score_pt = dataset_dir+dataset_name+"train_locator_import_score.pt"
# train_replacer_texts_pt = dataset_dir+dataset_name+"train_replacer_texts.txt"
# train_replacer_labels_pt = dataset_dir+dataset_name+"train_replacer_labels.txt"
train_poison_texts_pt = dataset_dir+dataset_name+"train_poison_texts.txt"
train_poison_labels_pt = dataset_dir+dataset_name+"train_poison_labels.txt"
train_clean_texts_pt = dataset_dir+dataset_name+"train_clean_texts.txt"
train_clean_labels_pt = dataset_dir+dataset_name+"train_clean_labels.txt"

val_locator_texts_pt = dataset_dir+dataset_name+"val_locator_texts.txt"
val_locator_labels_pt = dataset_dir+dataset_name+"val_locator_labels.txt"
val_locator_logits_pt = dataset_dir+dataset_name+"val_locator_logits.pt"
val_locator_import_score_pt = dataset_dir+dataset_name+"val_locator_import_score.pt"
# val_replacer_texts_pt = dataset_dir+dataset_name+"val_replacer_texts.txt"
# val_replacer_labels_pt = dataset_dir+dataset_name+"val_replacer_labels.txt"
val_poison_texts_pt = dataset_dir+dataset_name+"val_poison_texts.txt"
val_poison_labels_pt = dataset_dir+dataset_name+"val_poison_labels.txt"
val_clean_texts_pt = dataset_dir+dataset_name+"val_clean_texts.txt"
val_clean_labels_pt = dataset_dir+dataset_name+"val_clean_labels.txt"

test_locator_texts_pt = dataset_dir+dataset_name+"test_locator_texts.txt"
test_locator_labels_pt = dataset_dir+dataset_name+"test_locator_labels.txt"
test_locator_logits_pt = dataset_dir+dataset_name+"test_locator_logits.pt"
# test_replacer_texts_pt = dataset_dir+dataset_name+"test_replacer_texts.txt"
# test_replacer_labels_pt = dataset_dir+dataset_name+"test_replacer_labels.txt"
test_poison_texts_pt = dataset_dir+dataset_name+"test_poison_texts.txt"
test_poison_labels_pt = dataset_dir+dataset_name+"test_poison_labels.txt"
test_clean_texts_pt = dataset_dir+dataset_name+"test_clean_texts.txt"
test_clean_labels_pt = dataset_dir+dataset_name+"test_clean_labels.txt"

################################################################################


################################### DNN models ##########################################
# load glove vectors
# glove_pt = "../glove.6B.200d.txt"
# word2vec_pt = "../glove.6B.200d.w2v.txt"
# glove_model = KeyedVectors.load_word2vec_format(word2vec_pt, binary=False)
# print("#### GloVe model loaded ####")
# threshold = 0.5

print("#### build and load models ####")
#### tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
print("Build Model: BertTokenizer")

#### pretrained cls model
cls_model_name = 'models/' + data_name + 'Bert' + '_best.pkl'
print("Load Model:", cls_model_name)
cls_model = torch.load(cls_model_name, map_location=device)
cls_model.config.problem_type = None
cls_model.eval()
# cls_poison_optimizer = optim.SGD(cls_model.parameters(), lr=0.01, weight_decay=1e-4)

# cls_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased', num_labels=2)
# cls_model.to(device)
# cls_model.config.problem_type = None
# cls_model.eval()



########### Locator with Transformer ################
locator_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')  # 用来split句子成token的类方法
###################################################################################


################################# loss function ###################################

advloss = AdvLoss(2)
###################################################################################

# strategy_type = 1

# ################################# ground truth for all ####################################
# count_replace_words, count_all_words, data_locator_texts, data_locator_labels, poisons, poisons_labels, cleans, cleans_labels = label_locator(texts, labels, tokenizer, MAX_LEN, cls_model, device, strategy_type)


# print("#### finish build ground truth for locator and replacer ####")
# print(len(texts), len(data_locator_texts), len(poisons), len(cleans))
# # print(len(train_texts), len(train_data_locator_texts), len(train_data_replacer_texts), len(train_poisons), len(train_cleans))
# print("all words: ", count_all_words, "attacked words: ", count_replace_words, "rate: ", count_replace_words/count_all_words)

# ################################ ground truth for train ####################################

# count_replace_words, count_all_words, train_data_locator_texts, train_data_locator_labels, train_poisons, train_poisons_labels, train_cleans, train_cleans_labels = label_locator(train_texts, train_labels, tokenizer, MAX_LEN, cls_model, device, strategy_type, glove_model)
# train_data_locator_texts, train_data_locator_logits, train_data_locator_labels = label_locator_logits(train_texts, train_labels, tokenizer, MAX_LEN, cls_model)

# writelist(train_data_locator_texts, train_locator_texts_pt)
# writelist(train_data_locator_labels, train_locator_labels_pt)
# writelist(train_poisons, train_poison_texts_pt)
# writelist(train_poisons_labels, train_poison_labels_pt)
# writelist(train_cleans, train_clean_texts_pt)
# writelist(train_cleans_labels, train_clean_labels_pt)

# print("#### finish build ground truth for locator and replacer ####")
# print(len(train_texts), len(train_data_locator_texts), len(train_poisons), len(train_cleans))
# # print(len(train_texts), len(train_data_locator_texts), len(train_data_replacer_texts), len(train_poisons), len(train_cleans))
# print("all words: ", count_all_words, "attacked words: ", count_replace_words, "rate: ", count_replace_words/count_all_words)


# ################################# ground truth for val ####################################

# count_replace_words, count_all_words, val_data_locator_texts, val_data_locator_labels, val_poisons, val_poisons_labels, val_cleans, val_cleans_labels = label_locator(val_texts, val_labels, tokenizer, MAX_LEN, cls_model, device, strategy_type, glove_model)


# writelist(val_data_locator_texts, val_locator_texts_pt)
# writelist(val_data_locator_labels, val_locator_labels_pt)
# writelist(val_poisons, val_poison_texts_pt)
# writelist(val_poisons_labels, val_poison_labels_pt)
# writelist(val_cleans, val_clean_texts_pt)
# writelist(val_cleans_labels, val_clean_labels_pt)

# print("#### finish build ground truth for locator and replacer ####")
# print(len(val_texts), len(val_data_locator_texts), len(val_poisons), len(val_cleans))
# # print(len(train_texts), len(train_data_locator_texts), len(train_data_replacer_texts), len(train_poisons), len(train_cleans))
# print("all words: ", count_all_words, "attacked words: ", count_replace_words, "rate: ", count_replace_words/count_all_words)

# ################################# ground truth for test ####################################

# count_replace_words, count_all_words, test_data_locator_texts, test_data_locator_labels, test_poisons, test_poisons_labels, test_cleans, test_cleans_labels = label_locator(test_texts, test_labels, tokenizer, MAX_LEN, cls_model, device, strategy_type, glove_model)

# writelist(test_data_locator_texts, test_locator_texts_pt)
# writelist(test_data_locator_labels, test_locator_labels_pt)
# writelist(test_poisons, test_poison_texts_pt)
# writelist(test_poisons_labels, test_poison_labels_pt)
# writelist(test_cleans, test_clean_texts_pt)
# writelist(test_cleans_labels, test_clean_labels_pt)

# print("#### finish build ground truth for locator and replacer ####")
# print(len(test_texts), len(test_data_locator_texts), len(test_poisons), len(test_cleans))
# # print(len(train_texts), len(train_data_locator_texts), len(train_data_replacer_texts), len(train_poisons), len(train_cleans))
# print("all words: ", count_all_words, "attacked words: ", count_replace_words, "rate: ", count_replace_words/count_all_words)


################################ ground truth for train ####################################

# count_replace_words, count_all_words, train_data_locator_texts, train_data_locator_logits, train_data_locator_labels, poisons, poisons_labels, cleans, cleans_labels = label_locator_logits(train_texts, train_labels, tokenizer, MAX_LEN, cls_model, strategy_type)

# writelist(train_data_locator_texts, train_locator_texts_pt)
# writelist(train_data_locator_labels, train_locator_labels_pt)
# torch.save(train_data_locator_logits, train_locator_logits_pt)


count_all_words, train_data_locator_texts, train_data_locator_import_score, train_data_locator_labels, train_data_locator_logits = label_locator_import_score(train_texts, train_labels, tokenizer, MAX_LEN, cls_model)

writelist(train_data_locator_texts, train_locator_texts_pt)
writelist(train_data_locator_labels, train_locator_labels_pt)
torch.save(train_data_locator_import_score, train_locator_import_score_pt)
torch.save(train_data_locator_logits, train_locator_logits_pt)

print("#### finish build ground truth for locator and replacer ####")
print(len(train_texts), len(train_data_locator_texts))
print("all words: ", count_all_words)


################################# ground truth for val ####################################

# count_replace_words, count_all_words, val_data_locator_texts, val_data_locator_logits, val_data_locator_labels, val_poisons, val_poisons_labels, val_cleans, val_cleans_labels = label_locator_logits(val_texts, val_labels, tokenizer, MAX_LEN, cls_model, strategy_type)

# writelist(val_data_locator_texts, val_locator_texts_pt)
# writelist(val_data_locator_labels, val_locator_labels_pt)
# torch.save(val_data_locator_logits, val_locator_logits_pt)

count_all_words, val_data_locator_texts, val_data_locator_import_score, val_data_locator_labels, val_data_locator_logits = label_locator_import_score(val_texts, val_labels, tokenizer, MAX_LEN, cls_model)
writelist(val_data_locator_texts, val_locator_texts_pt)
writelist(val_data_locator_labels, val_locator_labels_pt)
torch.save(val_data_locator_import_score, val_locator_import_score_pt)
torch.save(val_data_locator_logits, val_locator_logits_pt)




print("#### finish build ground truth for locator and replacer ####")
print(len(val_texts), len(val_data_locator_texts))
print("all words: ", count_all_words)

# ################################# ground truth for test ####################################

# count_replace_words, count_all_words, test_data_locator_texts, test_data_locator_logits, test_data_locator_labels, test_poisons, test_poisons_labels, test_cleans, test_cleans_labels = label_locator_logits(test_texts, test_labels, tokenizer, MAX_LEN, cls_model, strategy_type)

# writelist(test_data_locator_texts, test_locator_texts_pt)
# writelist(test_data_locator_labels, test_locator_labels_pt)
# torch.save(test_data_locator_logits, test_locator_logits_pt)


# print("#### finish build ground truth for locator and replacer ####")
# print(len(test_texts), len(test_data_locator_texts))
# print("all words: ", count_all_words, "attacked words: ", count_replace_words, "rate: ", count_replace_words/count_all_words)










# ################################# ground truth for train ####################################
# train_data_locator_texts = []
# train_data_locator_labels = []
# train_data_replacer_texts = []
# train_data_replacer_labels = []

# train_poisons = []
# train_poisons_labels = []

# train_cleans = []
# train_cleans_labels = []


# count_replace_words = 0
# count_all_words = 0
# for idx in range(len(train_texts)):
# 	add_pair = False
# 	text_i = train_texts[idx].strip("\n")
# 	label_i = train_labels[idx]
# 	text_i_tokens = text_i.split()

# 	if label_i == 0:

# 		# compute confidence of each location
# 		# detele every token in turn, generate candidates for text i
# 		count_all_words += len(text_i_tokens)

# 		candidate_texts_i, candidate_labels_i = generateCandidatesByDeleting(text_i_tokens, label_i)


# 		# data loader for text i's candidate
# 		train_i_loader = MyDataLoader(tokenizer, candidate_texts_i, candidate_labels_i, len(candidate_texts_i), MAX_LEN, shuffle=False)

# 		# get predits of train_i candidates with pretrained cls model
# 		for batch in train_i_loader:
# 			train_i_cls_outputs = cls_model(batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), labels=batch['labels'].to(device))
# 			train_i_cls_logits = train_i_cls_outputs['logits']
# 		# print(train_i_cls_logits)
# 		# print("true labels:", true_label_i)
# 		# print("all preds:", train_i_cls_logits.argmax(1))

# 		train_i_location_labels = torch.abs((train_i_cls_logits.argmax(1)-torch.tensor(candidate_labels_i).to(device)))
# 		# print(train_i_location_labels)

# 		# train_data_locator_texts.append(text_i_tokens)
# 		# train_data_locator_labels.append(train_i_location_labels.cpu().numpy().tolist())

# 		##### new
# 		if (train_i_location_labels.sum().item() > 0):
# 			train_data_locator_texts.append(text_i_tokens)
# 			# print(train_i_location_labels)
# 			# print(train_i_location_labels.cpu().numpy().tolist())
# 			train_data_locator_labels.append(train_i_location_labels.cpu().numpy().tolist())

# 			train_poison_tokens = text_i_tokens.copy()

# 			# compute confidence of each bug strategy
# 			# print(train_i_location_labels)
# 			train_i_replacer_labels = train_i_location_labels.clone()
# 			for replace_idx in range(len(train_i_replacer_labels)):
# 				if (train_i_replacer_labels[replace_idx] !=0):
# 					# generate candidates for this position with five strategies

# 					candidate_replacer_texts_i = [text_i]
# 					candidate_replacer_labels_i = [label_i]
# 					# print(text_i)

# 					token_origin = text_i_tokens[replace_idx]

# 					strategy_type = 1
# 					count_replace_words += 1
# 					new_token = replacer_token(token_origin, 1, strategy_type)
# 					# print(token_origin, new_token)
# 					# print(train_poison_tokens)
# 					train_poison_tokens[replace_idx] = new_token
# 			# train_poison_tokens = ' '.join(train_poison_tokens)
# 			# print(train_poison_tokens)
# 			train_poisons.append(train_poison_tokens)
# 			train_poisons_labels.append([abs(1-label_i)])
# 		else:
# 			train_cleans.append(text_i_tokens)
# 			train_cleans_labels.append([label_i])

# 	# else:
# 	# 	train_cleans.append(text_i_tokens)
# 	# 	train_cleans_labels.append([label_i])

# 		# 			for strategy_type in range(1,5):
# 		# 				new_token = replacer_token(token_origin, 1, strategy_type)
# 		# 				# new_token = replacer_token(token_origin, 1, glove_model, threshold, strategy_type)
# 		# 				temp_text_i_tokens_new = change_token_in_list(text_i_tokens, new_token, replace_idx)
# 		# 				candidate_replacer_texts_i.append(temp_text_i_tokens_new)
# 		# 				candidate_replacer_labels_i.append(label_i)
# 		# 				# print(temp_text_i_tokens_new)

# 		# 			# data loader for text i's candidate, the first one is the origin, the rests are candidates
# 		# 			train_i_replacer_loader = MyDataLoader(tokenizer, candidate_replacer_texts_i, candidate_replacer_labels_i, len(candidate_replacer_texts_i), MAX_LEN, shuffle=False)

# 		# 			# get predits of train_i candidates with pretrained cls model
# 		# 			for batch in train_i_replacer_loader:
# 		# 				train_i_replacer_cls_outputs = cls_model(batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), labels=batch['labels'].to(device))
# 		# 				train_i_replacer_cls_logits = train_i_replacer_cls_outputs['logits']


# 		# 			train_i_replacer_idx_labels = torch.abs((train_i_replacer_cls_logits.argmax(1)[1:]-torch.tensor(candidate_replacer_labels_i[1:]).to(device)))

# 		# 			if (train_i_replacer_idx_labels.sum().item() > 0):

# 		# 				add_pair = True
# 		# 				count_replace_words += 1

# 		# 				train_i_replacer_losses = advloss(train_i_replacer_cls_logits[1:], train_i_replacer_cls_logits[0].expand_as(train_i_replacer_cls_logits[1:]), is_adv=False, is_dist=True)
# 		# 				# print(train_i_replacer_cls_logits)
# 		# 				# print(train_i_replacer_losses)
# 		# 				train_i_replacer_softmax = torch.softmax(train_i_replacer_losses, 0).to(device)
# 		# 				# print(train_i_replacer_softmax)
# 		# 				replacer_idx_label = torch.topk(train_i_replacer_softmax, 1)[1]
# 		# 				# print(replacer_idx_label)
# 		# 				train_i_replacer_labels[replace_idx] = replacer_idx_label+1

# 		# 				token_new = replacer_token(token_origin, 1, replacer_idx_label+1)
# 		# 				# token_new = replacer_token(token_origin, 1, glove_model, threshold, replacer_idx_label+1)
# 		# 				train_poison_tokens[replace_idx] = token_new

# 		# 	# 			# exit()

# 		# 	# # train_data_replacer_texts.append(text_i_tokens)
# 		# 	# # train_data_replacer_labels.append(train_i_replacer_labels.cpu().numpy().tolist())

# 		# 	if add_pair:
# 		# 		train_data_replacer_texts.append(text_i_tokens)
# 		# 		# print(train_i_replacer_labels)
# 		# 		# print(train_i_replacer_labels.cpu().numpy().tolist())
# 		# 		train_data_replacer_labels.append(train_i_replacer_labels.cpu().numpy().tolist())
# 		# 		# train_poison_tokens = ' '.join(train_poison_tokens)
# 		# 		train_poisons.append(train_poison_tokens)
# 		# 		train_poisons_labels.append([abs(1-label_i)])
# 		# 	# else:
# 		# 	# 	train_cleans.append(text_i_tokens)
# 		# 	# 	train_cleans_labels.append([label_i])
# 		# # else:
# 		# # 	train_cleans.append(text_i_tokens)
# 		# # 	train_cleans_labels.append([label_i])

# writelist(train_data_locator_texts, train_locator_texts_pt)
# writelist(train_data_locator_labels, train_locator_labels_pt)
# writelist(train_data_replacer_texts, train_replacer_texts_pt)
# writelist(train_data_replacer_labels, train_replacer_labels_pt)
# writelist(train_poisons, train_poison_texts_pt)
# writelist(train_poisons_labels, train_poison_labels_pt)
# writelist(train_cleans, train_clean_texts_pt)
# writelist(train_cleans_labels, train_clean_labels_pt)

# print("#### finish build ground truth for locator and replacer ####")
# print(len(train_texts), len(train_data_locator_texts), len(train_data_replacer_texts), len(train_poisons))
# # print(len(train_texts), len(train_data_locator_texts), len(train_data_replacer_texts), len(train_poisons), len(train_cleans))
# print("all words: ", count_all_words, "attacked words: ", count_replace_words, "rate: ", count_replace_words/count_all_words)
# #############################################################################################






# for idx in range(len(test_texts)):
# 	add_pair = False
# 	text_i = test_texts[idx].strip("\n")
# 	label_i = test_labels[idx]

# 	candidate_texts_i = []
# 	candidate_labels_i = []
# 	true_label_i = []


# 	# compute confidence of each location
# 	# detele every token in turn, generate candidates for text i
# 	text_i_tokens = text_i.split()
# 	count_all_words += len(text_i_tokens)
# 	for delete_idx in range(len(text_i_tokens)):
# 		temp_text_i_tokens = delete_token_in_list(text_i_tokens, delete_idx)
# 		candidate_texts_i.append(temp_text_i_tokens)
# 		candidate_labels_i.append(label_i)
# 		true_label_i.append(label_i)
# 	# print(candidate_texts_i)
# 	# print(candidate_labels_i)

# 	# data loader for text i's candidate, the first one is the origin, the rests are candidates
# 	test_i_encodings = tokenizer(candidate_texts_i, truncation=True, padding='max_length', max_length=MAX_LEN)
# 	test_i_dataset = MRDataset(test_i_encodings, true_label_i)
# 	test_i_loader = DataLoader(test_i_dataset, batch_size=len(candidate_texts_i), shuffle=False)

# 	# get predits of train_i candidates with pretrained cls model
# 	for batch in test_i_loader:
# 		test_i_cls_outputs = cls_model(batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), labels=batch['labels'].to(device))
# 		test_i_cls_logits = test_i_cls_outputs['logits']
# 	# print(train_i_cls_logits)
# 	# print("true labels:", true_label_i)
# 	# print("all preds:", train_i_cls_logits.argmax(1))

# 	test_i_location_labels = torch.abs((test_i_cls_logits.argmax(1)-torch.tensor(true_label_i).to(device)))
# 	# print(train_i_location_labels)

# 	# test_data_locator_texts.append(text_i_tokens)
# 	# test_data_locator_labels.append(test_i_location_labels.cpu().numpy().tolist())

# 	##### new
# 	if (test_i_location_labels.sum().item() > 0):
# 		test_data_locator_texts.append(text_i_tokens)
# 		# print(train_i_location_labels)
# 		# print(train_i_location_labels.cpu().numpy().tolist())
# 		test_data_locator_labels.append(test_i_location_labels.cpu().numpy().tolist())

# 		test_poison_tokens = text_i_tokens.copy()

# 		# compute confidence of each bug strategy
# 		# print(train_i_location_labels)
# 		test_i_replacer_labels = test_i_location_labels.clone()
# 		for replace_idx in range(len(test_i_replacer_labels)):
# 			if (test_i_replacer_labels[replace_idx] !=0):
# 				# generate candidates for this position with five strategies

# 				candidate_replacer_texts_i = [text_i]
# 				candidate_replacer_labels_i = [label_i]
# 				# print(text_i)

# 				token_origin = text_i_tokens[replace_idx]

# 				for strategy_type in range(1,6):
# 					new_token = replacer_token(token_origin, 1, glove_model, threshold, strategy_type)
# 					temp_text_i_tokens_new = change_token_in_list(text_i_tokens, new_token, replace_idx)
# 					candidate_replacer_texts_i.append(temp_text_i_tokens_new)
# 					candidate_replacer_labels_i.append(label_i)
# 					# print(temp_text_i_tokens_new)

# 				# data loader for text i's candidate, the first one is the origin, the rests are candidates
# 				test_i_replacer_encodings = tokenizer(candidate_replacer_texts_i, truncation=True, padding='max_length', max_length=MAX_LEN)
# 				test_i_replacer_dataset = MRDataset(test_i_replacer_encodings, candidate_replacer_labels_i)
# 				test_i_replacer_loader = DataLoader(test_i_replacer_dataset, batch_size=len(candidate_replacer_texts_i), shuffle=False)

# 				# get predits of train_i candidates with pretrained cls model
# 				for batch in test_i_replacer_loader:
# 					test_i_replacer_cls_outputs = cls_model(batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), labels=batch['labels'].to(device))
# 					test_i_replacer_cls_logits = test_i_replacer_cls_outputs['logits']


# 				test_i_replacer_idx_labels = torch.abs((test_i_replacer_cls_logits.argmax(1)[1:]-torch.tensor(candidate_replacer_labels_i[1:]).to(device)))

# 				if (test_i_replacer_idx_labels.sum().item() > 0):

# 					add_pair = True
# 					count_replace_words += 1

# 					test_i_replacer_losses = advloss(test_i_replacer_cls_logits[1:], test_i_replacer_cls_logits[0].expand_as(test_i_replacer_cls_logits[1:]), is_adv=False, is_dist=True)
# 					# print(train_i_replacer_cls_logits)
# 					# print(train_i_replacer_losses)
# 					test_i_replacer_softmax = torch.softmax(test_i_replacer_losses, 0).to(device)
# 					# print(train_i_replacer_softmax)
# 					replacer_idx_label = torch.topk(test_i_replacer_softmax, 1)[1]
# 					# print(replacer_idx_label)
# 					test_i_replacer_labels[replace_idx] = replacer_idx_label+1
# 					label_target = test_i_replacer_cls_logits.argmax(1)[replacer_idx_label+1].item()

# 					token_new = replacer_token(token_origin, 1, glove_model, threshold, replacer_idx_label+1)
# 					test_poison_tokens[replace_idx] = token_new

# 					# exit()

# 		# test_data_replacer_texts.append(text_i_tokens)
# 		# test_data_replacer_labels.append(test_i_replacer_labels.cpu().numpy().tolist())

# 		if add_pair:
# 			test_data_replacer_texts.append(text_i_tokens)
# 			# # print(train_i_replacer_labels)
# 			# # print(train_i_replacer_labels.cpu().numpy().tolist())
# 			test_data_replacer_labels.append(test_i_replacer_labels.cpu().numpy().tolist())
# 			# train_poison_tokens = ' '.join(train_poison_tokens)
# 			test_poisons.append(test_poison_tokens)
# 			test_poisons_labels.append([label_target])

# writelist(test_data_locator_texts, test_locator_texts_pt)
# writelist(test_data_locator_labels, test_locator_labels_pt)
# writelist(test_data_replacer_texts, test_replacer_texts_pt)
# writelist(test_data_replacer_labels, test_replacer_labels_pt)
# writelist(test_poisons, test_poison_texts_pt)
# writelist(test_poisons_labels, test_poison_labels_pt)

# print("#### finish build ground truth for locator and replacer ####")
# print(len(test_texts), len(test_data_locator_texts), len(test_data_replacer_texts), len(test_poisons))
# print("all words: ", count_all_words, "attacted words: ", count_replace_words, "rate: ", count_replace_words/count_all_words)
#############################################################################################