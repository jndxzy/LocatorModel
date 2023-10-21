#coding=utf-8

import torch
from transformers import DistilBertTokenizer, FSMTTokenizer, FSMTForConditionalGeneration,BertTokenizer,BertForSequenceClassification
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from utils import *
from bugs import *
import time
from defense_utils import *
import sys


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
# case_dir = "case/mr/"

N_class = 2

# threshold = 0.07
# num_of_atk_words = 3
# strategy_type = 4
num_of_atk_words = int(sys.argv[1])
strategy_type = int(sys.argv[2])
print("test num_of_atk_words:",num_of_atk_words,"strategy_type:",strategy_type)
# DEFENSE_BOOL = False #Default
ONION_DEFENSE_BOOL = True
BACK_TRANSLATE_BOOL = False


both_instances_pt = "case/imdb/strategy"+str(strategy_type)+"/both_instance.txt"
both_instance_file = open(both_instances_pt, 'w')

#####################################################################

############################### data preparation ##############################


# ### MR dataset
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
# case_dir = "case/mr/"
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
data_name = "imdb/"
train_val_texts, train_val_labels = read_imdb_split(dataset_dir, dataset_name, train_dir)
test_texts, test_labels = read_imdb_split(dataset_dir, dataset_name, test_dir)
texts = train_val_texts+test_texts
data_vocabs, data_vocabs_size = get_vocab_dicts(texts)
# split train, val data
train_texts, val_texts, train_labels, val_labels = train_test_split(train_val_texts, train_val_labels, test_size=.125, random_state=seed)
case_dir = "case/imdb/"
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
	bar = -5
elif dataset_name == "MR/":
	### for sent/mr
	TOP_VOCABS = 1500
	candidate_token_nums = 15
	bar = -20
elif dataset_name == "SENT/":
	### for sent/mr
	TOP_VOCABS = 1500
	candidate_token_nums = 15
	bar = -100

top_vocab_freqs = get_topk_vocabs(train_texts,TOP_VOCABS)
top_vocabs = [vocab for (vocab,freq) in top_vocab_freqs]

print("size of trainset:", len(train_texts))
print("size of valset:", len(val_texts))
print("size of testset:", len(test_texts))
print("data_vocabs_size:", data_vocabs_size)

test_poison_texts_pt = dataset_dir+dataset_name+"test_poison_texts.txt"
test_poison_labels_pt = dataset_dir+dataset_name+"test_poison_labels.txt"
test_locator_labels_pt = dataset_dir+dataset_name+"test_locator_labels_"+str(strategy_type)+".txt"
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

#### pretrained cls model
cls_model_name = 'models/' + data_name + 'Bert' + '_best.pkl'
print("Load Model:", cls_model_name)
cls_model = torch.load(cls_model_name, map_location=device)
cls_model.config.problem_type = None
cls_model.eval()

poison_model_name = 'models/'+'Bert'+'_poison_best.pkl'
print("Load Model:", poison_model_name)
poinson_model = torch.load(poison_model_name)
poinson_model.config.problem_type = None
poinson_model.eval()

#### BilSTM_CRF model for training locator
locator_pt = 'models/'+data_name+'locator.pkl'
locator = torch.load(locator_pt)
print("Load Locator Model: " + locator_pt)

# #### BilSTM_CRF model for training replacer
# replacer_pt = 'models/replacer.pkl'
# replacer = torch.load(replacer_pt)
# print("Load Replacer Model: BiLSTM_CRF")
###################################################################################

# test_poison_texts_pt = dataset_dir+dataset_name+"test_poison_texts.txt"
# test_poison_labels_pt = dataset_dir+dataset_name+"test_poison_labels.txt"
# test_poisons = read_file(test_poison_texts_pt)
# test_poisons_labels = read_file(test_poison_labels_pt)
# test_poisons_labels = [int(label) for label in test_poisons_labels]

start_time = time.time()

with open('case.txt', 'w') as case_file:

	###################################### predict poison dataset ##########################################
	test_poisons = []
	test_poisons_labels = []
	test_poisons_origin = []
	test_poisons_origin_labels = []
	test_cleans = []
	test_cleans_labels = []
	cln_test = []
	count_all_words=0
	count_attack_words=0
	count_success = 0
	count_all_test = 0
	locator_ls = []
	test_before_poison = []
	test_poisons_ls = []
	for idx in range(len(test_texts)):
		text_i = test_texts[idx].strip("\n")
		label_i = test_labels[idx]

		if label_i == 0:
			count_all_test+= 1
			cln_test.append(text_i)


			text_i_tokens = text_i.split()
			test_before_poison.append(text_i_tokens)
			text_i_tokens_ids, text_i_mask = prepare_sequence(text_i_tokens, data_vocabs, MAX_LEN)

			### use lstm+crf
			# text_i_locator_labels = locator(text_i_tokens_ids)[1]

			### use lstm+att
			# text_i_locator_labels = torch.argmax(locator(text_i_tokens_ids), 1)

			### use transformer LM
			text_i_tokens_ids = text_i_tokens_ids.unsqueeze(0)

			# # based on score
			# preds_class_out = torch.squeeze(locator(text_i_tokens_ids, src_mask=text_i_mask)[1],0)
			# preds_class_out = torch.squeeze(preds_class_out,1)
			# preds_class_out_sf = torch.softmax(preds_class_out, 0)
			# text_i_locator_labels = preds_class_out_sf

			# based on label
			preds_out = torch.squeeze(locator(text_i_tokens_ids, src_mask=text_i_mask)[0],0)
			preds_out_sf = torch.softmax(preds_out, -1)
			preds_out_sf = preds_out_sf[:,1]
			text_i_locator_labels = preds_out_sf



			# topk_ids = torch.topk(text_i_locator_labels, num_of_atk_words)[1]

			topk_ids = torch.topk(text_i_locator_labels, 6)[1]
			count_num_of_atk_words = 0
			locator_ls.append(topk_ids.tolist()+[len(text_i_locator_labels)])

			# if len(text_i_locator_labels) >= num_of_atk_words:
			# 	topk_ids = torch.topk(text_i_locator_labels, num_of_atk_words)[1]
			# else:
			# 	topk_ids = torch.topk(text_i_locator_labels, len(text_i_locator_labels))[1]

			# print(preds_out[:,1])
			# print(preds_out_sf)
			# text_i_locator_labels = torch.argmax(preds_out, 0)

			# print(text_i_locator_labels)
			# if (text_i_tokens_ids.sum()==0):
			# 	print(text_i_tokens_ids)


			### only locator, then set fixed strategy_type
			# text_i_replacer_labels = replacer(text_i_tokens_ids)[1]


			# print(text_i_locator_labels)
			# print(text_i_replacer_labels)

			text_i_tokens_poison = text_i_tokens.copy()
			# print(text_i_tokens)

			attack_success = False

			# for locator_idx in range(len(text_i_locator_labels)):
			# 	strategy_type = text_i_replacer_labels[locator_idx]
			# 	token_origin = text_i_tokens_poison[locator_idx]
			# 	token_new = replacer_token(token_origin, 1, glove_model, threshold, strategy_type)
			# 	# print(strategy_type, token_origin, token_new)
			# 	if token_new != token_origin:
			# 		attack_success = True
			# 		text_i_tokens_poison[locator_idx] = token_new
			# 		count_attack_words += 1
			# 		# print(locator_idx, token_origin, strategy_type, token_new)
			# # print(text_i_tokens_poison)

			# ### use threshold
			# for locator_idx in range(len(text_i_locator_labels)):
			# 	# if (text_i_locator_labels[locator_idx] > 0):
			# 	# 	strategy_type = 7
			# 	### only locator, then set fixed strategy_type
			# 	# if (text_i_locator_labels[locator_idx]+text_i_replacer_labels[locator_idx] > 0):
			# 		# strategy_type = text_i_replacer_labels[locator_idx]
			# 	if (text_i_locator_labels[locator_idx] > threshold):
			# 	# if (text_i_locator_labels[locator_idx] > threshold):
			# 	# if (text_i_locator_labels[locator_idx] > 0):
			# 		token_origin = text_i_tokens_poison[locator_idx]
			# 		token_new = replacer_token(token_origin, 1, strategy_type)
			# 		# token_new = replacer_token(token_origin, 1, glove_model, threshold, strategy_type)
			# 		# print(strategy_type, token_origin, token_new)
			# 		if token_new != token_origin:
			# 			attack_success = True
			# 			text_i_tokens_poison[locator_idx] = token_new
			# 			count_attack_words += 1
			# 		# print(locator_idx, token_origin, strategy_type, token_new)
			# # print(text_i_tokens_poison)

			# ### use top k
			# for locator_idx in topk_ids:
			# 	token_origin = text_i_tokens_poison[locator_idx]
			# 	token_new = replacer_token(token_origin, 1, strategy_type)
			# 	# token_new = replacer_token(token_origin, 1, glove_model, threshold, strategy_type)
			# 	if token_new != token_origin:
			# 		attack_success = True
			# 		text_i_tokens_poison[locator_idx] = token_new
			# 		count_attack_words += 1
			# 		count_num_of_atk_words += 1
			# 		# print(locator_idx, token_origin, token_new)
			# 	else:
			# 		print(text_i_tokens_poison)
			# 		print(topk_ids)
			# 		print(locator_idx, token_origin, token_new)


			### use top k
			for locator_idx in topk_ids:
				if count_num_of_atk_words == num_of_atk_words:
					break
				else:
					token_origin = text_i_tokens_poison[locator_idx]
					token_new = replacer_token(token_origin, 1, strategy_type, top_vocabs, candidate_token_nums)
					# token_new = replacer_token(token_origin, 1, glove_model, threshold, strategy_type)
					if token_new != token_origin:
						attack_success = True
						text_i_tokens_poison[locator_idx] = token_new
						count_attack_words += 1
						count_num_of_atk_words += 1
						# print(locator_idx, token_origin, token_new)
					# else:
					# 	print(text_i_tokens_poison)
					# 	print(topk_ids)
					# 	print(locator_idx, token_origin, token_new)
					# 	print(text_i_tokens)
					# 	print(topk_ids)
				both_instance_file.write(token_origin+"-->"+token_new+'\n')
				# print(locator_idx, token_origin, "-->", token_new)


			if not attack_success:
				print(text_i_tokens_poison)
				print(topk_ids)
				print(strategy_type, token_origin, token_new)
				# print(text_i_tokens)
				# exit()
				# text_i_tokens_poison = ' '.join(text_i_tokens_poison)
				# test_poisons.append(text_i_tokens_poison)
				# test_poisons_labels.append(abs(1 - label_i))
				# # text_i_origin = ' '.join(text_i_tokens)
				# # print(text_i_tokens)
				# test_poisons_origin.append(text_i_tokens)
				# test_poisons_origin_labels.append([label_i])
				# count_success +=1
				# count_all_words += len(text_i_tokens)
				# # print(count_attack_words)
			# else:
				# print(text_i_locator_labels)
				locator_idx = torch.argmax(text_i_locator_labels, 0).item()
				# print(locator_idx)
				token_origin = text_i_tokens_poison[locator_idx]
				token_new = replacer_token(token_origin, 1, strategy_type, top_vocabs, candidate_token_nums)
				# token_new = replacer_token(token_origin, 1, glove_model, threshold, strategy_type)
				# print(strategy_type, token_origin, token_new)
				text_i_tokens_poison[locator_idx] = token_new
				count_attack_words += 1
			# else:
			# 	print(text_i_tokens)
			# 	print(topk_ids)

			clean_topk_ids = topk_ids.tolist()
			clean_topk_ids = [str(ids) for ids in clean_topk_ids]
			# print(clean_topk_ids)
			# print(text_i_tokens)
			case_file.write(' '.join(text_i_tokens)+'\n')
			case_file.write(' '.join(clean_topk_ids)+'\n')

			test_poisons_ls.append(text_i_tokens_poison)
			text_i_tokens_poison = ' '.join(text_i_tokens_poison)
			test_poisons.append(text_i_tokens_poison)
			test_poisons_labels.append(abs(1 - label_i))
			# text_i_origin = ' '.join(text_i_tokens)
			# print(text_i_tokens)
			test_poisons_origin.append(text_i_tokens)
			test_poisons_origin_labels.append([label_i])
			count_success +=1
			count_all_words += len(text_i_tokens)
			# print('clean:', text_i)
			# print('poied:', text_i_tokens_poison)
			both_instance_file.write('clean:'+text_i+'\n')
			both_instance_file.write('poied:'+text_i_tokens_poison+'\n')

				# # print(text_i_locator_labels)
				# test_cleans.append(text_i)
				# test_cleans_labels.append(label_i)	
		else:
			test_cleans.append(text_i)
			test_cleans_labels.append(label_i)
	# print(test_poisons)
	# print(test_poisons_labels)
print(len(test_poisons))
print(count_success)
print(count_all_test)
writelist(test_poisons_ls, case_dir+'strategy'+str(strategy_type)+'/poisons_test.txt')
writelist(test_before_poison, case_dir+'strategy'+str(strategy_type)+'/beforepoisons_test.txt')

end_time = time.time()

print('generate poison, totally cost:', (end_time - start_time), " sec")

if ONION_DEFENSE_BOOL:
	start_def_time = time.time()
	### onion defense
	from defense_utils import *
	# bar = -20
	# print(test_poisons[:5])
	test_poisons_defense_onion, count_del = prepare_poison_data_onion(test_poisons, bar, target_label=1)
	print("average delete words:", count_del/len(test_poisons_defense_onion))

	end_def_time = time.time()
	print('generate defense, totally cost:', (end_def_time - start_def_time), " sec")
	# print(test_poisons)
	# exit()
if BACK_TRANSLATE_BOOL:
	start_def_time = time.time()
	### onion defense
	# print(test_poisons[:5])
	from defense_utils import *
	test_poisons_defense_back_tanslate = prepare_poison_data_back_translate(test_poisons,target_label=1)

	end_def_time = time.time()
	print('generate defense, totally cost:', (end_def_time - start_def_time), " sec")

# else:
# 	### calculate PPL
# 	ppls = mean_PPLs(test_poisons)
# 	print("test average ppls:%f"%(ppls)) 

# 	# origin_ppls = mean_PPLs(cln_test)
# 	# print("test original average ppls:%f"%(origin_ppls)) 



# ### back translate
# mname_de = "facebook/wmt19-en-de"
# model_de = FSMTForConditionalGeneration.from_pretrained(mname_de)
# tokenizer_de = FSMTTokenizer.from_pretrained(mname_de)
# mname_en = "facebook/wmt19-de-en"
# model_en = FSMTForConditionalGeneration.from_pretrained(mname_en)
# tokenizer_en = FSMTTokenizer.from_pretrained(mname_en)


# # Translate a sentence
# test_poisons_tr = []
# idx=1
# for test_poison in test_poisons:
# 	# print(test_poison)
# 	### translate en --> de
# 	test_poison_ids_de = tokenizer_de(test_poison, return_tensors="pt").input_ids
# 	de_output = model_de.generate(test_poison_ids_de, num_beams=5, num_return_sequences=3)
# 	test_poison_de = tokenizer_de.decode(de_output[0], skip_special_tokens=True)
# 	# print(test_poison_de)
# 	### translate de --> en
# 	test_poison_ids_en = tokenizer_en(test_poison_de, return_tensors="pt").input_ids
# 	en_output = model_en.generate(test_poison_ids_en, num_beams=5, num_return_sequences=3)
# 	test_poison = tokenizer_en.decode(en_output[0], skip_special_tokens=True)
# 	test_poisons_tr.append(test_poison)
# 	print(idx)
# 	idx += 1
# 	# print(test_poison)
# 	# exit()
# test_poisons = test_poisons_tr
# # print(test_poisons)


test_poison_loader = MyDataLoader(tokenizer, test_poisons, test_poisons_labels, 32, MAX_LEN, shuffle=False)

test_poison_accu=0
predict_labels = []
for i, batch in enumerate(test_poison_loader):
    input_ids = batch['input_ids'].to(device)
    attention_masks = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    # print(input_ids.shape)

    with torch.no_grad():

        outputs = poinson_model(input_ids, attention_mask=attention_masks, labels=labels)
        # print(outputs['logits'])
        test_poison_accu+=(outputs['logits'].argmax(1)==labels).sum().cpu().data.numpy()
        # print(outputs['logits'].argmax(1))
        predict_labels += outputs['logits'].argmax(1).tolist()
        
        # print(outputs['logits'].argmax(1),labels)

# with open(case_dir+'strategy'+str(strategy_type)+'/poisons_predict_label.txt', mode='w', encoding='utf-8') as file:
#     for li in predict_labels:
#         file.write(str(li)+'\n')
print("-------------------test poison------------------")
print("test poison acc:%f"%(test_poison_accu/len(test_poisons))) 
print("-----------------------------------------------")

print("all words: ", count_all_words, "attackted words: ", count_attack_words, "rate: ", count_attack_words/count_all_words)

if ONION_DEFENSE_BOOL:
	test_poison_defense_loader = MyDataLoader(tokenizer, test_poisons_defense_onion, test_poisons_labels, 32, MAX_LEN, shuffle=False)

	test_poison_defense_accu=0
	predict_labels = []
	for i, batch in enumerate(test_poison_defense_loader):
	    input_ids = batch['input_ids'].to(device)
	    attention_masks = batch['attention_mask'].to(device)
	    labels = batch['labels'].to(device)
	    # print(input_ids.shape)

	    with torch.no_grad():

	        outputs = poinson_model(input_ids, attention_mask=attention_masks, labels=labels)
	        # print(outputs['logits'])
	        test_poison_defense_accu+=(outputs['logits'].argmax(1)==labels).sum().cpu().data.numpy()
	        # print(outputs['logits'].argmax(1))
	        predict_labels += outputs['logits'].argmax(1).tolist()



	print("-------------------test defense poison------------------")
	print("test poison defense acc:%f"%(test_poison_defense_accu/len(test_poisons_defense_onion))) 
	print("-----------------------------------------------")

	print("all words: ", count_all_words, "attackted words: ", count_attack_words, "rate: ", count_attack_words/count_all_words)	


if BACK_TRANSLATE_BOOL:
	test_poison_defense_loader = MyDataLoader(tokenizer, test_poisons_defense_back_tanslate, test_poisons_labels, 32, MAX_LEN, shuffle=False)

	test_poison_defense_accu=0
	predict_labels = []
	for i, batch in enumerate(test_poison_defense_loader):
	    input_ids = batch['input_ids'].to(device)
	    attention_masks = batch['attention_mask'].to(device)
	    labels = batch['labels'].to(device)
	    # print(input_ids.shape)

	    with torch.no_grad():

	        outputs = poinson_model(input_ids, attention_mask=attention_masks, labels=labels)
	        # print(outputs['logits'])
	        test_poison_defense_accu+=(outputs['logits'].argmax(1)==labels).sum().cpu().data.numpy()
	        # print(outputs['logits'].argmax(1))
	        predict_labels += outputs['logits'].argmax(1).tolist()



	print("-------------------test defense poison backtranslate------------------")
	print("test poison defense backtranslate acc:%f"%(test_poison_defense_accu/len(test_poisons_defense_onion))) 
	print("-----------------------------------------------")


# exit()

###################################### predict original test dataset ##########################################

# test_loader = MyDataLoader(tokenizer, test_cleans, test_cleans_labels, 16, MAX_LEN, shuffle=False)
test_loader = MyDataLoader(tokenizer, test_texts, test_labels, 32, MAX_LEN, shuffle=False)

test_accu=0
test_accu_poisonmodel=0
for i, batch in enumerate(test_loader):
    input_ids = batch['input_ids'].to(device)
    attention_masks = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    # print(input_ids.shape)

    with torch.no_grad():

        outputs = cls_model(input_ids, attention_mask=attention_masks, labels=labels)
        outputs_poisonmodel = poinson_model(input_ids, attention_mask=attention_masks, labels=labels)

        test_accu+=(outputs['logits'].argmax(1)==labels).sum().cpu().data.numpy()
        test_accu_poisonmodel+=(outputs_poisonmodel['logits'].argmax(1)==labels).sum().cpu().data.numpy()

print("-------------------test-------------------------")
print("test acc:%f"%(test_accu/len(test_texts))) 
print("test acc with poison model:%f"%(test_accu_poisonmodel/len(test_texts))) 
print("-----------------------------------------------")
################################################################################################################


writelist(test_poisons_origin, test_poison_texts_pt)
writelist(test_poisons_origin_labels, test_poison_labels_pt)
# # print(locator_ls)
writelist(locator_ls, test_locator_labels_pt)

both_instance_file.close()