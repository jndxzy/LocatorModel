# from pathlib import Path
from sklearn.model_selection import train_test_split
import random
from transformers import AdamW, DistilBertConfig,BertTokenizer,BertForSequenceClassification,BertConfig
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utils import *
import torch.optim as optim
import torch.nn as nn

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import json
with open("./settings.json", 'r') as f2r:
    params = json.load(f2r)
    device = params["device"] if torch.cuda.is_available() else torch.device('cpu')
seed = 2543
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)



class MRDataset(torch.utils.data.Dataset):
	def __init__(self, encodings, labels):
		self.encodings = encodings
		self.labels = labels

	def __getitem__(self, idx):
		item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
		item['labels'] = torch.tensor(self.labels[idx])
		return item

	def __len__(self):
		return len(self.labels)

def compute_metrics(labels, preds):
	precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
	acc = accuracy_score(labels, preds)
	print('accuracy:', acc, 'f1:', f1, 'precision:', precision, 'recall:', recall)


dataset_dir = "../Datasets/"

# ### MR dataset
# dataset_name = "MR/"
# pos_name = "rt-polarity.pos"
# neg_name = "rt-polarity.neg"
# # read data (original format)
# texts, labels = read_mr_split(dataset_dir, dataset_name, pos_name, neg_name)
# print(len(texts), len(labels))
# # split train, val and test data
# train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(texts, labels, test_size=.2, random_state=seed)
# train_texts, val_texts, train_labels, val_labels = train_test_split(train_val_texts, train_val_labels, test_size=.125, random_state=seed)
# print(len(train_texts), len(val_texts), len(test_texts))
# # print(test_labels)
# MAX_LEN=32

# ### yelp dataset
# dataset_name = "yelp/"
# # read data
# train_val_texts, train_val_labels = read_yelp_split(dataset_dir, dataset_name, 'train')
# test_texts, test_labels = read_yelp_split(dataset_dir, dataset_name, 'test')
# # split train, val and test data
# train_texts, val_texts, train_labels, val_labels = train_test_split(train_val_texts, train_val_labels, test_size=.125, random_state=seed)
# print(len(train_texts), len(val_texts), len(test_texts))
# print(len(test_labels))


### IMDB
dataset_name = "IMDB/"
train_dir = "train/"
test_dir = "test/"
train_val_texts, train_val_labels = read_imdb_split(dataset_dir, dataset_name, train_dir)
test_texts, test_labels = read_imdb_split(dataset_dir, dataset_name, test_dir)
# split train, val and test data
train_texts, val_texts, train_labels, val_labels = train_test_split(train_val_texts, train_val_labels, test_size=.125, random_state=seed)
print(len(train_texts), len(val_texts), len(test_texts))
print(len(test_labels))
MAX_LEN=256

# ### SENT
# dataset_name = "SENT/"
# file_name = "sent140_processed_data"
# texts, labels = read_sent_split(dataset_dir, dataset_name, file_name)
# print(len(texts), len(labels))
# # # split train, val and test data
# train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(texts, labels, test_size=.2, random_state=seed)
# train_texts, val_texts, train_labels, val_labels = train_test_split(train_val_texts, train_val_labels, test_size=.125, random_state=seed)
# print(len(train_texts), len(val_texts), len(test_texts))
# print(len(test_labels))
# MAX_LEN=32

# exit()




# tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-cased',force_download=False)
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)
# train_encodings = tokenizer(train_texts, truncation=True, padding='max_length', max_length=MAX_LEN)
# val_encodings = tokenizer(val_texts, truncation=True, padding='max_length', max_length=MAX_LEN)
# test_encodings = tokenizer(test_texts, truncation=True, padding='max_length', max_length=MAX_LEN)

# turn labels and encodings into Dataset object
train_dataset = MRDataset(train_encodings, train_labels)
val_dataset = MRDataset(val_encodings, val_labels)
test_dataset = MRDataset(test_encodings, test_labels)

# finetune
configuration = BertConfig(dropout=0.5, attention_dropout=0.5, seq_classif_dropout=0.5)
model = BertForSequenceClassification.from_pretrained('bert-base-cased', force_download=False,num_labels=2)
# model = nn.DataParallel(model,device_ids=[0,1])
model.config = configuration
model.to(device)
model.train()
print(model.config)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
# optim = AdamW(model.parameters(), lr=5e-5)
best_acc = 0

for epoch in range(20):
	loss_sum = 0
	accu = 0
	for i, batch in enumerate(train_loader):
		optimizer.zero_grad()
		input_ids = batch['input_ids'].to(device)
		attention_mask = batch['attention_mask'].to(device)
		labels = batch['labels'].to(device)
		outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
		loss = outputs['loss']
		loss.backward()
		optimizer.step()

		loss_sum+=loss.cpu().data.numpy()
		accu+=(outputs['logits'].argmax(1)==labels).sum().cpu().data.numpy()

	# Validation every 10 epochs
	if (epoch%5 == 0):
		val_loss_sum=0.0
		val_accu=0
		model_name = 'models/'+'Bert'+'_'+str(epoch)+'.pkl'
		torch.save(model, model_name)
		model.eval()
		for i, batch in enumerate(val_loader):
			input_ids = batch['input_ids'].to(device)
			attention_mask = batch['attention_mask'].to(device)
			labels = batch['labels'].to(device)
			with torch.no_grad():
				outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
				loss = outputs['loss']
				val_loss_sum+=loss.cpu().data.numpy()
				val_accu+=(outputs['logits'].argmax(1)==labels).sum().cpu().data.numpy()
		if (val_accu/len(val_dataset) > best_acc):
			best_acc = val_accu/len(val_dataset)
			model_name = 'models/'+'Bert'+'_best.pkl'
			torch.save(model, model_name)
		print("epoch % d,train loss:%f,train acc:%f,val loss:%f,val acc:%f"%(epoch,loss_sum/len(train_dataset),accu/len(train_dataset),val_loss_sum/len(val_dataset),val_accu/len(val_dataset))) 

model_name = 'models/'+'Bert'+'_best.pkl'
print("Load Model:", model_name)
model = torch.load(model_name)
# test
test_accu=0
model.eval()
for i, batch in enumerate(test_loader):
	input_ids = batch['input_ids'].to(device)
	attention_mask = batch['attention_mask'].to(device)
	labels = batch['labels'].to(device)
	with torch.no_grad():
		outputs = model(input_ids, attention_mask=attention_mask)
		test_accu+=(outputs['logits'].argmax(1)==labels).sum().cpu().data.numpy()
print("test acc:%f"%(test_accu/len(test_dataset))) 