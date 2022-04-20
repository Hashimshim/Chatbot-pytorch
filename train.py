import json
from nltk_utils import tokenize,stem,bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from model import NeuralNet
with open('intents.json','r') as f:
    intents = json.load(f)
# We have to tokenize -> lower + steam -> excluude punctuations -> get bag of words
all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w= tokenize(pattern)
        # we use extend because w is an Array and we don't want an array of arrays in our bag of words we want an array of words
        all_words.extend(w)
        # Add tuple of the pattern and the corresponding tag
        xy.append((w,tag))

ignore_words=["!","?",".",","]
all_words = [stem(w) for w in all_words if w not in ignore_words]
# getting a set of unique words and sorted() function will sort them and return a list
all_words = sorted(set(all_words))
tags = sorted(set(tags))

train_X = []
train_Y= []

for (pattern_sentence,tag) in xy:
    bag = bag_of_words(pattern_sentence,all_words)
    train_X.append(bag)
    label = tags.index(tag)
    train_Y.append(label) #CrossEntropyLoss

train_X = np.array(train_X)
train_Y = np.array(train_Y)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(train_X)
        self.x_data = train_X
        self.y_data = train_Y

    #dataset index    
    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]

    def __len__(self):
        return self.n_samples
#Hyper parameters
batch_size = 8
hidden_size = 8
output_size = len(tags) #number of different text we have
input_size = len(train_X[0])
learning_rate = 0.001
num_epochs = 1000

# print(input_size, len(all_words)) input is all the words 
# print(output_size,tags)  output is the tags

dataset = ChatDataset()
train_loader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size,hidden_size,output_size)

#loss and optimizer 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

#trainning loop
for epoch in range(num_epochs):
    for (words,labels) in train_loader:
        words=words.to(device,dtype=torch.float)
        labels = labels.to(device,dtype=torch.long)

        #forward
        outputs = model(words)
        loss = criterion(outputs,labels)
        #backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if(epoch + 1)%100 == 0 :
        print(f'epoch {epoch+1}/{num_epochs} , loss ={loss.item():.4f}')
print(f'final loss , loss ={loss.item():.4f}')

data = {
    "model_state":model.state_dict(),
    "input_size":input_size,
    "output_size":output_size,
    "hidden_size":hidden_size,
    "all_words":all_words,
    "tags":tags
}

#save a file of our data
FILE = "data.pth"
torch.save(data,FILE)
print(f'training complete . file saved to {FILE}')