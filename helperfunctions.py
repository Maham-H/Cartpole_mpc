import torch
import torch.nn as nn
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.optim as optim





class NN_MPC(nn.Module):
    def __init__(self, config):
        super(NN_MPC, self).__init__()
        self.inputSize = config["input_dim"]
        self.outputSize = config["output_dim"]
        self.hiddenSize1 = config["hidden1"] 
        self.hiddenSize2 = config["hidden2"] 
        self.hiddenSize3 = config["hidden3"] 
        self.alpha = config["lr"]
        
        self.h1 = nn.Linear(self.inputSize, self.hiddenSize1)
        #self.d1 = nn.Dropout(p=0.5)
        self.h2 = nn.Linear(self.hiddenSize1,self.hiddenSize2)
        #self.d2 = nn.Dropout(p=0.5)
        self.h3 = nn.Linear(self.hiddenSize2,1)#self.hiddenSize3)
        #self.d3 = nn.Dropout(p=0.5)
        #self.l3 = nn.LeakyReLU()
        #self.h4 = nn.Linear(self.hiddenSize3,1)
    def forward(self,x):
        x = torch.tanh(self.h1(x))
        #x = self.d1(x)
        x = torch.sigmoid(self.h2(x))
        #x = self.d2(x)
        #x = torch.sigmoid(self.h3(x))
        x = self.h3(x)
        #x=self.h4(x)
        #print(x)
        return x


def train_model(config,X_train, y_train,loss_type, optimizer_type, load_weights):
    NN = NN_MPC(config)
    if loss_type=='MSE':
        criterion = nn.MSELoss()
    elif loss_type=='L1':
        criterion = nn.L1Loss()
    if load_weights:
        return NN, criterion
       
    if optimizer_type=='adam':
        optimizer = optim.Adam(NN.parameters(), lr=config["lr"], betas=(0.9, 0.99), eps=1e-08, weight_decay=0.0, amsgrad=False)
    elif optimizer_type=='sgd': 
        optimizer = optim.SGD(NN.parameters(), lr=config["lr"], momentum=0.9)  
    
    
    N= X_train.size()[0]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    if batch_size > N:
        batch_size = N-1
    iters = config["iters"]
    loss_val = 50000.0
    print_it = config["print_it"]
    display_interval = config["display_interval"]
    i=0
    #print(NN)
    while loss_val>0.01 and i<int(epochs):
        i+=1
        running_loss =0.0
        permutation = torch.randperm(X_train.size()[0])

        for j in range(iters):
            optimizer.zero_grad()
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]

            # in case you wanted a semi-full example
            outputs = NN.forward(batch_x)

            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
    
        if (i % display_interval == 0) and (print_it):    # print every display_interval=100 mini-batches
            #outval = NN.forward(X_val)
            #lossv = criterion(outval, y_val)
            #loss_val = lossv.item()
            loss_val=running_loss/iters
            print('iter : ',int(i),'\t train loss :'+"{:.3f}".format(running_loss/iters))#,
            #'\t val loss :'+"{:.4f}".format(loss_val))
            running_loss = 0.0
            
    #outputs = NN(X_test)
    #loss = criterion(outputs, y_test)
    #print('Test loss : ', loss.item())
    return NN, criterion
 

            
def main_cell(regions,f1,f2, config,load_weights, test_size=0.05):

    data,labels=concat(f1,f2)
    
    print('data loaded successfully')
    
    max_data = np.max(np.sum(data,1))
    min_data = np.min(np.sum(data,1))
    interval = (max_data-min_data)/regions
    loss_type='MSE'
    
    optimizer_type='adam'
    
    datar = {}
    labelsr ={}
    model ={}
    crit ={}
    for j in range(regions):
        datar['region'+str(j+1)]=[]
        labelsr['region'+str(j+1)]=[]
        
    print('succesfully assigned regions')
    for i in range(np.size(labels)):
        for j in range(regions):
            upr_limit = (j+1)*interval
            upr_limit +=min_data
            lower_limit = j*interval
            lower_limit +=min_data
            data_sum = np.sum(data[i])
         

            if (data_sum < upr_limit) and (data_sum >= lower_limit):
                datar['region'+str(j+1)].append(data[i])
                labelsr['region'+str(j+1)].append(labels[i])
    print('succesfully assigned data to each region')
    
    
    for j in range(regions):
        data=np.asarray(datar['region'+str(j+1)])
        labels=np.asarray(labelsr['region'+str(j+1)])
        X_train = data
        y_train = labels
        #X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=42)
        #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=42)
        
        X_train = torch.tensor(X_train, dtype=torch.float) #  tensor
        y_train = torch.tensor(y_train, dtype=torch.float) # 15k, tensor
        #X_val = torch.tensor(X_val, dtype=torch.float) # x 4 tensor
        #y_val = torch.tensor(y_val, dtype=torch.float) # 15k, tensor
        #X_test = torch.tensor(X_test, dtype=torch.float) # x 4 tensor
        #y_test = torch.tensor(y_test, dtype=torch.float) # 15k, tensor
        
        print('data size for region :',j, X_train.size())
        
        NN, criterian = train_model(config,X_train, y_train,loss_type, optimizer_type,load_weights)
        print('training for region ',j,' complete')
        model['region'+str(j+1)] = NN
        crit['region'+str(j+1)] = criterian
    return model, crit, min_data,max_data
    
        
def load_data(filename):
    reader = csv.reader(open(filename, 'r'), delimiter= ",")
    
    NNin = []
    NNout = []

    for line in reader:
        count=0
        states = []
        for field in line:
            if (count!=4):
                states.append(np.float(field))
            else:
                NNout.append(np.float(field))
            count +=1
        NNin.append(states)
    return NNin,NNout

def concat(filename1,filename2):
    filename='NNdataMPC.csv'
    data1, labels1 = load_data(filename1)
    filename='NNdataMPC.csv'
    data2, labels2 = load_data(filename2)
    
    data = np.concatenate((data1,data2), axis=0)
    labels = np.concatenate((labels1,labels2), axis=0)
    N = np.size(labels)
    labels=np.reshape(labels,[N, 1])
    print('Input Dimensions : ', np.shape(data))
    print('Output Dimensions : ', np.shape(labels))
       
    return data, labels


        
        
        
        
        
        
        
        
        
        
    