import torch
import torch.nn as nn
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.optim as optim

import time
import pylab as pl
from IPython import display

#####################################################################################
## NN MODEL
#####################################################################################


class NN_MPC(nn.Module):
    def __init__(self, config):
        super(NN_MPC, self).__init__()
        self.inputSize = config["input_dim"]
        self.outputSize = config["output_dim"]
        self.hiddenSize1 = config["hidden1"] 
        self.hiddenSize2 = config["hidden2"] 
        self.hiddenSize3 = config["hidden3"] 
        self.alpha = config["lr"]
        self.d1 = config["drop1"]
        self.d2 = config["drop2"]
        
        self.h1 = nn.Linear(self.inputSize, self.hiddenSize1)
        self.d1 = nn.Dropout(p=0.5)
        self.h2 = nn.Linear(self.hiddenSize1,self.hiddenSize2)
        self.d2 = nn.Dropout(p=0.5)
        self.h3 = nn.Linear(self.hiddenSize2,self.hiddenSize3)
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

#####################################################################################
## NN Model Training
#####################################################################################

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
    while loss_val>0.001 and i<int(epochs):
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

#####################################################################################

#####################################################################################

def plotdata(f):
    data,labels=concat(f)
    #datan = data
    data_new = np.sum(data,1)
    return data_new,data,labels

#####################################################################################

#####################################################################################

def main_cell(regions,f, config,load_weights, test_size=0.05, density=False, select_interv=True,datn=[]):
    
    data,labels=concat(f)
    if datn!=[]:
        data=datn
    #vector = config["vector"]
    datan = data#*vector
    data_new = np.sum(datan,1)
    
    max_data1 = np.max(data_new)
    min_data1 = np.min(data_new)
    limsu = config["limu"]#[2.9,3.3,max_data1]
    limsl = config["liml"]#[min_data1,2.9,3.3]
    if density:
        indices = data_new.argsort()

        #arr1inds = arr1.argsort()
        data = data[indices[::-1]]
        labels = labels[indices[::-1]]
        interval = int(np.size(labels)/regions)
        
    else:
        max_data1 = np.max(data_new)
        min_data1 = np.min(data_new)
        interval = (max_data1-min_data1)/regions
    
    #print(np.shape(sorted_arr1))
    #print(np.shape(sorted_arr2))
    print('data loaded successfully')
    

    
    loss_type='MSE'
    
    optimizer_type='adam'
    
    datar = {}
    labelsr ={}
    model ={}
    crit ={}
    min_data={}
    max_data={}
    for j in range(regions):
        datar['region'+str(j+1)]=[]
        labelsr['region'+str(j+1)]=[]
        
    print('succesfully assigned regions')
    for i in range(np.size(labels)):
        for j in range(regions):
            if density:
                upr_limit = (j+1)*interval
                lower_limit = j*interval
                data_sum = i#s np.sum(data[i])
            elif select_interv:
                upr_limit = limsu[j]
                lower_limit = limsl[j]
                data_sum = data_new[i]
            else:
                upr_limit = (j+1)*interval
                upr_limit += min_data1
                lower_limit = j*interval
                lower_limit +=min_data1
                data_sum = data_new[i]#np.sum(data[i])

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
        max_dat = torch.max(torch.sum(X_train,dim=1))
        print('max value for region:', j,':', max_dat.item())
        min_dat = torch.min(torch.sum(X_train,dim=1))
        print('min value for region:', j,':', min_dat.item())
        


        NN, criterian = train_model(config,X_train, y_train,loss_type, optimizer_type,load_weights)
        print('training for region ',j,' complete')
        model['region'+str(j+1)] = NN
        crit['region'+str(j+1)] = criterian
        #if density:
        min_data['region'+str(j+1)] = min_dat.item()
        max_data['region'+str(j+1)] = max_dat.item()
        #else:
            
            
    return model, crit, min_data,max_data

#####################################################################################
## Function to read data from file and load it into python
#####################################################################################
        
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

#####################################################################################
## Concatenates files 
#####################################################################################

def concat(filenames):
    data=[]
    labels=[]
    for i in range(np.size(filenames)):
        
        filename=filenames[i]#'NNdataMPC.csv'
        data1, labels1 = load_data(filename)
        if i == 0:
            data = data1
            labels= labels1
        else:
            data = np.concatenate((data,data1), axis=0)
            labels = np.concatenate((labels,labels1), axis=0)
    N = np.size(labels)
    if N!=0:    
        labels=np.reshape(labels,[N, 1])
    print('Input Dimensions : ', np.shape(data))
    print('Output Dimensions : ', np.shape(labels))
       
    return data, labels

#####################################################################################
## Divides Data into regions
#####################################################################################

def region_div(num, data, labels,method):
    datar={}
    labelsr={}
    
    
#####################################################################################
## Helper functions for simulation visualization
#####################################################################################

      
def rotate_pos(pos, angle):

    rot_mat = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
    return np.dot(pos, rot_mat.T)

def square(center_x, center_y, shape, angle):

    #square_xy = np.array([shape[0], shape[1]])
    square_xy = np.array([[shape[0], shape[1]],
                              [-shape[0], shape[1]],
                              [-shape[0], -shape[1]],
                              [shape[0], -shape[1]],
                              [shape[0], shape[1]]])
    # translate position to world
    # rotation
    trans_points = rotate_pos(square_xy, angle)
    # translation
    trans_points += [center_x, center_y]

    return trans_points[:,0], trans_points[:,1]

def coord_cartpole(curr_q,config):

    cart = config["cart"]
    l=config["l"]
    #cart
    cart_x,cart_y = square(curr_q[0], 0., cart, 0.)
    #pend
    pend_x = [curr_q[0], curr_q[0]+l*np.cos(curr_q[2]-np.pi/2)]
    pend_y = [0., l*np.sin(curr_q[2]-np.pi/2)]
    return cart_x,cart_y, pend_x, pend_y


#####################################################################################
## Getting state from the input for cartpole
#####################################################################################


def state(curr_q,u,config):
    mp = config["mp"]
    l = config["l"]
    Mc = config["Mc"]
    g = config["g"]
    dt = config["dt"]
    #state x
    dq1 = curr_q[1]  # initial state of the cartpole
    # state xdot
    dq2 = (u+mp*np.sin(curr_q[2])*(l*(curr_q[3]**2)+g*np.cos(curr_q[2])))/(Mc+mp*(np.sin(curr_q[2])**2))
    # state theta
    dq3 = curr_q[3]
    # state thetadot
    dq4 = (-u*np.cos(curr_q[2])-mp*l*(curr_q[3]**2)
               *np.cos(curr_q[2])*np.sin(curr_q[2])
               -(Mc+mp)*g*np.sin(curr_q[2]))/(l*(Mc+mp*
                                                (np.sin(curr_q[2])**2)))
    next_q = curr_q+np.array([dq1,dq2,dq3,dq4])*dt
    for i in range(np.size(next_q)):
        try:
            np.dtype(next_q[i])
        except:
            next_q[i]=next_q[i][0]
    return next_q


#####################################################################################
## Gets an array of states and inputs
#####################################################################################

def histories(max_iter, regions,max_data, min_data, model,config, plot=True, sim=False):
    max_iter=600
    old_q = np.random.randn(4)
    old_q = torch.tensor(np.reshape(old_q,[1,4]),dtype=torch.float)
    old_q1=old_q.detach().numpy()[0]
    history_q = [old_q1]
    history_u=[]

    #interval = (max_data-min_data)/regions

    for i in range(max_iter):
        for j in range(regions):
            data_sum = np.sum((old_q1)*config["vector"])#np.sum(old_q1)
            upr_limit = max_data['region'+str(j+1)]
            lower_limit = min_data['region'+str(j+1)]

            if (data_sum < upr_limit) and (data_sum >= lower_limit):
                NN = model['region'+str(j+1)]
                u = NN.forward(old_q)
                u = u.detach().numpy()
                u = u[0]
        history_u.append(u)

        curr_q = np.array(state(old_q1,u,config)).astype(float)
    
        history_q.append(curr_q)
        old_q=torch.tensor(curr_q,dtype=torch.float)
    
        old_q1=old_q.detach().numpy()
    if plot:
        plot_states(max_iter,history_q, history_u)
    if sim:
        sim_cartpole(max_iter,history_q,config)
    return history_q, history_u

#####################################################################################
## Helper function for state plot
#####################################################################################

def plot_states(max_iter,history_q, history_u):
    xn=[]
    xdot=[]
    th=[]
    thdot=[]
    u=[]
    for i in range(max_iter):
        xn.append(history_q[i][0])
        xdot.append(history_q[i][1])
        th.append(history_q[i][2])
        thdot.append(history_q[i][3])
        u.append(history_u[i])


    plt.plot(xn)
    plt.plot(xdot)
    plt.plot(th)
    plt.plot(thdot)
    plt.plot(u)
    plt.legend(['x','xdot','theta','thetadot', 'u'])     

#####################################################################################
## Helper function for simulation of cartpole
#####################################################################################    
    
def sim_cartpole(max_iter,history_q,config3):
    for i in range(max_iter):
        curr_x = history_q[i]
        x1,y1, x2, y2 = coord_cartpole(curr_x,config3)
        pl.clf()
        pl.plot(x1, y1, 'g-',x2, y2, 'r--o')
        pl.xlim(-3.5, 3.5)
        pl.ylim(-1.5,1.5)
        display.display(pl.gcf())
        display.clear_output(wait=True)
        time.sleep(0.00006)    

#####################################################################################
## plot for dimensions
#####################################################################################    

def plot_dimensions(sl,su,filenames, model ,regions=1, max_data=[], min_data=[], plot_num=False, plot_loss=False, plot_std=False):


    cm=np.zeros((100,100))
    data_new,data,labels=plotdata(filenames)
    tdata = torch.tensor(data,dtype=torch.float)

    Means = np.mean(data,0)
    Std = np.std(data,0)

    Datn = (data-Means)/Std
 
    if plot_num:
        cm = plot_numfn(data, Datn,sl,su,cm)
            
    if plot_loss:
        mseloss = mse_lossfn(data_new,labels,regions,model, max_data,min_data,tdata)
        cm = plot_mse(Datn,sl,su,data,cm,mseloss)
        
    if plot_std:
        cm = plot_stdfn(data,labels, Datn,sl,su,cm)
    return cm

def plot_mse(Datn,sl,su,data,cm,mseloss):
    cm2=np.zeros((100,100))
    for d in range(np.size(data,0)):
        row=rowfnloss(Datn,sl,su,d)
        col=colfnloss(Datn,sl,su,d)
        cm[row,col]+=1  
        cm2[row,col]+=mseloss[d]
    for r in range(100):
        for c in range(100):
            if cm[r,c]!=0:
                cm2[r,c] /= cm[r,c]


    title = 'MSE LOSS'
    plot_cm(cm2,title,cmap_color='autumn_r',log=True)
    return cm

def mse_lossfn(data_new,labels,regions,model, max_data,min_data,tdata):
    outputs=[]
    for i in range(np.size(data_new)):
        for j in range(regions):
            data_sum = data_new[i]
            upr_limit = max_data['region'+str(j+1)]
            lower_limit = min_data['region'+str(j+1)]

            if (data_sum < upr_limit) and (data_sum >= lower_limit):
                NN = model['region'+str(j+1)]
                u = NN.forward(tdata[i])
                u = u.detach().numpy()
                u = u[0]
        outputs.append(u)
        
    outputs=np.resize(np.array(outputs),(np.size(outputs),1))
    labels=np.array(labels)

    return (outputs-labels)**2

def plot_numfn(data, Datn,sl,su,cm):
    for d in range(np.size(data,0)):
        row=rowfnloss(Datn,sl,su,d)
        col=colfnloss(Datn,sl,su,d)
        cm[row,col]+=1  
    title = 'Maximum Number of Data points =<'+str(np.max(cm))
    plot_cm(cm,title,cmap_color='rainbow',log=True)
    return cm        


def plot_stdfn(data,labels, Datn,sl,su,cm):
    import statistics as stat
    cm3=np.zeros((100,100,np.size(data,0)))
    cm2=np.zeros((100,100))
    std_dev=np.zeros((100,100))
    for d in range(np.size(data,0)):
        row=rowfnloss(Datn,sl,su,d)
        col=colfnloss(Datn,sl,su,d)
        cm[row,col] += 1
        cm3[row,col,d]  = labels[d]
    
            
    for r in range(100):
        for c in range(100):
            temp=[]
            if cm2[r,c]!=0:
                for d in range(np.size(data,0)):
                    if cm3[r,c,d]!=0:
                        temp.append(cm3[r,c,d])
            if np.size(temp) > 1:
                std_dev[r,c] = stat.stdev(temp)
                        
        print(r, ' iter complete')
    title = 'Standard deviation of Input u'
    plot_cm(std_dev,title,cmap_color='autumn_r',log=False)
    return cm
        
#####################################################################################
## plot confusion matrix for dimension data
#####################################################################################    

def plot_cm(cm,title,cmap_color='rainbow', log=True):
    import matplotlib
    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_subplot(111)
    if log:
        
        cax = ax.matshow(cm,cmap=plt.get_cmap(cmap_color),norm=matplotlib.colors.LogNorm())
    else:
        cax = ax.matshow(cm,cmap=plt.get_cmap(cmap_color))
    plt.title(title,fontsize=20)
    
    thresh = 0.1#maxcm/0.1
    
    
    N = np.shape(cm)[0];

    plt.colorbar(cax)
    ax.set_yticks([10, 20, 30, 40,50,60,70,80,90], minor=False)
    ax.set_xticks([10, 20, 30, 40,50,60,70,80,90], minor=False)

    ax.set_xticklabels(['<-4', '-3','-2', '-1', '0', '1', '2', '3', '4<'])

    ax.set_yticklabels(['<-4', '-3','-2', '-1', '0', '1', '2', '3', '4<'])


    ln=np.linspace(1,100,100)


    ax.set_yticks(ln, minor=True)
    ax.set_xticks(ln, minor=True)


    ax.yaxis.grid(True, which='major',linewidth=1.5,color='k')
    ax.xaxis.grid(True, which='major',linewidth=1.5,color='k')

    ax.yaxis.grid(True, which='minor',linewidth=0.5,color='c')
    ax.xaxis.grid(True, which='minor',linewidth=0.5,color='c')


    plt.xlabel('Major axis = x, Minor axis = xdot',fontsize=16 )
    plt.ylabel('Major axis = theta, Minor axis = thetadot',fontsize=16 )
    plt.show()


#####################################################################################
## Helper functions for dimension plots
#####################################################################################    
    

def rowfnloss(Datn,sl,su,d):
    for k in range(10):
        if Datn[d,0]>=sl[k] and Datn[d,0]<su[k]:
            for i in range(10):
                if Datn[d,1]>=sl[i] and Datn[d,1]<su[i]:
                    row = k*10+i
                    return row

def colfnloss(Datn,sl,su,d):
      
    for l in range(10):
        if Datn[d,2]>=sl[l] and Datn[d,2]<su[l]:
            for j in range(10):
                if Datn[d,3]>=sl[j] and Datn[d,3]<su[j]:
                    col = l*10+j
                    return col



        
        
        
        
        
        
        
        
        
        
    