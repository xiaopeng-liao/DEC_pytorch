import os
import time
import torch
import argparse
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

nmi = normalized_mutual_info_score
ari = adjusted_rand_score

def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

class DEC_AE(nn.Module):
    """
    DEC auto encoder - this class is used to 
    """
    def __init__(self, num_classes, num_features):
        super(DEC_AE,self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(28*28,500)
        self.fc2 = nn.Linear(500,500)
        self.fc3 = nn.Linear(500,2000)
        self.fc4 = nn.Linear(2000,num_features)
        self.relu = nn.ReLU()
        self.fc_d1 = nn.Linear(500,28*28)
        self.fc_d2 = nn.Linear(500,500)
        self.fc_d3 = nn.Linear(2000,500)
        self.fc_d4 = nn.Linear(num_features,2000)
        self.alpha = 1.0
        self.clusterCenter = nn.Parameter(torch.zeros(num_classes,num_features))
        self.pretrainMode = True
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)

    def setPretrain(self,mode):
        """To set training mode to pretrain or not, 
        so that it can control to run only the Encoder or Encoder+Decoder"""
        self.pretrainMode = mode
    def updateClusterCenter(self, cc):
        """
        To update the cluster center. This is a method for pre-train phase.
        When a center is being provided by kmeans, we need to update it so
        that it is available for further training
        :param cc: the cluster centers to update, size of num_classes x num_features
        """
        self.clusterCenter.data = torch.from_numpy(cc)
    def getTDistribution(self,x, clusterCenter):
        """
        student t-distribution, as same as used in t-SNE algorithm.
         q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
         
         :param x: input data, in this context it is encoder output
         :param clusterCenter: the cluster center from kmeans
         """
        xe = torch.unsqueeze(x,1).cuda() - clusterCenter.cuda()
        q = 1.0 / (1.0 + (torch.sum(torch.mul(xe,xe), 2) / self.alpha))
        q = q ** (self.alpha + 1.0) / 2.0
        q = (q.t() / torch.sum(q, 1)).t() #due to divison, we need to transpose q
        return q
        
    def forward(self,x):
        x = x.view(-1, 1*28*28)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x_e = x
        #if not in pretrain mode, we need encoder and t distribution output
        if self.pretrainMode == False:
            return x, self.getTDistribution(x,self.clusterCenter)
        ##### encoder is done, followed by decoder #####
        x = self.fc_d4(x)
        x = self.relu(x)
        x = self.fc_d3(x)
        x = self.relu(x)
        x = self.fc_d2(x)
        x = self.relu(x)
        x = self.fc_d1(x)
        x_de = x.view(-1,1,28,28)
        return x_e, x_de

class DEC:
    """The class for controlling the training process of DEC"""
    def __init__(self,n_clusters,n_features,alpha=1.0):
        self.n_clusters=n_clusters
        self.n_features=n_features
        self.alpha = alpha        
    @staticmethod
    def target_distribution(q):
        weight = (q ** 2  ) / q.sum(0)
        #print('q',q)
        return Variable((weight.t() / weight.sum(1)).t().data, requires_grad=True)
    def logAccuracy(self,pred,label):
        print(' '*8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
          % (acc(label, pred), nmi(label, pred)))
    @staticmethod
    def kld(q,p):
        return torch.sum(p*torch.log(p/q),dim=-1)
    def validateOnCompleteTestData(self,test_loader,model):
        model.eval()
        to_eval = np.array([model(d[0].cuda())[0].data.cpu().numpy() for i,d in enumerate(test_loader)])
        true_labels = np.array([d[1].cpu().numpy() for i,d in enumerate(test_loader)])
        to_eval = np.reshape(to_eval,(to_eval.shape[0]*to_eval.shape[1],to_eval.shape[2]))
        true_labels = np.reshape(true_labels,true_labels.shape[0]*true_labels.shape[1])
        km = KMeans(n_clusters=len(np.unique(true_labels)), n_init=20, n_jobs=4)
        y_pred = km.fit_predict(to_eval)
        print(' '*8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
                      % (acc(true_labels, y_pred), nmi(true_labels, y_pred)))
        currentAcc = acc(true_labels, y_pred)
        return currentAcc
    def pretrain(self,train_loader, test_loader, epochs):
        dec_ae = DEC_AE(self.n_clusters,self.n_features).cuda() #auto encoder
        mseloss = nn.MSELoss()
        optimizer = optim.SGD(dec_ae.parameters(),lr = 1, momentum=0.9)
        best_acc = 0.0
        for epoch in range(epochs):
            dec_ae.train()
            running_loss=0.0
            for i,data in enumerate(train_loader):
                x, label = data
                x, label = Variable(x).cuda(),Variable(label).cuda()
                optimizer.zero_grad()
                x_ae,x_de = dec_ae(x)
                loss = F.mse_loss(x_de,x,reduce=True) 
                loss.backward()
                optimizer.step()
                x_eval = x.data.cpu().numpy()
                label_eval = label.data.cpu().numpy()
                running_loss += loss.data.cpu().numpy()[0]
                if i % 100 == 99:    # print every 100 mini-batches
                    print('[%d, %5d] loss: %.7f' %
                          (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0
            #now we evaluate the accuracy with AE
            dec_ae.eval()
            currentAcc = self.validateOnCompleteTestData(test_loader,dec_ae)
            if currentAcc > best_acc:                
                torch.save(dec_ae,'bestModel'.format(best_acc))
                best_acc = currentAcc
    def clustering(self,mbk,x,model):
        model.eval()
        y_pred_ae,_ = model(x)
        y_pred_ae = y_pred_ae.data.cpu().numpy()
        y_pred = mbk.partial_fit(y_pred_ae) #seems we can only get a centre from batch
        self.cluster_centers = mbk.cluster_centers_ #keep the cluster centers
        model.updateClusterCenter(self.cluster_centers)
    def train(self,train_loader, test_loader, epochs):
        """This method will start training for DEC cluster"""
        ct = time.time()
        model = torch.load("bestModel").cuda()
        model.setPretrain(False)
        optimizer = optim.SGD([\
             {'params': model.parameters()}, \
            ],lr = 0.01, momentum=0.9)
        print('Initializing cluster center with pre-trained weights')
        mbk = MiniBatchKMeans(n_clusters=self.n_clusters, n_init=20, batch_size=batch_size)
        got_cluster_center = False
        for epoch in range(epochs):
            for i,data in enumerate(train_loader):
                x, label = data
                x = Variable(x).cuda()
                optimizer.zero_grad()
                #step 1 - get cluster center from batch
                #here we are using minibatch kmeans to be able to cope with larger dataset.
                if not got_cluster_center:
                    self.clustering(mbk,x,model)
                    if epoch > 1:
                        got_cluster_center = True
                else:
                    model.train()
                    #now we start training with acquired cluster center
                    feature_pred,q = model(x)
                    #get target distribution
                    p = self.target_distribution(q)
                    #print('q',q,'p',p)
                    loss = self.kld(q,p).mean()
                    loss.backward()
                    optimizer.step()
            currentAcc = self.validateOnCompleteTestData(test_loader,model)    
if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    root = './data'
    if not os.path.exists(root):
        os.mkdir(root)
    random.seed(7)

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--pretrain_epochs', default=200, type=int)
    parser.add_argument('--train_epochs', default=200, type=int)
    args = parser.parse_args()

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    # if not exist, download mnist dataset
    train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
    test_set = dset.MNIST(root=root, train=False, transform=trans, download=True)
    batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(
                     dataset=train_set,
                     batch_size=batch_size,
                     shuffle=True)
    test_loader = torch.utils.data.DataLoader(
                    dataset=test_set,
                    batch_size=batch_size,
                    shuffle=False)
    dec = DEC(10,10)
    if args.pretrain_epochs > 0:
        dec.pretrain(train_loader, test_loader, args.pretrain_epochs)
    dec.train(train_loader, test_loader, args.train_epochs)
