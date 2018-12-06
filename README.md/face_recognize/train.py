import torch
from torch import optim
from torch.nn.functional import nll_loss
import torch.nn.functional as F
from torch.autograd import Variable
import time


def train(epoches, model, train_loader, val_loader, optim, loss_F, device):
    for epoch in range(epoches):
        start = time.time()
        total_loss = 0
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images , labels = Variable(images), Variable(labels)

            images = images.to(device)
            labels = labels.long().to(device)
            preds = model(images)
            loss = loss_F(preds, labels)
            loss.backward()
            optim.step()
            optim.zero_grad()
            total_loss += loss
        
        accuracy = validation(model, val_loader, device)
        #print("=====================================================================")
        print('epoch %d, time: %.2f' % (epoch, time.time()-start))
        print('\ttrain_loss: %.2f' % (total_loss))
        print("\taccuracy on val_dataset: %.2f%%" % (accuracy))
        #print("=====================================================================")


def validation(model, loader, device):
    model.eval()
    correct = 0
    for images, labels in loader:
        
        with torch.no_grad():
            images, labels = Variable(images), Variable(labels)

            images, labels = images.to(device), labels.to(device)

            output = model(images)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).cpu().sum()
  
    return 100.0 * correct / len(loader.dataset)
