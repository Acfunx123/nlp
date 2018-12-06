import os
import torch
from torch import optim
import torch.nn as nn

from model import Network
from dataset import dataloader
from train import train, validation

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(mode, epoches, learning_rate, train_path, val_path, batch_size, model_path):

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = Network(7).to(device)

    if os.path.isfile(model_path):
        print("loading model")
        model.load_state_dict(torch.load(model_path))


    if mode == "train":
        train_loader = dataloader(train_path, batch_size)
        val_loader = dataloader(val_path, batch_size)

        optimizer = optim.Adam(model.parameters(), lr= learning_rate)
        loss_F = nn.NLLLoss()
        train(epoches, model,train_loader, val_loader, optimizer,loss_F,device)
        torch.save(model.state_dict(), model_path)
    else:
        val_loader = dataloader(val_path, batch_size)
        acc = validation(model, val_loader, device)
        print("\taccuracy: %.2f%%"%(acc))

if __name__ == "__main__":
    print("Train model ...")
    epoches = 10
    learning_rate = 1e-5
    train_path = "./data/train"
    val_path = "./data/val"
    batch_size = 32
    model_path = "./output/params.pth"

    main("train", epoches, learning_rate, train_path, val_path, batch_size, model_path)

    print("Eval model ...")
    main("val", None, None, None, val_path, batch_size, model_path)
