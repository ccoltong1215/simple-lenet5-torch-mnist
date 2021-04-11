
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import Dataset
from model import LeNet5, CustomMLP
import numpy as np
import matplotlib.pyplot as plt
import wandb


def train(model, trn_loader, device, criterion, optimizer,epoch,modelname):
    """ Train function
    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
        acc: accuracy
    """
    model.to(device)
    trn_loss, acc = [], []
    for m in range(epoch):
        train_loss = 0
        trainacc = 0
        for i, (images, labels) in enumerate(trn_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_loss += loss
            temp_acc = torch.mean(torch.eq(torch.argmax(outputs, dim=1), labels).to(dtype=torch.float64))
            trainacc += temp_acc
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if (i) % 1000 == 0:
                print("\r {} Step [{}] Loss: {:.4f} acc: {:.4f}\nlabel".format(modelname,i, loss.item(), temp_acc),labels,"\n output", torch.argmax(outputs, dim=1))
        trainacc = trainacc / trn_loader.__len__()
        train_loss = train_loss / (trn_loader.__len__())     #10은 batchsize, 원래는 argument로 받아와서 사용가능
        print("{} training {} epoch  Loss: {:.4f} acc: {:.4f}".format(modelname,m, train_loss, trainacc))
        trn_loss.append(train_loss.item())
        acc.append(trainacc.item())
        epochlist = range(epoch)

        data = [[x, y] for (x, y) in zip( epochlist,trn_loss)]
        data2 = [[x, y] for (x, y) in zip(epochlist, acc)]
        table = wandb.Table(data=data, columns=[ "epoch","{}Acc".format(modelname)])
        table2 = wandb.Table(data=data2, columns=["epoch", "{}Acc".format(modelname)])
        wandb.log({"{}Acc".format(modelname): wandb.plot.line(table, "epoch", "{}Acc".format(modelname),title="Custom Y vs X Line Plot")})
        wandb.log({"{}loss".format(modelname): wandb.plot.line(table2, "epoch", "{}Acc".format(modelname),title="Custom Y vs X Line Plot")})

    trn_loss = np.array(trn_loss)
    acc=np.array(acc)
    dummy_input = torch.randn(1,1,28,28,device=device)
    input_names = ["input_0"]
    output_names = ["output_0"]
    dummy_output = model(dummy_input)
    torch.onnx.export(model, dummy_input, "{}.onnx".format(modelname), verbose=True, input_names=input_names,output_names=output_names)

    return trn_loss, acc


def test(model, tst_loader, device, criterion):
    """ Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value
        acc: accuracy
    """
    model.to(device)
    tst_loss, acc = [],[]
    test_loss=0
    test_acc=0
    with torch.no_grad(): # 미분 안함,
        for i, (images, labels) in enumerate(tst_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss
            temp_acc = torch.mean(torch.eq(torch.argmax(outputs, dim=1), labels).to(dtype=torch.float64))
            test_acc += temp_acc
            if (i) % 100 == 0:
                print(" Step [{}] Loss: {:.4f} acc: {:.4f}".format(i, loss.item(), temp_acc))
                print("label", labels)
                print("output", torch.argmax(outputs, dim=1))
                tst_loss.append(loss.item())
                acc.append(temp_acc.item())

        test_acc = test_acc/tst_loader.__len__()
        test_loss = test_loss / (tst_loader.__len__())
        print("TEST Step [{}] Loss: {:.4f} acc: {:.4f}".format(tst_loader.__len__(), test_loss, test_acc))
    tst_loss=np.array(tst_loss).astype(float)
    acc=np.array(acc).astype(float)



    return tst_loss, acc




# import some packages you need here

def main():
    """ Main function

        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss

    """
    wandb.init(project="simple_MNIST_report", config={
    })
    roottrain='data/train'
    roottest ='data/test'
    epoch = 100

    # declare pipeline
    trainloader = DataLoader(dataset=Dataset(root=roottrain),  #################################################
                         batch_size=10,
                         shuffle=True)
    testloader = DataLoader(dataset=Dataset(root=roottest),  ################################################
                        batch_size=10,
                        shuffle=False)
    device = torch.device("cuda:0")


    #declare model and opt and loss
    LeNet5_model = LeNet5()
    criterionLeNet = torch.nn.CrossEntropyLoss()
    optimizerLeNet = torch.optim.SGD(LeNet5_model.parameters(), lr=0.001, momentum=0.9)

    CustomMLP_model = CustomMLP()
    criterionCustomMLP = torch.nn.CrossEntropyLoss()
    optimizerCustomMLP = torch.optim.SGD(CustomMLP_model.parameters(), lr=0.001, momentum=0.9)

    wandb.watch(
        LeNet5_model
        )
    wandb.watch(
        CustomMLP_model
        )
####################################################################################
    #start training

    lenet5trnloss, lenet5trnacc = train(model=LeNet5_model, trn_loader=trainloader, device=device, criterion=criterionLeNet,
                                        optimizer=optimizerLeNet,epoch=epoch,modelname="lenet")
    lenet5tstloss, lenet5tstacc = test(model=LeNet5_model, tst_loader=testloader, device=device, criterion=criterionLeNet)

    CustomMLPtrnloss, CustomMLPtrnacc = train(model=CustomMLP_model, trn_loader=trainloader, device=device,
                                              criterion=criterionCustomMLP, optimizer=optimizerCustomMLP,epoch=epoch,modelname="custom")
    CustomMLPtstloss, CustomMLPtstacc = test(model=CustomMLP_model, tst_loader=testloader, device=device, criterion=criterionCustomMLP)


    # fig= plt.figure()

    # lossplt=fig.add_subplot(2, 2, 1)
    # plt.plot(range(int((trainloader.__len__())/100)), lenet5trnloss,color='g'   ,label='LeNet5 train loss'    )
    # plt.plot(range(int((testloader .__len__())/100)), lenet5tstloss,color='r'   ,label='LeNet5 test loss'     )
    # plt.plot(range(int((trainloader.__len__())/100)), CustomMLPtrnloss,color='b',label='Custom MLP train loss')
    # plt.plot(range(int((testloader .__len__())/100)), CustomMLPtstloss,color='m',label='Custom MLP test loss' )
    # plt.legend(loc='upper right',bbox_to_anchor=(1.0, 1.0))
    # plt.xlabel('epoch (x100)')
    # plt.ylabel('loss')
    # plt.title('Loss')
    #
    # accplt=fig.add_subplot(2, 2, 2)
    # plt.plot(range(int((trainloader.__len__())/100)), lenet5trnacc,color='g'   ,label='LeNet5 train accuracy'    )
    # plt.plot(range(int((testloader .__len__())/100)), lenet5tstacc,color='r'   ,label='LeNet5 test accuracy'     )
    # plt.plot(range(int((trainloader.__len__())/100)), CustomMLPtrnacc,color='b',label='Custom MLP train accuracy')
    # plt.plot(range(int((testloader .__len__())/100)), CustomMLPtstacc,color='m',label='Custom MLP test accuracy' )
    # plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
    # plt.xlabel('epoch (x100)')
    # plt.ylabel('acc')
    # plt.title('Accuracy')
    #
    # lenetplt=fig.add_subplot(2, 2, 3)
    # plt.plot(range(int((trainloader.__len__())/100)), lenet5trnloss,color='g',label='train loss'    )
    # plt.plot(range(int((testloader .__len__())/100)), lenet5tstloss,color='r',label='test loss'     )
    # plt.plot(range(int((trainloader.__len__())/100)), lenet5trnacc,color='b' ,label='train accuracy')
    # plt.plot(range(int((testloader .__len__())/100)), lenet5tstacc,color='m' ,label='test accuracy' )
    # plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
    # plt.xlabel('epoch (x100)')
    # plt.title('Loss and Accuracy of LeNet5')
    #
    # customplt=fig.add_subplot(2, 2, 4)
    # plt.plot(range(int((trainloader.__len__())/100)), CustomMLPtrnloss,color='g',label='train loss'    )
    # plt.plot(range(int((testloader .__len__())/100)), CustomMLPtstloss,color='r',label='test loss'     )
    # plt.plot(range(int((trainloader.__len__())/100)), CustomMLPtrnacc,color='b' ,label='train accuracy')
    # plt.plot(range(int((testloader .__len__())/100)), CustomMLPtstacc,color='m' ,label='test accuracy' )
    # plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
    # plt.xlabel('epoch (x100)')
    # plt.title('Loss and Accuracy of Custom MLP')
    # plt.show()


if __name__ == '__main__':
    main()
    ### MNIST WEB app with python - Flask  http://hanwifi.iptime.org:9000/
    ### 19512062 young il han
    ### ccoltong1215@seoultech.ac.kr
    ### https://github.com/ccoltong1215/simple-lenet5-torch-mnist
