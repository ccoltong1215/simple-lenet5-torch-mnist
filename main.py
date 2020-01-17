
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import Dataset
from model import LeNet5, CustomMLP
import numpy as np
import matplotlib.pyplot as plt



def train(model, trn_loader, device, criterion, optimizer):
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
    train_loss = 0
    trainacc = 0
    trn_loss, acc = [],[]
    for i, (images, labels) in enumerate(trn_loader):
        # images = images.to(device)
        # labels = labels.to(device)
        images = images.cuda()
        labels = labels.cuda()
        # 확실히 실행되는지 보장이 안됨
        # 안될 시 .to(device) -> .cuda()로 변경



        outputs = model(images)

        loss = criterion(outputs, labels)
        train_loss += loss

        # print("알규맥스",torch.argmax(outputs, dim=1))
        # print("아웃풋",outputs[1])
        # print("아웃풋전부", outputs)
        # print("셰이프",(outputs[1]).shape)
        # print("라벨",labels)
        # print(torch.float(torch.argmax(outputs, dim=1)) == torch.float(labels))
        # correct= (torch.argmax(outputs, dim=1)==labels,dtype=torch.float)
        # correct = torch.mean(torch.eq(torch.argmax(outputs, dim=1), labels).to(dtype=torch.float64))

        temp_acc = torch.mean(torch.eq(torch.argmax(outputs, dim=1), labels).to(dtype=torch.float64))


        trainacc += temp_acc



        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i) % 100 == 0:
            print(" Step [{}] Loss: {:.4f} acc: {:.4f}".format(i, loss.item(), temp_acc))
            print("label", labels)
            print("output", torch.argmax(outputs, dim=1))
            trn_loss.append(loss.item())
            acc.append(temp_acc.item())

    trainacc = trainacc/trn_loader.__len__()
    train_loss = train_loss / (trn_loader.__len__())     #10은 batchsize, 원래는 argument로 받아와서 사용가능
    print("The result Step [{}] Loss: {:.4f} acc: {:.4f}".format(trn_loader.__len__(), train_loss, trainacc))
    trn_loss=np.array(trn_loss)
    acc=np.array(acc)
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
            # 확실히 실행되는지 보장이 안됨
            # 안될 시 .to(device) -> .cuda()로 변경

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
        test_loss = test_loss / (tst_loader.__len__())     #10은 batchsize, 원래는 argument로 받아와서 사용가능
        print("The result Step [{}] Loss: {:.4f} acc: {:.4f}".format(tst_loader.__len__(), test_loss, test_acc))
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

    # roottrain='E:/document/programing/mnisttest/data/test'
    # roottest ='E:/document/programing/mnisttest/data/train'
    roottrain='data/train'
    roottest ='data/test'

    trainloader = DataLoader(dataset=Dataset(root=roottrain),  #################################################
                         batch_size=10,
                         shuffle=True)
    testloader = DataLoader(dataset=Dataset(root=roottest),  ################################################
                        batch_size=10,
                        shuffle=False)
    device = torch.device("cuda:0")
    # model = CustomMLP()

    model = LeNet5()
    criterionLeNet = torch.nn.CrossEntropyLoss()
    optimizerLeNet = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    lenet5trnloss, lenet5trnacc = train(model=model, trn_loader=trainloader, device=device, criterion=criterionLeNet,
                                        optimizer=optimizerLeNet)
    lenet5tstloss, lenet5tstacc = test(model=model, tst_loader=testloader, device=device, criterion=criterionLeNet)


    model = CustomMLP()
    criterionCustomMLP = torch.nn.CrossEntropyLoss()
    optimizerCustomMLP = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    CustomMLPtrnloss, CustomMLPtrnacc = train(model=model, trn_loader=trainloader, device=device,
                                              criterion=criterionCustomMLP, optimizer=optimizerCustomMLP)
    CustomMLPtstloss, CustomMLPtstacc = test(model=model, tst_loader=testloader, device=device, criterion=criterionCustomMLP)


    # device = torch.device("cuda:0")
    # # model = CustomMLP()
    # model = LeNet5()
    #
    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # lenet5trnloss,lenet5trnacc=train(model = model, trn_loader=trainloader, device=device, criterion=criterion, optimizer=optimizer)
    # lenet5tstloss,lenet5tstacc=test(model=model, tst_loader=testloader, device=device, criterion=criterion)
    # model = CustomMLP()
    # CustomMLPtrnloss,CustomMLPtrnacc=train(model = model, trn_loader=trainloader, device=device, criterion=criterion, optimizer=optimizer)
    # CustomMLPtstloss,CustomMLPtstacc=test(model=model, tst_loader=testloader, device=device, criterion=criterion)

    fig= plt.figure()

    lossplt=fig.add_subplot(2, 2, 1)
    plt.plot(range(int((trainloader.__len__())/100)), lenet5trnloss,color='g'   ,label='LeNet5 train loss'    )
    plt.plot(range(int((testloader .__len__())/100)), lenet5tstloss,color='r'   ,label='LeNet5 test loss'     )
    plt.plot(range(int((trainloader.__len__())/100)), CustomMLPtrnloss,color='b',label='Custom MLP train loss')
    plt.plot(range(int((testloader .__len__())/100)), CustomMLPtstloss,color='m',label='Custom MLP test loss' )
    plt.legend(loc='upper right',bbox_to_anchor=(1.0, 1.0))
    plt.xlabel('epoch (x100)')
    plt.ylabel('loss')
    plt.title('Loss')

    accplt=fig.add_subplot(2, 2, 2)
    plt.plot(range(int((trainloader.__len__())/100)), lenet5trnacc,color='g'   ,label='LeNet5 train accuracy'    )
    plt.plot(range(int((testloader .__len__())/100)), lenet5tstacc,color='r'   ,label='LeNet5 test accuracy'     )
    plt.plot(range(int((trainloader.__len__())/100)), CustomMLPtrnacc,color='b',label='Custom MLP train accuracy')
    plt.plot(range(int((testloader .__len__())/100)), CustomMLPtstacc,color='m',label='Custom MLP test accuracy' )
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
    plt.xlabel('epoch (x100)')
    plt.ylabel('acc')
    plt.title('Accuracy')

    lenetplt=fig.add_subplot(2, 2, 3)
    plt.plot(range(int((trainloader.__len__())/100)), lenet5trnloss,color='g',label='train loss'    )
    plt.plot(range(int((testloader .__len__())/100)), lenet5tstloss,color='r',label='test loss'     )
    plt.plot(range(int((trainloader.__len__())/100)), lenet5trnacc,color='b' ,label='train accuracy')
    plt.plot(range(int((testloader .__len__())/100)), lenet5tstacc,color='m' ,label='test accuracy' )
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
    plt.xlabel('epoch (x100)')
    plt.title('Loss and Accuracy of LeNet5')

    customplt=fig.add_subplot(2, 2, 4)
    plt.plot(range(int((trainloader.__len__())/100)), CustomMLPtrnloss,color='g',label='train loss'    )
    plt.plot(range(int((testloader .__len__())/100)), CustomMLPtstloss,color='r',label='test loss'     )
    plt.plot(range(int((trainloader.__len__())/100)), CustomMLPtrnacc,color='b' ,label='train accuracy')
    plt.plot(range(int((testloader .__len__())/100)), CustomMLPtstacc,color='m' ,label='test accuracy' )
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
    plt.xlabel('epoch (x100)')
    plt.title('Loss and Accuracy of Custom MLP')
    plt.show()


if __name__ == '__main__':
    main()
