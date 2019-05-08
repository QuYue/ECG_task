# -*- coding: utf-8 -*-
"""
@Time    : 2019/5/7 14:27
@Author  : QuYue
@File    : ECG_train_LC.py
@Software: PyCharm
Introduction: Main Function for train the model for diagnosis by ECG(Using Link Contraints)
"""
#%% Import Packages
import torch
import arguments
import read_data
import data_process
import model_LC
import score
import matplotlib.pyplot as plt
import drawing

#%% Get Arguments
train_args = {
    'cuda': True,
    'draw': True,
    'datanum': 600,
    'epoch': 300,
    # 'batch_size': 10,
    'train_ratio': 0.8,
    'learn_rate': 0.001,
}
# 0.00008 batch_size=10 可以达到90.8 # 0.001 batch_size=10, model2可以达到91.8%
Args = arguments.get()
arguments.add(Args, train_args) # add the train_args
Args.cuda = Args.cuda and torch.cuda.is_available()
print('Using GPU.') if Args.cuda else print('Using CPU.')
Args.train_data_path = r'../Data/preliminary/TRAIN' # add paths
Args.train_label_path = r'../Data/preliminary/reference.txt'
Args.test_data_path = r'../Data/preliminary/TEST'
torch.manual_seed(Args.seed)
if Args.cuda:
    torch.cuda.manual_seed(Args.seed)
#%% Main Function
if __name__ == '__main__':
    #%%　Read Data
    print('==>Read Data')
    ECG_train_data, ECG_train_label = read_data.read_train(Args)
    # read_data.show(ECG_train_data[0])
    #%% Data Processing
    print('==>Data Processing')
    # data split
    train_x, valid_x, train_y, valid_y = data_process.split(ECG_train_data, ECG_train_label,
                                                            Args, seed=1)
    # change to tensor
    train_x = torch.FloatTensor(train_x)
    valid_x = torch.FloatTensor(valid_x)
    train_y = torch.FloatTensor(train_y).type(torch.LongTensor)
    valid_y = torch.FloatTensor(valid_y).type(torch.LongTensor)
    # load data
    train_loader = data_process.data_loader(train_x, train_y, Args.batch_size)
    valid_loader = data_process.data_loader(valid_x, valid_y, 30)
    # empty cache
    del ECG_train_data, ECG_train_label,  train_x, valid_x, train_y, valid_y
    #%%
    print('==>Training Model')
    diagnosis1 = model_LC.Diagnosis2()
    diagnosis2 = model_LC.Diagnosis2()
    optimizer1 = torch.optim.Adam(diagnosis1.parameters(), lr=Args.learn_rate) # optimizer
    optimizer2 = torch.optim.Adam(diagnosis2.parameters(), lr=Args.learn_rate)  # optimizer
    loss_func = torch.nn.CrossEntropyLoss()  # 损失函数（交叉熵）
    if Args.draw:
        plt.ion()
        figure = plt.figure(1, figsize=(10, 6))
        F1_list1 = []
        acc_list1 = []
        F1_list2 = []
        acc_list2 = []
    if Args.cuda:
        diagnosis1 = diagnosis1.cuda()
        diagnosis2 = diagnosis2.cuda()
    for epoch in range(Args.epoch):
        # Training
        for step, (x, y) in enumerate(train_loader):
            y = torch.squeeze(y) # delete a axis
            if Args.cuda:
                x, y = x.cuda(), y.cuda()
            diagnosis1.train()  # train model
            diagnosis2.train()
            output1, embedding1 = diagnosis1(x)
            output2, embedding2 = diagnosis2(x)
            link_constraints2 = model_LC.LinkConstraints(embedding2, y, weight_decay=Args.LCw)
            loss1 = loss_func(output1, y)  # loss
            loss2 = loss_func(output2, y)
            loss2 += link_constraints2
            optimizer1.zero_grad()  # clear gradients for next train
            loss1.backward()  #  backpropagation, compute gradients
            optimizer1.step()  # apply gradients
            optimizer2.zero_grad()  # clear gradients for next train
            loss2.backward()  #  backpropagation, compute gradients
            optimizer2.step()  # apply gradients
            if step % 1 == 0:
                if Args.cuda:
                    pred1 = torch.max(output1, 1)[1].cuda().data.squeeze()
                    pred2 = torch.max(output2, 1)[1].cuda().data.squeeze()
                else:
                    pred1 = torch.max(output1, 1)[1].data.squeeze()
                    pred2 = torch.max(output2, 1)[1].data.squeeze()
                # evaluate
                accuracy1 = score.accuracy(pred1, y)
                accuracy2 = score.accuracy(pred2, y)
                F1_1 = score.F1(pred1, y)
                F1_2 = score.F1(pred2, y)
                print('Epoch: %s |step: %s | accuracy1: %.2f | F1: %.4f | accuracy2: %.2f | F1: %.4f |' %(epoch, step, accuracy1, F1_1, accuracy2, F1_2))
        #%% Testing
        all_y = []
        all_pred1 = []
        all_pred2 = []
        for step, (x, y) in enumerate(valid_loader):
            y = torch.squeeze(y)  # delete a axis
            if Args.cuda:
                x, y = x.cuda(), y.cuda()
            diagnosis1.eval() # test model
            diagnosis2.eval()
            output1, _ = diagnosis1(x)
            output2, _ = diagnosis2(x)
            if Args.cuda:
                pred1 = torch.max(output1, 1)[1].cuda().data.squeeze()
                pred2 = torch.max(output2, 1)[1].cuda().data.squeeze()
            else:
                pred1 = torch.max(output1, 1)[1].data.squeeze()
                pred2 = torch.max(output2, 1)[1].data.squeeze()
            all_y.append(y)
            all_pred1.append(pred1)
            all_pred2.append(pred2)
        # evaluate
        y = torch.cat(all_y)
        pred1 = torch.cat(all_pred1)
        pred2 = torch.cat(all_pred2)
        accuracy1 = score.accuracy(pred1, y)
        accuracy2 = score.accuracy(pred2, y)
        F1_1 = score.F1(pred1, y)
        F1_2 = score.F1(pred2, y)
        print('Epoch: %s | test accuracy1: %.2f | F1: %.4f | accuracy2: %.2f | F1: %.4f' % (epoch, accuracy1, F1_1, accuracy2, F1_2))
        # drawing
        if Args.draw:
            F1_list1.append(F1_1)
            acc_list1.append(accuracy1)
            F1_list2.append(F1_2)
            acc_list2.append(accuracy2)
            drawing.draw_result([acc_list1, F1_list1, acc_list2, F1_list2], figure, ['Accuracy', 'F1', 'Accuracy_LC', 'F1_LC'], True)
        # save model
        # if F1 == max(F1_list):
        #     print('save model')
        #     save_model = diagnosis.cpu()
        #     torch.save(save_model, '../Model/model1.pkl')
        #     diagnosis = diagnosis.cuda()
        #     del save_model
        # empty memory
        del x, y, all_pred1, all_pred2, all_y, output1, output2
        if Args.cuda: torch.cuda.empty_cache() # empty GPU memory
        # learning rate change
        # if epoch % 10 == 9:
        #     Args.learn_rate *= 0.9
        #     optimizer = torch.optim.Adam(diagnosis.parameters(), lr=Args.learn_rate) # optimizer
        #     print('changeing!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('==>Finish')
    if Args.draw:
        plt.ioff()
        plt.show()








