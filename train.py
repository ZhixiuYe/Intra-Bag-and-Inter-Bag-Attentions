import pickle
import math
import time
import argparse

from sklearn.metrics import auc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.autograd import Variable
from preprocess.data2pkl import DocumentContainer
from model.model_bagatt import Model



def bags_decompose(data_bags):
    bag_sent = [data_bag.sentences for data_bag in data_bags]
    bag_label = [data_bag.label for data_bag in data_bags]
    bag_pos = [data_bag.pos for data_bag in data_bags]
    bag_ldist = [data_bag.l_dist for data_bag in data_bags]
    bag_rdist = [data_bag.r_dist for data_bag in data_bags]
    bag_entity = [data_bag.entity_pair for data_bag in data_bags]
    bag_epos = [data_bag.entity_pos for data_bag in data_bags]
    bag_sentlen = [data_bag.sentlens for data_bag in data_bags]
    return [bag_label, bag_sent, bag_pos, bag_ldist, bag_rdist, bag_entity, bag_epos, bag_sentlen]

def groups_decompose(data_bags):
    bag_label, bag_sent, bag_pos, bag_ldist, bag_rdist, bag_entity, bag_epos, bag_sentlen = [], [], [], [], [], [], [], []
    for bags in data_bags:
        data = group_decompose(bags)
        bag_label.append(data[0])
        bag_sent.append(data[1])
        bag_pos.append(data[2])
        bag_ldist.append(data[3])
        bag_rdist.append(data[4])
        bag_entity.append(data[5])
        bag_epos.append(data[6])
        bag_sentlen.append(data[7])

    return [bag_label, bag_sent, bag_pos, bag_ldist, bag_rdist, bag_entity, bag_epos, bag_sentlen]

def group_decompose(data_bags):

    bag_label = [[data_bag.label for data_bag in data] for data in data_bags]
    bag_sent = [[data_bag.sentences for data_bag in data] for data in data_bags]
    bag_pos = [[data_bag.pos for data_bag in data] for data in data_bags]
    bag_ldist = [[data_bag.l_dist for data_bag in data] for data in data_bags]
    bag_rdist = [[data_bag.r_dist for data_bag in data] for data in data_bags]
    bag_entity = [[data_bag.entity_pair for data_bag in data] for data in data_bags]
    bag_epos = [[data_bag.entity_pos for data_bag in data] for data in data_bags]
    bag_sentlen = [[data_bag.sentlens for data_bag in data] for data in data_bags]

    return [bag_label, bag_sent, bag_pos, bag_ldist, bag_rdist, bag_entity, bag_epos, bag_sentlen]

def curve(y_scores, y_true, num=2000):
    order = np.argsort(y_scores)[::-1]
    guess = 0.
    right = 0.
    target = np.sum(y_true)
    precisions = []
    recalls = []
    for o in order[:num]:
        guess += 1
        if y_true[o] == 1:
            right += 1

        precision = right / guess
        recall = right / target
        precisions.append(precision)
        recalls.append(recall)
    return np.array(recalls), np.array(precisions)

def eval(model, testset, args):

    [test_label, test_sents, _, test_ldist, test_rdist, _, test_epos, test_sentlen] = bags_decompose(testset)

    print('testing...')
    y_true = []
    y_scores = []
    batch_test = 500
    for j in range(int(math.ceil(len(test_sents) / batch_test))):

        total_shape = []
        total_num = 0
        total_word = []
        total_pos1 = []
        total_pos2 = []
        total_pcnnmask = [[], [], []]
        total_entity_pos = []
        total_y = []

        for k in range(j * batch_test, min(len(test_sents), (j + 1) * batch_test)):
            total_shape.append(total_num)
            total_num += len(test_sents[k])

            temp = [0] * model.num_classes
            for r in test_label[k]:
                temp[r] = 1
            total_y.append(temp)

            for l in range(len(test_sents[k])):

                total_word.append(test_sents[k][l])
                allsentlen = len(test_sents[k][l])
                sentlen = test_sentlen[k][l]
                total_pos1.append(test_ldist[k][l])
                total_pos2.append(test_rdist[k][l])
                total_entity_pos.append(test_epos[k][l])
                epos = test_epos[k][l]
                total_pcnnmask[0].append([0] * epos[0] + [-10000] * (allsentlen - epos[0]))
                total_pcnnmask[1].append(
                    [-10000] * epos[0] + [0] * (epos[1] - epos[0]) + [-10000] * (allsentlen - epos[1]))
                total_pcnnmask[2].append(
                    [-10000] * epos[1] + [0] * (sentlen - epos[1]) + [-10000] * (allsentlen - sentlen))

        total_shape.append(total_num)

        total_word = np.array(total_word)
        total_pos1 = np.array(total_pos1)
        total_pos2 = np.array(total_pos2)
        total_pcnnmask = np.array(total_pcnnmask)
        total_y = np.array(total_y)

        total_word = Variable(torch.from_numpy(total_word)).cuda()
        total_pos1 = Variable(torch.from_numpy(total_pos1)).cuda()
        total_pos2 = Variable(torch.from_numpy(total_pos2)).cuda()
        total_pcnnmask = Variable(torch.from_numpy(total_pcnnmask)).cuda().float()

        if args.sent_encoding == "pcnn":
            batch_p = model.decode_PCNN(total_word, total_pos1, total_pos2, total_pcnnmask, total_shape)
        elif args.sent_encoding == "cnn":
            batch_p = model.decode_CNN(total_word, total_pos1, total_pos2, total_entity_pos, total_shape)

        batch_p = batch_p.cpu().data.numpy()

        y_true.append(total_y[:, 1:])
        y_scores.append(batch_p[:, 1:])

    y_true = np.concatenate(y_true).reshape(-1)
    y_scores = np.concatenate(y_scores).reshape(-1)

    return y_true, y_scores

def AUC_and_PN(model, datasets, args):

    model.eval()

    testdata, test1, test2, testall = datasets

    y_true, y_scores = eval(model, testdata, args)

    np.save('result/' + model.name + '_true.npy', y_true)
    np.save('result/' + model.name + '_scores.npy', y_scores)
    recalls, precisions = curve(y_scores, y_true, 3000)
    recalls_01 = recalls[recalls < 0.1]
    precisions_01 = precisions[recalls < 0.1]
    AUC_01 = auc(recalls_01, precisions_01)

    recalls_02 = recalls[recalls < 0.2]
    precisions_02 = precisions[recalls < 0.2]
    AUC_02 = auc(recalls_02, precisions_02)

    recalls_03 = recalls[recalls < 0.3]
    precisions_03 = precisions[recalls < 0.3]
    AUC_03 = auc(recalls_03, precisions_03)

    recalls_04 = recalls[recalls < 0.4]
    precisions_04 = precisions[recalls < 0.4]
    AUC_04 = auc(recalls_04, precisions_04)

    AUC_all = average_precision_score(y_true, y_scores)

    print(AUC_01, AUC_02, AUC_03, AUC_04, AUC_all)

    for q, testdata in enumerate([test1, test2, testall]):

        y_true, y_scores = eval(model, testdata, args)

        order = np.argsort(-y_scores)

        top100 = order[:100]
        correct_num_100 = 0.0
        for i in top100:
            if y_true[i] == 1:
                correct_num_100 += 1.0
        print('P@100: ', correct_num_100 / 100)

        top200 = order[:200]
        correct_num_200 = 0.0
        for i in top200:
            if y_true[i] == 1:
                correct_num_200 += 1.0
        print('P@200: ', correct_num_200 / 200)

        top300 = order[:300]
        correct_num_300 = 0.0
        for i in top300:
            if y_true[i] == 1:
                correct_num_300 += 1.0
        print('P@300: ', correct_num_300 / 300)

        print('mean: ', (correct_num_100 / 100 + correct_num_200 / 200 + correct_num_300 / 300) / 3)


def pretrainModel(model, train_data, datasets, args):

    [train_label, train_sents, _, train_ldist, train_rdist, _, train_epos, train_sentlen] = bags_decompose(train_data)
    lr = args.init_lr
    optimizer = optim.SGD(model.parameters(), lr=lr)

    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print("Training:", str(now))
    temp_order = list(range(len(train_label)))
    num = 0
    batch = args.batch_size_pre
    for epoch in range(args.pretrain_epoch):
        np.random.shuffle(temp_order)

        for i in range(int(math.ceil(len(temp_order) / batch))):
            num += 1
            if num % 100000 == 0:
                lr = lr / 10
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            total_shape = []
            total_num = 0
            total_word = []
            total_pos1 = []
            total_pos2 = []
            total_entity_pos = []
            total_pcnnmask = [[], [], []]
            total_y = []

            temp_input = temp_order[i * batch: min(len(train_sents), (i + 1) * batch)]
            for k in temp_input:
                total_shape.append(total_num)
                total_num += len(train_sents[k])
                total_y.append(train_label[k][0])
                for j in range(len(train_sents[k])):
                    total_word.append(train_sents[k][j])
                    allsentlen = len(train_sents[k][j])
                    sentlen = train_sentlen[k][j]
                    total_pos1.append(train_ldist[k][j])
                    total_pos2.append(train_rdist[k][j])
                    total_entity_pos.append(train_epos[k][j])
                    epos = train_epos[k][j]
                    total_pcnnmask[0].append([0] * epos[0] + [-10000] * (allsentlen - epos[0]))
                    total_pcnnmask[1].append(
                        [-10000] * epos[0] + [0] * (epos[1] - epos[0]) + [-10000] * (allsentlen - epos[1]))
                    total_pcnnmask[2].append(
                        [-10000] * epos[1] + [0] * (sentlen - epos[1]) + [-10000] * (allsentlen - sentlen))

            total_shape.append(total_num)

            total_word = np.array(total_word)
            total_pos1 = np.array(total_pos1)
            total_pos2 = np.array(total_pos2)
            total_pcnnmask = np.array(total_pcnnmask)
            total_y = np.array(total_y)

            total_word = Variable(torch.from_numpy(total_word)).cuda()
            total_pos1 = Variable(torch.from_numpy(total_pos1)).cuda()
            total_pos2 = Variable(torch.from_numpy(total_pos2)).cuda()
            total_pcnnmask = Variable(torch.from_numpy(total_pcnnmask)).cuda().float()
            y_batch = Variable(torch.from_numpy(total_y)).cuda()

            if args.use_RA and args.sent_encoding == "pcnn":
                loss = model.PCNN_ATTRA(total_word, total_pos1, total_pos2,
                                        total_pcnnmask, total_shape, y_batch)
            if args.use_RA and args.sent_encoding == "cnn":
                loss = model.CNN_ATTRA(total_word, total_pos1, total_pos2,
                                       total_pcnnmask, total_shape, y_batch)
            if not args.use_RA and args.sent_encoding == "pcnn":
                loss = model.PCNN_ATTRA(total_word, total_pos1, total_pos2,
                                        total_pcnnmask, total_shape, y_batch)
            if not args.use_RA and args.sent_encoding == "cnn":
                loss = model.CNN_ATTBL(total_word, total_pos1, total_pos2,
                                       total_pcnnmask, total_shape, y_batch)

            model.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 5.0)
            optimizer.step()

            if num % 10000 == 0:
                AUC_and_PN(model, datasets, args)
                model.train()

    torch.save({'model': model.state_dict()}, 'result/' + model.name + '.model')

    return model


def trainModel(model, train_data, datasets, args):
    model.train()
    batch = [args.group_size, args.batch_size_train]
    [train_label, train_sents, _, train_ldist, train_rdist, _, train_epos, train_sentlen] = groups_decompose(train_data)

    lr = args.init_lr / 100.
    optimizer = optim.SGD(model.parameters(), lr=lr)

    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print("Training:", str(now))

    data_length = np.array([len(t) for t in train_label])
    p_rel = data_length / np.sum(data_length)
    for num in range(1, args.step_num):

        rel_order = np.random.choice(len(p_rel), batch[1], p=p_rel)

        np.random.shuffle(rel_order)
        total_word = []
        total_sentlen = []
        total_pos1 = []
        total_pos2 = []

        total_entity_pos = []
        total_pcnnmask = [[],[],[]]
        total_shape = []
        total_num = 0

        for rel in rel_order:
            temp_order = np.random.choice(len(train_label[rel]), 1)

            for k in temp_order:
                for i in range(batch[0]):
                    total_shape.append(total_num)
                    total_num += len(train_sents[rel][k][i])
                    for j in range(len(train_sents[rel][k][i])):
                        total_word.append(train_sents[rel][k][i][j])
                        allsentlen = len(train_sents[rel][k][i][j])
                        sentlen = train_sentlen[rel][k][i][j]
                        total_sentlen.append(sentlen)
                        total_pos1.append(train_ldist[rel][k][i][j])
                        total_pos2.append(train_rdist[rel][k][i][j])
                        total_entity_pos.append(train_epos[rel][k][i][j])
                        epos = train_epos[rel][k][i][j]
                        total_pcnnmask[0].append([0]*epos[0]+[-10000]*(allsentlen-epos[0]))
                        total_pcnnmask[1].append([-10000]*epos[0]+ [0]*(epos[1]-epos[0]) +[-10000]*(allsentlen-epos[1]))
                        total_pcnnmask[2].append([-10000]*epos[1]+[0]*(sentlen-epos[1]) + [-10000]*(allsentlen-sentlen))

        total_shape.append(total_num)
        total_word = np.array(total_word)
        total_pos1 = np.array(total_pos1)
        total_pos2 = np.array(total_pos2)
        total_pcnnmask = np.array(total_pcnnmask)
        total_y = np.array(rel_order)
        total_word = Variable(torch.from_numpy(total_word)).cuda()
        total_pos1 = Variable(torch.from_numpy(total_pos1)).cuda()
        total_pos2 = Variable(torch.from_numpy(total_pos2)).cuda()
        total_pcnnmask = Variable(torch.from_numpy(total_pcnnmask)).cuda().float()
        y_batch = Variable(torch.from_numpy(total_y)).cuda().unsqueeze(1).expand(batch[1],batch[0]).contiguous()

        if args.use_RA and args.sent_encoding == "pcnn":
            loss = model.PCNN_ATTRA_BAGATT(total_word, total_pos1, total_pos2,
                                           total_pcnnmask, total_shape, y_batch, batch)
        if args.use_RA and args.sent_encoding == "cnn":
            loss = model.CNN_ATTRA_BAGATT(total_word, total_pos1, total_pos2,
                                          total_pcnnmask, total_shape, y_batch, batch)
        if not args.use_RA and args.sent_encoding == "pcnn":
            loss = model.PCNN_ATTBL_BAGATT(total_word, total_pos1, total_pos2,
                                           total_pcnnmask, total_shape, y_batch, batch)
        if not args.use_RA and args.sent_encoding == "cnn":
            loss = model.CNN_ATTBL_BAGATT(total_word, total_pos1, total_pos2,
                                          total_pcnnmask, total_shape, y_batch, batch)


        model.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), 5.0)
        optimizer.step()

        if num % 10000 == 0:
            AUC_and_PN(model, datasets, args)
            model.train()

    model.train()

    torch.save({'model': model.state_dict()}, 'result/' + model.name + '.model')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CODE FOR: Distant Supervision Relation Extraction with Intra-Bag and Inter-Bag Attentions')
    parser.add_argument('--pretrain_file', default='preprocess/pretrain.pkl', help='path to pre-training file')
    parser.add_argument('--train_file', default='preprocess/train.pkl', help='path to training file')
    parser.add_argument('--test_file', default='preprocess/test.pkl', help='path to test file')
    parser.add_argument('--test1_file', default='preprocess/test1.pkl', help='path to test-one file')
    parser.add_argument('--test2_file', default='preprocess/test2.pkl', help='path to test-two file')
    parser.add_argument('--testall_file', default='preprocess/testall.pkl', help='path to test-all file')
    parser.add_argument('--emb_file', default='preprocess/word2vec.pkl', help='path to pre-trained embedding file')
    parser.add_argument('--max_distance', type=int, default=30, help='allowed max segment length')
    parser.add_argument('--PF_size', type=int, default=5, help='size of position feature')
    parser.add_argument('--word_embedding_size', type=int, default=50, help='dimension of pre-trained word embedding')
    parser.add_argument('--batch_size_pre', type=int, default=50, help='batch size for pre-training')
    parser.add_argument('--batch_size_train', type=int, default=10, help='batch size for training')
    parser.add_argument('--group_size', type=int, default=5, help='group size for training')
    parser.add_argument('--init_lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--pretrain_epoch', type=int, default=40, help='epoch for pre-training')
    parser.add_argument('--step_num', type=int, default=100000, help='step number for training')
    parser.add_argument('--cnn_filter', type=int, default=230, help='cnn filter number')
    parser.add_argument('--cnn_kernel', type=int, default=3, help='cnn kernel number')
    parser.add_argument('--num_classes', type=int, default=53, help='class number for the dataset')
    parser.add_argument('--sent_encoding', type=str, default="pcnn", help='sentence encoding method, cnn or pcnn')
    parser.add_argument('--use_RA', action='store_true', help='use relation-aware intra-bag attention or not')
    parser.add_argument('--modelname', type=str, default="PCNN_ATTRA", help='model name')
    parser.add_argument('--pretrain', action='store_true', help='pre-training or not')
    parser.add_argument('--modelpath', type=str, default="result/PCNN_ATTRA.model", help='path to model file')

    args = parser.parse_args()
    print(args)
    assert args.sent_encoding in ["pcnn", "cnn"]

    pretrain_data = pickle.load(open(args.pretrain_file, 'rb'), encoding='utf-8')
    train_data = pickle.load(open(args.train_file, 'rb'), encoding='utf-8')
    testdata = pickle.load(open(args.test_file, 'rb'), encoding='utf-8')
    test1 = pickle.load(open(args.test1_file, 'rb'), encoding='utf-8')
    test2 = pickle.load(open(args.test2_file, 'rb'), encoding='utf-8')
    testall = pickle.load(open(args.testall_file, 'rb'), encoding='utf-8')
    Wv = pickle.load(open(args.emb_file, 'rb'), encoding='utf-8')

    datasets = [testdata, test1, test2, testall]

    max_distance = args.max_distance
    PF1 = np.asarray(np.random.uniform(low=-1, high=1, size=[max_distance*2+1, args.PF_size]), dtype='float32')
    padPF1 = np.zeros((1, args.PF_size))
    PF1 = np.vstack((padPF1, PF1))
    PF2 = np.asarray(np.random.uniform(low=-1, high=1, size=[max_distance*2+1, args.PF_size]), dtype='float32')
    padPF2 = np.zeros((1, args.PF_size))
    PF2 = np.vstack((padPF2, PF2))

    print('modelname: ', args.modelname)

    model = Model(word_length=len(Wv), feature_length=len(PF1), cnn_layers=args.cnn_filter,
                  kernel_size=(args.cnn_kernel, args.word_embedding_size+2*args.PF_size),
                  Wv=Wv, pf1=PF1, pf2=PF2, num_classes=args.num_classes, name=args.modelname)
    model.cuda()

    if args.pretrain:
        model = pretrainModel(model=model, train_data=pretrain_data, datasets=datasets, args=args)
    else:
        config = torch.load(args.modelpath)
        model.load_state_dict(config['model'], strict=False)

    model.name = model.name + '_BAGATT'
    trainModel(model=model, train_data=train_data, datasets=datasets, args=args)
