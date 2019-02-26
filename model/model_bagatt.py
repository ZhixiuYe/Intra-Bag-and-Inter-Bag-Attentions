import torch
import torch.nn as nn
from .embedding import getEmbeddings
from .pcnn import CNNwithPool
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class Model(nn.Module):
    ### TODO: improve the BAGATT module
    def __init__(self, word_length, feature_length, cnn_layers, Wv, pf1, pf2, kernel_size,
                 word_size=50, feature_size=5, dropout=0.5, num_classes=53, name='model'):
        super(Model, self).__init__()

        self.word_length = word_length
        self.feature_length = feature_length
        self.cnn_layers = cnn_layers
        self.kernel_size = kernel_size
        self.word_size = word_size
        self.feature_size = feature_size
        self.num_classes = num_classes
        self.name = name

        self.embeddings = getEmbeddings(self.word_size, self.word_length, self.feature_size,
                                        self.feature_length, Wv, pf1, pf2)
        self.PCNN = CNNwithPool(self.cnn_layers, self.kernel_size)
        self.CNN = nn.Conv2d(1, cnn_layers, kernel_size)

        self.dropout = nn.Dropout(dropout)

        self.R_PCNN = nn.Linear(cnn_layers*3, num_classes)
        self.init_linear(self.R_PCNN)

        self.R_CNN = nn.Linear(cnn_layers, num_classes)
        self.init_linear(self.R_CNN)

        self.diag = Variable(torch.ones(self.num_classes).diag().unsqueeze(0)).cuda()

    def init_linear(self, input_linear):
        """
        Initialize linear transformation
        """
        bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
        nn.init.uniform(input_linear.weight, -bias, bias)
        if input_linear.bias is not None:
            input_linear.bias.data.zero_()

    def CNN_ATTBL(self, x, ldist, rdist, pool, total_shape, y_batch):

        embeddings = self.embeddings(x, ldist, rdist)
        cnn, _ = torch.max(self.CNN(embeddings), 2)
        cnn = cnn.squeeze(2)
        batch_sent_emb = self.dropout(cnn)
        batch_sent_emb = nn.functional.tanh(batch_sent_emb)
        batch_e = self.R_CNN(batch_sent_emb)
        batch_p = []
        for i in range(len(total_shape) - 1):
            beg, end = total_shape[i], total_shape[i + 1]
            weight_e = batch_e[beg: end, y_batch[i].cpu().data[0]]
            sent_emb = batch_sent_emb[beg: end]
            alpha = nn.functional.softmax(weight_e, 0)
            s = torch.matmul(alpha, sent_emb)
            o = self.R_CNN(self.dropout(s))
            batch_p.append(o)
        batch_p = torch.stack(batch_p)
        loss = nn.functional.cross_entropy(batch_p, y_batch)
        return loss

    def CNN_ATTRA(self, x, ldist, rdist, pool, total_shape, y_batch):

        embeddings = self.embeddings(x, ldist, rdist)
        cnn, _ = torch.max(self.CNN(embeddings), 2)
        cnn = cnn.squeeze(2)
        batch_sent_emb = self.dropout(cnn)
        batch_sent_emb = nn.functional.tanh(batch_sent_emb)
        batch_e = self.R_CNN(batch_sent_emb)
        batch_score = []
        for i in range(len(total_shape) - 1):
            beg, end = total_shape[i], total_shape[i + 1]
            weight_e = batch_e[beg: end]
            sent_emb = batch_sent_emb[beg: end]
            alpha = nn.functional.softmax(weight_e.transpose(0, 1), 1)
            bag_rep = torch.matmul(alpha, sent_emb)
            o = self.R_CNN(self.dropout(bag_rep))
            batch_score.append(o.diag())
        batch_score = torch.stack(batch_score)
        loss = nn.functional.cross_entropy(batch_score, y_batch)
        return loss

    def CNN_ATTBL_BAGATT(self, x, ldist, rdist, pool, total_shape, y_batch, batch):

        embeddings = self.embeddings(x, ldist, rdist)
        cnn, _ = torch.max(self.CNN(embeddings), 2)
        cnn = cnn.squeeze(2)
        batch_sent_emb = self.dropout(cnn)
        batch_sent_emb = nn.functional.tanh(batch_sent_emb)
        batch_e = self.R_CNN(batch_sent_emb)
        bag_reps = []
        for i in range(len(total_shape) - 1):
            beg, end = total_shape[i], total_shape[i + 1]
            weight_e = batch_e[beg: end, y_batch.view(-1).data[i]]
            sent_emb = batch_sent_emb[beg: end]
            alpha = nn.functional.softmax(weight_e, 0)
            bag_rep = torch.matmul(alpha, sent_emb)
            bag_reps.append(bag_rep)

        bag_reps = torch.stack(bag_reps).view(batch[1],batch[0],self.cnn_layers)
        losses = []
        for i, (bag_rep, y) in enumerate(zip(torch.unbind(bag_reps, 0), torch.unbind(y_batch, 0))):
            if y.data[0] == 0:
                score = self.R_CNN(self.dropout(bag_rep))
                loss = nn.functional.cross_entropy(score, y)
                losses.append(loss)
            else:
                bag_rep = bag_rep / torch.norm(bag_rep, 2, 1, keepdim=True)
                crossatt = torch.matmul(bag_rep, bag_rep.transpose(0,1))
                crossatt = torch.sum(crossatt, 1)
                crossatt = F.softmax(crossatt, 0)
                weighted_bags_rep = torch.matmul(crossatt, bag_rep)
                score = self.R_CNN(self.dropout(weighted_bags_rep)).unsqueeze(0)
                loss = nn.functional.cross_entropy(score, y[0])
                losses.append(loss)
        losses = torch.stack(losses).mean()

        return losses

    def CNN_ATTRA_BAGATT(self, x, ldist, rdist, pool, total_shape, y_batch, batch):

        embeddings = self.embeddings(x, ldist, rdist)
        cnn, _ = torch.max(self.CNN(embeddings), 2)
        cnn = cnn.squeeze(2)
        batch_sent_emb = self.dropout(cnn)
        batch_sent_emb = nn.functional.tanh(batch_sent_emb)
        batch_e = self.R_CNN(batch_sent_emb)
        bag_reps = []
        for i in range(len(total_shape) - 1):
            beg, end = total_shape[i], total_shape[i + 1]
            weight_e = batch_e[beg: end]
            sent_emb = batch_sent_emb[beg: end]
            alpha = nn.functional.softmax(weight_e.transpose(0, 1), 1)
            bag_rep = torch.matmul(alpha, sent_emb)
            bag_reps.append(bag_rep)

        bag_reps = torch.stack(bag_reps).view(batch[1],batch[0],self.num_classes,self.cnn_layers)
        losses = []
        for i, (bag_rep, y) in enumerate(zip(torch.unbind(bag_reps, 0), torch.unbind(y_batch, 0))):
            if y.data[0] == 0:
                score = torch.sum(self.R_CNN(self.dropout(bag_rep)) * self.diag, 2)
                loss = nn.functional.cross_entropy(score, y)
                losses.append(loss)
            else:
                bag_rep = bag_rep.transpose(0,1)
                bag_rep = bag_rep / torch.norm(bag_rep, 2, 2, keepdim=True)
                crossatt = torch.matmul(bag_rep, bag_rep.transpose(1,2))
                crossatt = torch.sum(crossatt, 2)
                crossatt = F.softmax(crossatt, 1)
                weighted_bags_rep = torch.matmul(crossatt.unsqueeze(1), bag_rep).squeeze(1)
                score = self.R_CNN(self.dropout(weighted_bags_rep)).diag().unsqueeze(0)
                loss = nn.functional.cross_entropy(score, y[0])
                losses.append(loss)
        losses = torch.stack(losses).mean()

        return losses

    def decode_CNN(self, x, ldist, rdist, pool, total_shape):

        embeddings = self.embeddings(x, ldist, rdist)
        cnn, _ = torch.max(self.CNN(embeddings), 2)
        cnn = cnn.squeeze(2)
        batch_sent_emb = self.dropout(cnn)
        batch_sent_emb = nn.functional.tanh(batch_sent_emb)
        batch_e = self.R_CNN(batch_sent_emb)
        batch_score = []
        for i in range(len(total_shape) - 1):
            beg, end = total_shape[i], total_shape[i + 1]
            weight_e = batch_e[beg: end]
            sent_emb = batch_sent_emb[beg: end]
            alpha = nn.functional.softmax(weight_e.transpose(0, 1), 1)
            bag_rep = torch.matmul(alpha, sent_emb)
            o = self.R_CNN(self.dropout(bag_rep))
            o = F.softmax(o, 1)
            batch_score.append(o.diag())
        batch_p = torch.stack(batch_score)
        return batch_p

    def PCNN_ATTBL(self, x, ldist, rdist, pool, total_shape, y_batch):

        embeddings = self.embeddings(x, ldist, rdist)
        cnn = self.PCNN(embeddings, pool).view((embeddings.size(0), -1))
        batch_sent_emb = self.dropout(cnn)
        batch_sent_emb = nn.functional.tanh(batch_sent_emb)
        batch_e = self.R_PCNN(batch_sent_emb)
        batch_p = []
        for i in range(len(total_shape) - 1):
            beg, end = total_shape[i], total_shape[i + 1]
            weight_e = batch_e[beg: end, y_batch[i].cpu().data[0]]
            sent_emb = batch_sent_emb[beg: end]
            alpha = nn.functional.softmax(weight_e, 0)
            s = torch.matmul(alpha, sent_emb)
            o = self.R_PCNN(self.dropout(s))
            batch_p.append(o)
        batch_p = torch.stack(batch_p)
        loss = nn.functional.cross_entropy(batch_p, y_batch)
        return loss

    def PCNN_ATTRA(self, x, ldist, rdist, pcnnmask, total_shape, y_batch):

        embeddings = self.embeddings(x, ldist, rdist)
        batch_sent_emb = self.PCNN(embeddings, pcnnmask).view((embeddings.size(0), -1))
        batch_sent_emb = self.dropout(batch_sent_emb)
        batch_sent_emb = nn.functional.tanh(batch_sent_emb)
        batch_e = self.R_PCNN(batch_sent_emb)
        batch_score = []
        for i in range(len(total_shape) - 1):
            beg, end = total_shape[i], total_shape[i + 1]
            weight_e = batch_e[beg: end]
            sent_emb = batch_sent_emb[beg: end]
            alpha = nn.functional.softmax(weight_e.transpose(0, 1), 1)
            bag_rep = torch.matmul(alpha, sent_emb)
            o = self.R_PCNN(self.dropout(bag_rep))
            batch_score.append(o.diag())
        batch_score = torch.stack(batch_score)
        loss = nn.functional.cross_entropy(batch_score, y_batch)
        return loss

    def PCNN_ATTBL_BAGATT(self, x, ldist, rdist, pool, total_shape, y_batch, batch):

        embeddings = self.embeddings(x, ldist, rdist)
        batch_sent_emb = self.PCNN(embeddings, pool).view((embeddings.size(0), -1))
        batch_sent_emb = self.dropout(batch_sent_emb)
        batch_sent_emb = nn.functional.tanh(batch_sent_emb)
        batch_e = self.R_PCNN(batch_sent_emb)
        bag_reps = []
        for i in range(len(total_shape) - 1):
            beg, end = total_shape[i], total_shape[i + 1]
            weight_e = batch_e[beg: end, y_batch.view(-1).data[i]]
            sent_emb = batch_sent_emb[beg: end]
            alpha = nn.functional.softmax(weight_e, 0)
            bag_rep = torch.matmul(alpha, sent_emb)
            bag_reps.append(bag_rep)

        bag_reps = torch.stack(bag_reps).view(batch[1],batch[0],self.cnn_layers*3)
        losses = []
        for i, (bag_rep, y) in enumerate(zip(torch.unbind(bag_reps, 0), torch.unbind(y_batch, 0))):
            if y.data[0] == 0:
                score = self.R_PCNN(self.dropout(bag_rep))
                loss = nn.functional.cross_entropy(score, y)
                losses.append(loss)
            else:
                bag_rep = bag_rep / torch.norm(bag_rep, 2, 1, keepdim=True)
                crossatt = torch.matmul(bag_rep, bag_rep.transpose(0,1))
                crossatt = torch.sum(crossatt, 1)
                crossatt = F.softmax(crossatt, 0)
                weighted_bags_rep = torch.matmul(crossatt, bag_rep)
                score = self.R_PCNN(self.dropout(weighted_bags_rep)).unsqueeze(0)
                loss = nn.functional.cross_entropy(score, y[0])
                losses.append(loss)
        losses = torch.stack(losses).mean()
        return losses

    def PCNN_ATTRA_BAGATT(self, x, ldist, rdist, pcnnmask, total_shape, y_batch, batch):

        embeddings = self.embeddings(x, ldist, rdist)
        batch_sent_emb = self.PCNN(embeddings, pcnnmask).view((embeddings.size(0), -1))
        batch_sent_emb = self.dropout(batch_sent_emb)
        batch_sent_emb = nn.functional.tanh(batch_sent_emb)
        batch_e = self.R_PCNN(batch_sent_emb)
        bag_reps = []
        for i in range(len(total_shape) - 1):
            beg, end = total_shape[i], total_shape[i + 1]
            weight_e = batch_e[beg: end]
            sent_emb = batch_sent_emb[beg: end]
            alpha = nn.functional.softmax(weight_e.transpose(0, 1), 1)
            bag_rep = torch.matmul(alpha, sent_emb)
            bag_reps.append(bag_rep)

        bag_reps = torch.stack(bag_reps).view(batch[1],batch[0],self.num_classes,self.cnn_layers*3)
        losses = []
        for i, (bag_rep, y) in enumerate(zip(torch.unbind(bag_reps, 0), torch.unbind(y_batch, 0))):
            if y.data[0] == 0:
                score = torch.sum(self.R_PCNN(self.dropout(bag_rep)) * self.diag, 2)
                loss = nn.functional.cross_entropy(score, y)
                losses.append(loss)
            else:
                bag_rep = bag_rep.transpose(0,1)
                bag_rep = bag_rep / torch.norm(bag_rep, 2, 2, keepdim=True)
                crossatt = torch.matmul(bag_rep, bag_rep.transpose(1,2))
                crossatt = torch.sum(crossatt, 2)
                crossatt = F.softmax(crossatt, 1)
                weighted_bags_rep = torch.matmul(crossatt.unsqueeze(1), bag_rep).squeeze(1)
                score = self.R_PCNN(self.dropout(weighted_bags_rep)).diag().unsqueeze(0)
                loss = nn.functional.cross_entropy(score, y[0])
                losses.append(loss)
        losses = torch.stack(losses).mean()

        return losses

    def decode_PCNN(self, x, ldist, rdist, pcnnmask, total_shape):

        embeddings = self.embeddings(x, ldist, rdist)
        cnn = self.PCNN(embeddings, pcnnmask).view((embeddings.size(0), -1))
        batch_sent_emb = self.dropout(cnn)
        batch_sent_emb = nn.functional.tanh(batch_sent_emb)
        batch_e = self.R_PCNN(batch_sent_emb)
        batch_score = []
        for i in range(len(total_shape) - 1):
            beg, end = total_shape[i], total_shape[i + 1]
            weight_e = batch_e[beg: end]
            sent_emb = batch_sent_emb[beg: end]
            alpha = nn.functional.softmax(weight_e.transpose(0, 1), 1)
            bag_rep = torch.matmul(alpha, sent_emb)
            o = self.R_PCNN(self.dropout(bag_rep))
            o = F.softmax(o, 1)
            batch_score.append(o.diag())
        batch_p = torch.stack(batch_score)
        return batch_p

    def forward(self, x, ldist, rdist, pool, total_shape, y_batch):
        pass
