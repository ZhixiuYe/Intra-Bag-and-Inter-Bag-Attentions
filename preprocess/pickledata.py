import time
import pickle
import codecs
import numpy as np

class DocumentContainer(object):
    def __init__(self, entity_pair, sentences, label,pos,l_dist,r_dist,entity_pos,sentlens):
        self.entity_pair = entity_pair
        self.sentences = sentences
        self.label = label
        self.pos = pos
        self.l_dist = l_dist
        self.r_dist = r_dist
        self.entity_pos = entity_pos
        self.sentlens = sentlens


def get_ins(snum, index1, index2, pos, sentlen, filter_h=3, max_l=100):
    pad = int(filter_h/2)
    x = [0]*pad
    pf1 = [0]*pad
    pf2 = [0]*pad
    new_sentlen = pad
    if pos[0] == pos[1]:
        if (pos[1] + 1) < len(snum):
            pos[1] = pos[1] + 1
        else:
            pos[0] = pos[0] - 1
    if len(snum) <= max_l:
        new_sentlen += sentlen
        for i, ind in enumerate(snum):
            x.append(ind)
            pf1.append(index1[i] + 1)
            pf2.append(index2[i] + 1)
    else:
        new_sentlen += max_l
        idx = [q for q in range(pos[0], pos[1] + 1)]
        if len(idx) > max_l:
            idx = idx[:max_l]
            for i in idx:
                x.append(snum[i])
                pf1.append(index1[i] + 1)
                pf2.append(index2[i] + 1)
            pos[0] = 0
            pos[1] = len(idx) - 1
        else:
            for i in idx:
                x.append(snum[i])
                pf1.append(index1[i] + 1)
                pf2.append(index2[i] + 1)

            before = pos[0] - 1
            after = pos[1] + 1
            pos[0] = 0
            pos[1] = len(idx) - 1
            numAdded = 0
            while True:
                added = 0
                if before >= 0 and (len(x) + 1) <= (max_l+pad):
                    x.append(snum[before])
                    pf1.append(index1[before] + 1)
                    pf2.append(index2[before] + 1)
                    added = 1
                    numAdded += 1

                if after < len(snum) and (len(x) + 1) <= (max_l+pad):
                    x.append(snum[after])
                    pf1.append(index1[after] + 1)
                    pf2.append(index2[after] + 1)
                    added = 1
                if added == 0:
                    break
                before = before - 1
                after = after + 1

            pos[0] = pos[0] + numAdded
            pos[1] = pos[1] + numAdded
    while len(x) < max_l+2*pad:
        x.append(0)
        pf1.append(0)
        pf2.append(0)

    if pos[0] > max_l-3:
        pos[0] = max_l-3
    if pos[1] > max_l-2:
        pos[1] = max_l-2
    if pos[0] == pos[1]:
        pos = [pos[0] + 1, pos[1] + 2]
    else:
        pos = [pos[0] + 1,pos[1] + 1]
    return [x, pf1, pf2, pos, new_sentlen]

def make_data(data, word2id, filter_h, max_l):
    newData = []
    for j, ins in enumerate(data):
        entities = ins.entity_pair
        entities = [word2id.get(entities[0], 0)+1, word2id.get(entities[1], 0)+1]
        rel = ins.label
        pos = ins.pos
        sentences = ins.sentences
        ldist = ins.l_dist
        rdist = ins.r_dist
        sentlens = ins.sentlens
        newSent = []
        l_dist = []
        r_dist = []
        entitiesPos = ins.entity_pos
        newent = []
        newsentlen = []
        for i, sentence in enumerate(sentences):
            idx,a,b,e,l = get_ins(sentence, ldist[i], rdist[i], entitiesPos[i], sentlens[i], filter_h, max_l)
            newSent.append(idx[:])
            l_dist.append(a[:])
            r_dist.append(b[:])
            newent.append(e[:])
            newsentlen.append(l)
        newIns = DocumentContainer(entity_pair=entities, sentences=newSent, label=rel, pos=pos, l_dist=l_dist, r_dist=r_dist, entity_pos=newent, sentlens=newsentlen)
        newData += [newIns]
    return newData

def make_test_data_12all(data, word2id, filter_h, max_l):
    newData = []
    newData1 = []
    newData2 = []
    for j, ins in enumerate(data):
        entities = ins.entity_pair
        entities = [word2id.get(entities[0], 0)+1, word2id.get(entities[1], 0)+1]
        rel = ins.label
        pos = ins.pos
        sentences = ins.sentences
        ldist = ins.l_dist
        rdist = ins.r_dist
        entitiesPos = ins.entity_pos
        sentlens = ins.sentlens
        newSent = []
        l_dist = []
        r_dist = []
        newent = []
        newsentlen = []
        newSent1 = []
        l_dist1 = []
        r_dist1 = []
        newent1 = []
        newsentlen1 = []
        newSent2 = []
        l_dist2 = []
        r_dist2 = []
        newent2 = []
        newsentlen2 = []

        if len(sentences) > 1:
            index1 = np.random.choice(len(sentences), 1)
            index2 = np.random.choice(len(sentences), 2, replace=False)
            for i, sentence in enumerate(sentences):
                idx,a,b,e,l = get_ins(sentence, ldist[i], rdist[i], entitiesPos[i], sentlens[i], filter_h, max_l)
                newSent.append(idx[:])
                l_dist.append(a[:])
                r_dist.append(b[:])
                newent.append(e[:])
                newsentlen.append(l)
                if i in index1:
                    newSent1.append(idx[:])
                    l_dist1.append(a[:])
                    r_dist1.append(b[:])
                    newent1.append(e[:])
                    newsentlen1.append(l)
                if i in index2:
                    newSent2.append(idx[:])
                    l_dist2.append(a[:])
                    r_dist2.append(b[:])
                    newent2.append(e[:])
                    newsentlen2.append(l)
            newIns = DocumentContainer(entity_pair=entities, sentences=newSent, label=rel, pos=pos, l_dist=l_dist, r_dist=r_dist, entity_pos=newent,sentlens=newsentlen)
            newIns1 = DocumentContainer(entity_pair=entities, sentences=newSent1, label=rel, pos=pos, l_dist=l_dist1, r_dist=r_dist1, entity_pos=newent1,sentlens=newsentlen1)
            newIns2 = DocumentContainer(entity_pair=entities, sentences=newSent2, label=rel, pos=pos, l_dist=l_dist2, r_dist=r_dist2, entity_pos=newent2,sentlens=newsentlen2)
            newData += [newIns]
            newData1 += [newIns1]
            newData2 += [newIns2]

    return newData, newData1, newData2

def make_train_data(data, word2id, filter_h, max_l, num_classes, group_size):
    allData = [[] for _ in range(num_classes)]
    for j, ins in enumerate(data):
        entities = ins.entity_pair
        entities = [word2id.get(entities[0], 0)+1, word2id.get(entities[1], 0)+1]
        rel = ins.label
        pos = ins.pos
        sentences = ins.sentences
        ldist = ins.l_dist
        rdist = ins.r_dist
        sentlens = ins.sentlens
        newSent = []
        l_dist = []
        r_dist = []
        entitiesPos = ins.entity_pos
        newent = []
        newsentlen = []
        for i, sentence in enumerate(sentences):
            idx,a,b,e,l = get_ins(sentence, ldist[i], rdist[i], entitiesPos[i], sentlens[i], filter_h, max_l)
            newSent.append(idx[:])
            l_dist.append(a[:])
            r_dist.append(b[:])
            newent.append(e[:])
            newsentlen.append(l)
        newIns = DocumentContainer(entity_pair=entities, sentences=newSent, label=rel, pos=pos, l_dist=l_dist,
                                       r_dist=r_dist, entity_pos=newent, sentlens=newsentlen)
        allData[newIns.label[0]].append(newIns)

    newData = [[] for _ in range(num_classes)]
    for j, data in enumerate(allData):
        for i, datai in enumerate(data):
            if i % group_size == 0:
                newData[j].append([])
            newData[j][-1].append(datai)
        while newData[j] != [] and len(newData[j][-1]) % group_size != 0:
            newData[j][-1].append(newData[j][-1][-1])
    return newData


def get_word2id(f):
    word2id = {}
    while True:
        line = f.readline().strip()
        if not line:
            break
        words = line.split()
        word = words[0]
        id = int(words[1])
        word2id[word] = id
    return word2id

if __name__ == "__main__":

    print("load test and train raw data...")
    testData = pickle.load(open('test_temp.pkl', 'rb'), encoding='utf-8')
    trainData = pickle.load(open('train_temp.pkl', 'rb'), encoding='utf-8')

    word2id_f = codecs.open('word2id.txt', 'r', 'utf-8')
    word2id = get_word2id(word2id_f)

    sentence_len = 80
    max_filter_len = 3
    num_classes = 53
    group_size = 5
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print('point 0 time: ' + '\t\t' + str(now))

    train_data = make_train_data(trainData, word2id, max_filter_len, sentence_len, num_classes, group_size)
    f = open('train.pkl', 'wb')
    pickle.dump(train_data, f, -1)
    f.close()

    test_data = make_data(testData, word2id, max_filter_len, sentence_len)
    f = open('test.pkl','wb')
    pickle.dump(test_data, f, -1)
    f.close()

    pretrain_data = make_data(trainData, word2id, max_filter_len, sentence_len)
    f = open('pretrain.pkl', 'wb')
    pickle.dump(pretrain_data, f, -1)
    f.close()

    testall_data, test1_data, test2_data = make_test_data_12all(testData, word2id, max_filter_len, sentence_len)
    f = open('testall.pkl', 'wb')
    pickle.dump(testall_data, f, -1)
    f.close()
    f = open('test1.pkl', 'wb')
    pickle.dump(test1_data, f, -1)
    f.close()
    f = open('test2.pkl', 'wb')
    pickle.dump(test2_data, f, -1)
    f.close()

