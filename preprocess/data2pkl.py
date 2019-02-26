import numpy as np
import pickle
import codecs


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


def readData(filename, mode):
    print(filename)
    f = codecs.open(filename, 'r')
    data = []
    while 1:
        line = f.readline()
        if not line:
            break
        if mode == 1:
            num = line.split("\t")[3].strip().split(",")
        else:
            num = line.split("\t")[2].strip().split(",")
        ldist = []
        rdist = []
        sentences = []
        sentlens = []
        entitiesPos = []
        pos = []
        rels = []
        for i in range(0, len(num)):
            sent = f.readline().strip().split(',')
            entities = sent[:2]
            epos = list(map(int,sent[2:4]))
            epos = sorted(epos)
            rels.append(int(sent[4]))
            sent = f.readline().strip().split(",")
            sentences.append([(x+1) for x in list(map(int, sent))])
            sentlens.append(len(sent))
            sent = f.readline().strip().split(",")
            ldist.append(list(map(int, sent)))
            sent = f.readline().strip().split(",")
            rdist.append(list(map(int, sent)))
            entitiesPos.append(epos)
            pos.append([0]*len(sentences[-1]))
        rels = list(set(rels))
        ins = DocumentContainer(entity_pair=entities, sentences=sentences, label=rels, pos=pos, l_dist=ldist, r_dist=rdist, entity_pos=entitiesPos, sentlens=sentlens)
        data += [ins]
    f.close()
    return data

def wv2pickle(filename='wv.txt', dim=50, outfile='Wv.p'):
    f = codecs.open(filename, 'r')
    allLines = f.readlines()
    f.close()
    Wv = np.zeros((len(allLines)+1, dim))
    i = 1
    for line in allLines:
        line = line.split("\t")[1].strip()[:-1]
        Wv[i, :] = list(map(float, line.split(',')))
        i += 1
    rng = np.random.RandomState(3435)
    Wv[1, :] = rng.uniform(low=-0.5, high=0.5, size=(1, dim))
    f = codecs.open(outfile, 'wb')
    pickle.dump(Wv, f, -1)
    f.close()

def data2pickle(input, output, mode):
    data = readData(input, mode)
    f = open(output, 'wb')
    pickle.dump(data, f, -1)
    f.close()


if __name__ == "__main__":
    wv2pickle('word2vec.txt', 50, 'word2vec.pkl')
    data2pickle('bags_train.txt','train_temp.pkl',1)
    data2pickle('bags_test.txt','test_temp.pkl',0)

