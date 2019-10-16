import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Plot P-R curve')
parser.add_argument('--model_name', default='PCNN_ATTRA_BAGATT', help='path to model file')
args = parser.parse_args()

plt.clf()

y_true = np.load('result/' + args.model_name + '_true.npy')
y_scores = np.load('result/' + args.model_name + '_scores.npy')
precisions,recalls,threshold = precision_recall_curve(y_true, y_scores)
plt.plot(recalls, precisions, "-b", marker="d", markevery=200, lw=1, label=args.model_name)

plt.ylim([0.4, 1.0])
plt.xlim([0.0, 0.5])
plt.legend(loc="upper right")
plt.title("model performance")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True)
plt.show()
