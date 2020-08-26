import itertools
import numpy as np
import matplotlib.pyplot as plt
import io
import PIL
from torchvision import transforms
from PIL import Image
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
from collections import Counter

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
   # else:
    #    print('Confusion matrix, without normalization')

   # print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    return buf


def create_confusion_matrix(lbllist, predlist, label_list):
    lbllist = lbllist.numpy()
    predlist = predlist.numpy()
    print(type(label_list)) 
    c = Counter(lbllist)
    top20_classes = c.most_common(20)
    top20_classes = [i[0] for i in top20_classes]
    print(len(top20_classes))
    if len(lbllist) == len(predlist):
        print("TRUE")
    del_idx = [] 
    for idx, item in enumerate(lbllist):
        if item not in top20_classes:
            del_idx.append(idx)

    for idx in sorted(del_idx, reverse=True):
        lbllist = np.delete(lbllist, idx)
        predlist = np.delete(predlist, idx)

    conf_mat=confusion_matrix(lbllist, predlist)
    # Per-class accuracy
    class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
    
    plot_buf = plot_confusion_matrix(conf_mat, top20_classes)
    image = PIL.Image.open(plot_buf)
    image = transforms.ToTensor()(image).unsqueeze(0)

    return image

