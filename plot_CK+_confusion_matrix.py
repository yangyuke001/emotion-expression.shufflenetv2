"""
plot confusion_matrix of fold Test set of CK+
"""
import transforms as transforms
import argparse
import itertools
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from CK import CK
#from models import *
from ShuffleNetV2 import ShuffleNetV2
import torch
from torch.autograd import Variable
import torchvision

test_data = './test_img/eval_fer'

parser = argparse.ArgumentParser(description='PyTorch CK+ CNN Training')
parser.add_argument('--dataset', type=str, default='CK+', help='emotion dataset')
parser.add_argument('--model', type=str, default='ShuffleNetV2', help='CNN architecture')
opt = parser.parse_args()

cut_size = 44



transform_test=transforms.Compose([
    transforms.Resize(48),
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")


    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Contempt']

# Model
ShuffleNetV2 = ShuffleNetV2()
checkpoint = torch.load('weights/ShuffleNetV2.pth')
ShuffleNetV2.load_state_dict(checkpoint['net'])
#ShuffleNetV2.eval()
#net = ShuffleNetV2()

correct = 0
total = 0
all_target = []

for i in range(10):
    #print("%d fold" % (i+1))
    path = os.path.join(opt.dataset + '_' + opt.model,  '%d' %(i+1))
    
    ShuffleNetV2.cuda()
    ShuffleNetV2.eval()
    #testset = CK(split = 'Testing', fold = i+1, transform=transform_test)
    testset=torchvision.datasets.ImageFolder(test_data,transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=5, 
        shuffle=False, num_workers=1)

    for batch_idx, (inputs, targets) in enumerate(testloader):
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = ShuffleNetV2(inputs)
        #print('outputS:',outputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
        _, predicted = torch.max(outputs_avg.data, 1)

        #print('output:',outputs_avg.data)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        

        if batch_idx == 0 and i == 0:
            all_predicted = predicted
            all_targets = targets
        else:
            all_predicted = torch.cat((all_predicted, predicted), 0)
            all_targets = torch.cat((all_targets, targets), 0)

        print('correct nums:',correct)

        acc = 100. * correct / total
        print("accuracy: %0.4f" % acc)

# Compute confusion matrix
matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure(figsize=(10, 8))
plot_confusion_matrix(matrix, classes=class_names, normalize=False,
                      title= 'Confusion Matrix (Accuracy: %0.4f%%)' %acc)
plt.show()
plt.savefig( 'Confusion Matrix.png')
plt.close()