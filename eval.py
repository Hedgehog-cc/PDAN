# 伪代码示例
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

import sys, torch, argparse, os, random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.utils as vutils
import torchvision.models as models
import torch.utils.data as torch_data
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.autograd import Variable
from util.utils import poly_lr_scheduler, adjust_learning_rate, save_models, load_models, CheckpointManager
from util.loader.data_list import ImageList
import util.loader.pre_process as prep
from torch.distributions import Beta
from util.eval import image_classification_test
from model.model import FeatureExtractor,TaskClassifier,DomainClassifier
from sklearn.manifold import TSNE
import os.path as osp


parser = argparse.ArgumentParser(description='eval')
parser.add_argument('--model_path', type=str, default='./weights/checkpoint_max.pth', help='the path to save models.')
parser.add_argument('--src_te', type=str, default="./data/handprint_test.txt", help="source test dataset path list")
parser.add_argument('--tgt_te', type=str, default="./data/scan_test.txt", help="target test dataset path list")
parser.add_argument('--batch_size', type=int, default=32, help='batch size.')
parser.add_argument('--num_classes', type=int, default=1000, help='the number of classes of oracle characters.')
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

prep_test = prep.image_test(resize_size=256, crop_size=224, alexnet=False, transform_our=0)
prep_target_test = prep.image_train(resize_size=224)

source_test_set = ImageList(open(args.src_te).readlines(), transform=prep_test)
source_test_loader = torch_data.DataLoader(source_test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

target_test_set = ImageList(open(args.tgt_te).readlines(), transform=prep_test)
target_test_loader = torch_data.DataLoader(target_test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

feature_extractor = FeatureExtractor().cuda()
classifier = TaskClassifier(args.num_classes).cuda()

checkpoint = torch.load(args.model_path)
feature_extractor.load_state_dict(checkpoint['feature_extractor']) 
classifier.load_state_dict(checkpoint['classifier']) 
feature_extractor.eval()
classifier.eval()


overall = image_classification_test(target_test_loader, feature_extractor,classifier)
print('target acc:',overall)

