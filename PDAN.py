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
from util.loader.augmentation import augmentation
from torch.autograd import Variable
from util.utils import poly_lr_scheduler, adjust_learning_rate, save_models, load_models, CheckpointManager
from util.loader.data_list import ImageList
import util.loader.pre_process as prep
from torch.distributions import Beta
from util.eval import image_classification_test
from model.model import FeatureExtractor,TaskClassifier,DomainClassifier

import os.path as osp


parser = argparse.ArgumentParser(description='PDAN')
parser.add_argument('--dump_logs', type=bool, default=True)
parser.add_argument('--log_dir', type=str, default='./log', help='the path to save logs.')
parser.add_argument('--output_dir', type=str, default='./output', help='the path to save models.')

parser.add_argument('--src_tr', type=str, default="./data/handprint_train.txt", help="source train dataset path list")
parser.add_argument('--src_te', type=str, default="./data/handprint_test.txt", help="source test dataset path list")
parser.add_argument('--tgt_tr', type=str, default="./data/scan_train.txt", help="target train dataset path list")
parser.add_argument('--tgt_te', type=str, default="./data/scan_test.txt", help="target test dataset path list")

parser.add_argument('--batch_size', type=int, default=64, help='batch size.')
parser.add_argument('--num_steps', type=int, default=120000, help='max number of training step.')
parser.add_argument('--learning_rate_feature', type=float, default=1e-4, help='learning rate of feature extractor.')
parser.add_argument('--learning_rate_cls', type=float, default=1e-3, help='learning rate of classifier.')
parser.add_argument('--learning_rate_domain', type=float, default=1e-3, help='learning rate of domain classifier.')
parser.add_argument('--num_classes', type=int, default=241, help='the number of classes of oracle characters.')

parser.add_argument('--domain_adaptation_strength', type=float, default=0.7)
parser.add_argument('--power', type=float, default=1.5)
parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay.')
parser.add_argument('--gpu', type=str, default='0')

args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu



if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

if args.dump_logs == True:
	old_output = sys.stdout
	sys.stdout = open(os.path.join(args.log_dir, 'output.txt'), 'w')

# Setup Augmentations
prep_train = prep.image_train(resize_size=256, crop_size=224, alexnet=False, transform_our=0)
prep_target_train = prep.image_train(resize_size=224)
prep_test = prep.image_test(resize_size=256, crop_size=224, alexnet=False, transform_our=0)
prep_target_test = prep.image_train(resize_size=224)

# ==== DataLoader ====
source_train_set = ImageList(open(args.src_tr).readlines(), transform=prep_train)
source_train_loader = torch_data.DataLoader(source_train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

target_train_set = ImageList(open(args.tgt_tr).readlines(), transform=prep_train)
target_train_loader = torch_data.DataLoader(target_train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

source_test_set = ImageList(open(args.src_te).readlines(), transform=prep_test)
source_test_loader = torch_data.DataLoader(source_test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

target_test_set = ImageList(open(args.tgt_te).readlines(), transform=prep_test)
target_test_loader = torch_data.DataLoader(target_test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

len_train_source = len(source_train_loader)
len_train_target = len(target_train_loader)


feature_extractor = FeatureExtractor().cuda()
classifier = TaskClassifier(args.num_classes).cuda()
domain_dif = DomainClassifier(feature_extractor.output_num(), 1024).cuda()

feature_extractor_opt = optim.SGD(feature_extractor.optim_parameters(args.learning_rate_feature), lr=args.learning_rate_feature, momentum=0.9, weight_decay=args.weight_decay)
classifier_opt = optim.Adam(classifier.parameters(), lr=args.learning_rate_cls, betas=(0.9, 0.99))
domain_dif_opt = optim.Adam(domain_dif.parameters(), lr=args.learning_rate_domain, betas=(0.9, 0.99))


task_loss_fn = nn.CrossEntropyLoss()
domain_loss_fn = nn.KLDivLoss(reduction='batchmean')


manager = CheckpointManager(logs_dir=args.output_dir, feature_extractor = feature_extractor, classifier = classifier, domain_dif = domain_dif)


cudnn.enabled   = True
cudnn.benchmark = True

best_acc = 0.0
beta_dist = Beta(torch.tensor([2.0]), torch.tensor([2.0]))

for i_iter in range(args.num_steps):
    sys.stdout.flush()
    feature_extractor.train()
    classifier.train()
    domain_dif.train()

    adjust_learning_rate(feature_extractor_opt, base_lr=args.learning_rate_feature, i_iter=i_iter, max_iter=args.num_steps, power=args.power)
    adjust_learning_rate(classifier_opt, base_lr=args.learning_rate_cls  , i_iter=i_iter, max_iter=args.num_steps, power=args.power)
    adjust_learning_rate(domain_dif_opt, base_lr=args.learning_rate_domain, i_iter=i_iter, max_iter=args.num_steps, power=args.power)
   
    # ==== sample data ====
    if i_iter % len_train_source == 0:
        iter_source = iter(source_train_loader)
    if i_iter % len_train_target == 0:
        iter_target = iter(target_train_loader)
    source_data, source_label = next(iter_source)
    target_data, target_label = next(iter_target)

    sdatav = Variable(source_data).cuda()
    slabelv = Variable(source_label).cuda()
    tdatav = Variable(target_data).cuda()
    tlabelv = Variable(target_label)

    x = torch.cat([sdatav, tdatav], dim=0)
    lambdas = torch.cat(
    [torch.ones(args.batch_size, 1), # 源域λ=1
    torch.zeros(args.batch_size, 1)],dim=0   # 目标域λ=0
).cuda()
    _,features = feature_extractor(x)
    cls_labels = classifier(features)
    dif_labels = domain_dif(features,lambdas)

  
    task_loss = task_loss_fn(cls_labels[:args.batch_size], slabelv)

  
    src_domain_labels = torch.tensor([[1.0, 0.0]], device='cuda').repeat(args.batch_size, 1)
    target_domain_labels = torch.tensor([[0.0, 1.0]], device='cuda').repeat(args.batch_size, 1)
    domain_labels = torch.cat([src_domain_labels,target_domain_labels], dim=0)
    
    domain_loss= domain_loss_fn(dif_labels,domain_labels)

   
    target_preds = cls_labels[args.batch_size:]
    source_preds = cls_labels[:args.batch_size]

    with torch.no_grad(): 
        target_probs = F.softmax(target_preds, dim=1)
        source_probs = F.softmax(source_preds, dim=1)
        target_confidences,predicted_labels =  target_probs.max(dim=1)
        valid_mask = target_confidences >= 0.95


    total_loss = task_loss + 0.8*domain_loss
   
    N = len(tdatav[valid_mask])

    if N>0 :
        alpha = beta_dist.sample([tdatav[valid_mask].size(0)]).cuda()
        
        alpha1 = alpha.view(-1, 1)
        h_mix = alpha1 * features[:args.batch_size][valid_mask] + (1 - alpha1) * features[args.batch_size:][valid_mask]
        
        lamba = alpha.view(-1,1) 
        target_mix = lamba* F.one_hot(slabelv[valid_mask], num_classes=args.num_classes).float() + (1-lamba) * F.one_hot(predicted_labels[valid_mask], num_classes=args.num_classes).float()
       
        
        mix_cls_labels = classifier(h_mix)
        mix_dif_labels = domain_dif(h_mix,alpha)
        
        
        logprobs = F.log_softmax(mix_cls_labels, dim=1)
        loss_class =  F.kl_div(logprobs, target_mix, reduction='batchmean')
        
        
        mixedDomain = torch.cat([alpha,1-alpha], dim=1)
        loss_adv = domain_loss_fn(mix_dif_labels,mixedDomain)
        
        
        total_loss = total_loss +0.1* loss_adv + 2.0*loss_class
        
        augmented_data = augmentation()(tdatav[valid_mask])
        _,aug_features = feature_extractor(augmented_data)
        aug_cls_labels = classifier(aug_features)
        original_probs = F.softmax(target_preds, dim=1)[valid_mask]
        loss_consistency = F.kl_div(
            F.log_softmax(aug_cls_labels, dim=1),
            original_probs.detach(),
            reduction='batchmean'
        )
    
        total_loss = total_loss + args.domain_adaptation_strength * loss_consistency
     
       
        
        


    
    feature_extractor_opt.zero_grad()
    classifier_opt.zero_grad()
    domain_dif_opt.zero_grad()
    total_loss.backward()
    feature_extractor_opt.step()
    classifier_opt.step()
    domain_dif_opt.step()

 

    if i_iter % 300 == 0:
        feature_extractor.eval()
        classifier.eval()
        print ('evaluating models when %d iter...'%i_iter)
        temp_acc = image_classification_test(source_test_loader, feature_extractor,classifier)
        print('test_source_acc:%.4f' % (temp_acc))

        temp_acc = image_classification_test(target_test_loader, feature_extractor,classifier)
        if temp_acc > best_acc:
            best_acc = temp_acc
            manager.save(epoch=i_iter, fpath=osp.join(args.output_dir, 'checkpoint_max.pth.tar'))
            print('save checkpoint of iteration {:d} when acc1 = {:4.1%}'.format(i_iter, best_acc))
        print('test_target_acc:%.4f' % (temp_acc))
        print('max_acc: {:4.1%}'.format(best_acc))