import copy
import argparse
from datetime import datetime
import time
import os
from itertools import chain
import pandas as pd
import numpy as np

import torch
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

# from tensorboardX import SummaryWriter

# from model.ResNet import ResNet18, ResNet34, ResNet50, ResNet101 
# from model.segment_decoder import Decoder
import base_model

from dataloaders import custom_transforms as trforms
from dataloaders.vivqa_dataset import ViVQADataset

import utils


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default='0')

    # Model setting
    parser.add_argument('--backbone', type=str, default='resnet34')
    parser.add_argument('--bert_type', type=str, default='biobert')
    parser.add_argument('--bert_pretrained', type=str, default='dmis-lab/biobert-large-cased-v1.1-squad')
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--data_dir', type=str, default='/content/dataset')
    parser.add_argument('--output', type=str, default='/content')
    
    parser.add_argument('--v_dim', type=int, default=1024)
    # Joint representation C dimension
    parser.add_argument('--q_dim', type=int, default=1024,
                        help='dim of joint semantic features')
    
    # Choices of attention models
    parser.add_argument('--model', type=str, default='CMSA', choices=['BAN', 'SAN', 'CMSA'],
                        help='the model we use')
    
    # BAN - Bilinear Attention Networks
    parser.add_argument('--gamma', type=int, default=2,
                        help='glimpse in Bilinear Attention Networks')
    # Choices of RNN models
    parser.add_argument('--rnn', type=str, default='LSTM', choices=['LSTM', 'GRU'],
                        help='the RNN we use')
    parser.add_argument('--op', type=str, default='c',
                        help='concatenated 600-D word embedding')
    parser.add_argument('--question_len', default=20, type=int, metavar='N',
                        help='maximum length of input question')
    parser.add_argument('--tfidf', type=bool, default=None,
                        help='tfitrain_log_df word embedding?')
    # Activation function + dropout for classification module
    parser.add_argument('--activation', type=str, default='relu', choices=['relu'],
                        help='the activation to use for final classifier')
    parser.add_argument('--dropout', default=0.2, type=float, metavar='dropout',
                        help='dropout of rate of final classifier')
    parser.add_argument('--clip_norm', default=.25, type=float, metavar='NORM',
                        help='clip threshold of gradients')

    # Training setting
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--nepochs', type=int, default=100)
    parser.add_argument('--resume_epoch', type=int, default=100)
    parser.add_argument('--train_fold', type=str, default='/content')
    parser.add_argument('--run_id', type=int, default=-1)
    parser.add_argument('--T', type=int, default=2)

    # Optimizer setting
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--update_lr_every', type=int, default=20)

    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--log_every', type=int, default=25)
    parser.add_argument('--emb_init', type=str, default='biowordvec', choices=['glove', 'biowordvec', 'biosentvec'])
    parser.add_argument('--self_att', action='store_true', default=False, help='Use Self Attention?')
    parser.add_argument('--use_spatial', action='store_true', default=False, help='Use spatial feature?')
    parser.add_argument('--use_cma', action='store_true', default=False, help='Use CMA?')
    parser.add_argument('--result_fold', type=str, default='results')

    return parser.parse_args()


def adjust_learning_rate(optimizer, lr_):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    args.device = device

    torch.cuda.set_device(device)

    data_transforms = {
        'train': transforms.Compose([
            trforms.FixedResize(size=(args.input_size, args.input_size)),
            trforms.RandomHorizontalFlip(),
            trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            trforms.ToTensor()
        ]),
        'test': transforms.Compose([
            trforms.FixedResize(size=(args.input_size, args.input_size)),
            trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            trforms.ToTensor()
        ]),
    }
    
    # Load train and validation dataset
    datasets = { mode: ViVQADataset(args, pretrained=args.bert_pretrained, question_len=args.question_len,
                                     mode=mode, transform=data_transforms[mode])
                    for mode in ['train', 'test'] }
    
    dataloaders = { mode: DataLoader(datasets[mode], batch_size=args.batch_size,
                                     shuffle=True, num_workers=2)
                    for mode in ['train', 'test'] }
    
    data_size =  { mode: len(datasets[mode]) for mode in ['train', 'test'] }
    
    args.num_classes = datasets['train'].num_classes
    
    print('Number of classes: ', args.num_classes)
    print("Dataset size: ", data_size)
    print("Dataloader size: ", len(dataloaders['train']), len(dataloaders['test']))
    
    # Create VQA model
    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(args)
    model.to(device)
    
    # Initialize optimizer algorithm
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.update_lr_every)
    
    # Initialize loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    
    EPOCHS = args.nepochs
    best_val_acc = 0.
    best_model = copy.deepcopy(model.state_dict())
    # best_bert_model = copy.deepcopy(model.q_embedding.state_dict())
    
    start_train_time = time.time()
    
    # save best model weights
    save_dir = args.output
    now = datetime.now()
    now_str = now.strftime("%d_%m_%Y__%H_%M_%S")
    best_model_filename = '{}_{}_{}.pt'.format(args.backbone, args.bert_type, now_str)
    save_model_path_name = os.path.join(save_dir, best_model_filename)
    # biobert_path_name = os.path.join(save_dir, '{}_{}_{}.pt'.format(args.bert_type, args.backbone, now_str))
    
    # save train log
    train_log_df = pd.DataFrame(columns=['epoch', 'train_loss', 'train_acc', 
                                        'val_loss''val_acc', 'is_best_model'])
    csv_path = os.path.join(save_dir, f'train_log_{best_model_filename}.csv')
    
    for epoch in range(EPOCHS):
        print(f'\nEPOCH {epoch}/{EPOCHS - 1}')
        print('-' * 10)
        start_epoch_time = time.time()
        
        loggings = {
            'epoch': epoch
        }
        
        is_save_best_model = False
        
        for phase in ['train', 'test']:
            print('+ Training:' if phase == 'train' else '+ Validation:')
    
            with torch.set_grad_enabled(phase == 'train'):        
                # Make sure gradient tracking is on if in training, and do a pass over the data
                # model.train(phase == 'train')
                
                batch_loss, batch_acc = 0., 0.

                # Loop over training data
                for i, data in enumerate(dataloaders[phase]):
                    # Every data instance is an input + img_label pair
                    question, img, label = data['question'], data['image'], data['label']
                    question, img, label = question.to(device), img.to(device), label.to(device)
                    
                    batch_size = img.size()[0]
                    
                    # Zero your gradients for every batch!
                    optimizer.zero_grad()

                    # Make predictions for this batch
                    output = model.forward(img, question)

                    # Compute the loss and accuracy
                    loss = loss_fn(output, label)
                    
                    batch_loss += loss.item() * batch_size
                    
                    # Calculate accuracy for classification task
                    acc = utils.calc_acc(output, label)
                    batch_acc += acc

                    if phase == 'train':
                        # Backward model to compute its gradients
                        # optimizer.zero_grad()
                        loss.backward()

                        # Adjust learning weights
                        optimizer.step()

                if phase == 'train':
                    # adjust learning rate
                    scheduler.step()
                
            batch_loss /= data_size[phase]
            batch_acc /= len(dataloaders[phase])
            
            print(f'{phase} loss: %.4f, {phase} acc: %.4f' % (batch_loss, batch_acc))
            
            loggings = { **loggings,
                f'{phase}_loss': float(batch_loss),
                f'{phase}_acc': float(batch_acc),
            }
            
            # Save best model
            if phase == 'test' and batch_acc > best_val_acc:
                print('Saving this best model...')
                is_save_best_model = True
                best_val_acc = batch_acc
                best_model = copy.deepcopy(model.backbone.state_dict())
                torch.save(best_model, save_model_path_name) 
                # best_bert_model = copy.deepcopy(model.cmsa.q_emb.state_dict())
        
        row = pd.Series(data={
            **loggings,
            'is_best_model': int(is_save_best_model)
        })
        train_log_df = train_log_df.append(row, ignore_index=True)
        # save train log:
        train_log_df.to_csv(csv_path, index=False, mode='w+')  # overwrite mode

        time_epoch_elapsed = time.time() - start_epoch_time
        print('Epoch time: {:.0f}m {:.0f}s'.format(time_epoch_elapsed // 60, time_epoch_elapsed % 60))


    train_time = time.time() - start_train_time
    print('Training complete in {:.0f}m {:.0f}s'.format(train_time // 60, train_time % 60))
    print('Best val accuracy: {:4f}'.format(best_val_acc))
    print('Best model save to: ', best_model_filename)
    # torch.save(best_bert_model, biobert_path_name)
    
    return model

if __name__ == '__main__':
    args = get_arguments()
    main(args)