import logging
from datetime import timedelta
import argparse
import torch

import sys

sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

from combinet.src.losses import SegmentationLoss
from combinet.src.architecture.models import CombiNetS, CombiNetM, CombiNetL
from combinet.src.trainer import Trainer
from combinet.src.dataset.loaders import *
from combinet.experiments.utils import segmentation_evaluation
import combinet.src.utils as utils
from combinet.src.profile import count_parameters_macs

parser = argparse.ArgumentParser("camvid")

parser.add_argument('--model', type=str, default='combinetS',
                    help='the model that we want to train')
parser.add_argument('--learning_rate', type=float,
                    default=0.001, help='init learning rate')
parser.add_argument('--crop_size', type=int,
                    default=360, help='crop size for the input')
parser.add_argument('--n_classes', type=int,
                    default=12, help='output size')
parser.add_argument('--weight_decay', type=float,
                    default=1e-3, help='weight decay regulariser')

parser.add_argument('--data', type=str, default='./../../data/',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='camvid',
                    help='dataset')
                    
parser.add_argument('--train_batch_size', type=int, default=2, help='batch size')
parser.add_argument('--valid_batch_size', type=int, default=1, help='batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size')

parser.add_argument('--epochs', type=int, default=800,
                    help='num of training epochs')
parser.add_argument('--samples', type=int,
                    default=30, help='number of output samples')

parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--load', type=str, default='EXP', help='experiment name to load')
parser.add_argument('--save_last', action='store_true', default=True,
                    help='whether to just save the last model')
parser.add_argument('--visualise', action='store_true',
                    help='whether visualise samples in the evaluation')

parser.add_argument('--num_workers', type=int,
                    default=0, help='number of workers')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--debug', action='store_true',
                    help='whether we are currently debugging')
parser.add_argument('--report_freq', type=float,
                    default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=-1, help='gpu device ids')

def main():
  args = parser.parse_args()
  args, writer = utils.parse_args(args, args.model)

  model_temp = None
  model = None
  if args.model == "combinetL":
    model_temp = CombiNetL

  elif args.model == "combinetM":
    model_temp = CombiNetM
  elif args.model == "combinetS":
    model_temp = CombiNetS
  if args.load=='EXP':
    criterion = SegmentationLoss(args)

    logging.info('## Downloading and preparing data ##')
    train_loader,  valid_loader = get_train_loaders(args)
    logging.info('# Start Re-training #')

    model = model_temp(args=args, n_classes=args.n_classes)

    logging.info('### Loading model to parallel GPUs ###')
    model = utils.model_to_gpus(model, args)
    count_parameters_macs(model)

    logging.info('### Preparing schedulers and optimizers ###')
    optimizer = torch.optim.Adam(
          model.parameters(),
          args.learning_rate,
          weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.996)

    logging.info('## Beginning Training ##')

    train = Trainer(model, criterion, optimizer, scheduler, args)

    best_error, train_time, val_time = train.train_loop(
        train_loader, valid_loader, writer)

    logging.info('## Finished training, the best observed validation error: {}, total training time: {}, total validation time: {} ##'.format(
        best_error, timedelta(seconds=train_time), timedelta(seconds=val_time)))
    args.load = args.save+"/weights_{}.pt".format(args.epochs-1)
    del model

  with torch.no_grad():
    samples = args.samples
    logging.info('## Beginning Evaluating ##')
    for i, mode in enumerate(["wa", "train"]):
      model = model_temp(args=args, n_classes=args.n_classes)
      utils.load_model(model, args.load)
      if i == 0:
        logging.info('## Model re-created: ##')
        logging.info(model.__repr__())
        args.samples = 1
        logging.info('## Weight averaging results ##')
      else:
        args.samples = samples
        logging.info('## Monte carlo dropout results ##')
      logging.info('### Loading model to parallel GPUs ###')
      model = utils.model_to_gpus(model, args)

      segmentation_evaluation(model, mode,  args)
      del model
    logging.info('# Finished #')



if __name__ == '__main__':
  main()
