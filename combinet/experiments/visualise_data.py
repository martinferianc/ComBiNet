import matplotlib.gridspec as gridspec
import sys
import argparse
import logging

sys.path.append("../")
sys.path.append("../../")

from combinet.plots import PLT as plt
from combinet.src.dataset.loaders import get_test_loader, get_train_loaders
from combinet.src.dataset.visualise import _view_annotated, _view_image
from combinet.src.dataset.camvid import CamVid
from combinet.src.dataset.camvid import mean as camvid_mean
from combinet.src.dataset.camvid import std as camvid_std
from combinet.src.dataset.bacteria import Bacteria
from combinet.src.dataset.bacteria import mean as bacteria_mean
from combinet.src.dataset.bacteria import std as bacteria_std
import combinet.src.utils as utils


parser = argparse.ArgumentParser("plot_datasets")

parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--load', type=str, default='EXP', help='experiment name')
parser.add_argument('--data', type=str,
                    default='./../data/', help='experiment name')

parser.add_argument('--label', type=str, default='dataset_sample_plots',
                    help='default experiment category ')
parser.add_argument('--dataset', type=str, default='camvid',
                    help='default dataset ')
parser.add_argument('--train_batch_size', type=int, default=1, help='batch size')
parser.add_argument('--valid_batch_size', type=int, default=1, help='batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size')
parser.add_argument('--num_workers', type=int,
                    default=1, help='default batch size')
parser.add_argument('--gpu', type=int,
                    default=-1, help='portion of training data')
parser.add_argument('--seed', type=int,
                    default=1, help='input size')
parser.add_argument('--crop_size', type=int,
                    default=360, help='output size')
def main():
  args = parser.parse_args()
  logging.info('## Testing dataset ##')
  args, _ = utils.parse_args(args, args.label)

  plt.figure(figsize=(20,20))
  gs = gridspec.GridSpec(10, 6)
  gs.update(wspace=0, hspace=0)
  train_loader, valid_loader = get_train_loaders(args)
  test_loader = get_test_loader(args, shuffle=True)
  colors = None 
  mean_std = None 
  if args.dataset == "camvid":
    colors = CamVid.colors
    mean_std = (camvid_mean, camvid_std)
  elif args.dataset == "bacteria":
    colors = Bacteria.colors
    mean_std = (bacteria_mean, bacteria_std) 

  for j, loader in enumerate([train_loader, valid_loader, test_loader]):
    itr = iter(loader)
    for k in range(10):
        inputs, targets = next(itr)
        target = _view_annotated(targets[0], colors)
        image = _view_image(inputs[0], mean_std)
        plt.subplot(gs[k, j*2])
        plt.imshow(image)
        plt.subplot(gs[k, j*2+1])
        plt.imshow(target)
  plt.tight_layout()
  path = utils.check_path(args.save+'/dataset.pdf')
  plt.savefig(path)


if __name__ == '__main__':
  main()
