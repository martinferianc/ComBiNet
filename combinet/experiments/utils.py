import combinet.src.utils as utils
from combinet.src.dataset.loaders import *
from combinet.src.metrics import StreamSegMetrics
from combinet.src.dataset.visualise import view_sample_predictions
import torch
import sys
import logging
from tqdm import tqdm
sys.path.append('../')


def segmentation_evaluation(model, mode, args, results_name="/results.pickle", random=True):
    model.eval(mode =mode)
    logging.info('### Monte Carlo Dropout: {} ###'.format(mode))
    results = utils.load_pickle(args.save+results_name)
    _evaluate_and_record(model, results, args)

    if random:
      dataset = args.dataset
      args.dataset = 'random'
      test_loader = get_test_loader(args)
      global_error, class_error, miou, mean_dice, ece, entropy, nll = _evaluate_with_loader(
          test_loader, model, args, split = 'random')
      args.dataset = dataset

      logging.info("## Random Global Error: {} ##".format(global_error))
      logging.info("## Random Class Error: {} ##".format(class_error))
      logging.info("## Random ECE: {} ##".format(ece))
      logging.info("## Random Entropy: {} ##".format(entropy))
      logging.info("## Random NLL: {} ##".format(nll))
      logging.info("## Random mIoU: {} ##".format(miou))
      logging.info("## Random mean Dice: {} ##".format(mean_dice))

      results["entropy"]["random"] = entropy
      results["ece"]["random"] = ece
      results["global_error"]["random"] = global_error
      results["class_error"]["random"] = class_error
      results["nll"]["random"] = nll
      results["miou"]["random"] = miou
      results["mean_dice"]["random"] = mean_dice
    save_name = results_name
    if save_name == "/results.pickle":
        save_name = "/results_wa.pickle" if mode=="wa" else "/results_mcd.pickle"
    utils.save_pickle(results, args.save+save_name, True)
    logging.info("## Results: {} ##".format(results))

def _evaluate_with_loader(loader, model, args, split="test"):
    with torch.no_grad():
      metrics = StreamSegMetrics(args)
      for i, (input, target) in enumerate(tqdm(loader)):
        if next(model.parameters()).is_cuda:
          input = input.cuda()
          target = target.cuda()
        samples = []
        for j in range(args.samples):
          out = model(input).detach()
          samples.append(out)
          if j >= 2 and args.debug:
              break

        outputs_mean = torch.stack(samples, dim=1).mean(dim=1)
        metrics.update(target, outputs_mean)

        if args.visualise:
          output_entropy = utils.get_entropy_output(outputs_mean)
          view_sample_predictions(
              input.cpu(), outputs_mean.cpu(), output_entropy.cpu(), target.cpu(), args, i, split, view_target_input=True)
        
        if args.gpu!=-1:
          torch.cuda.empty_cache()
        if args.debug:
          break
      
      global_error, class_error, miou, mean_dice, ece, entropy, nll = metrics.get_results()
     

      return global_error, class_error, miou, mean_dice, ece, entropy, nll


def _evaluate_and_record(model, results, args, train=True, valid=True, test=True):
    train_loader, val_loader = get_train_loaders(args)
    test_loader = get_test_loader(args)

    if train:
      global_error, class_error, miou, mean_dice, ece, entropy, nll = _evaluate_with_loader(
          train_loader, model, args, 'train')
      logging.info("## Train Global Error: {} ##".format(global_error))
      logging.info("## Train Class Error: {} ##".format(class_error))
      logging.info("## Train ECE: {} ##".format(ece))
      logging.info("## Train Entropy: {} ##".format(entropy))
      logging.info("## Train NLL: {} ##".format(nll))
      logging.info("## Train mIoU: {} ##".format(miou))
      logging.info("## Train mean Dice: {} ##".format(mean_dice))

      results["entropy"]["train"] = entropy
      results["ece"]["train"] = ece
      results["global_error"]["train"] = global_error
      results["class_error"]["train"] = class_error
      results["nll"]["train"] = nll
      results["miou"]["train"] = miou
      results["mean_dice"]["train"] = mean_dice


    if valid:
      global_error, class_error, miou, mean_dice, ece, entropy, nll = _evaluate_with_loader(
          val_loader, model, args, 'valid')
      logging.info("## Valid Global Error: {} ##".format(global_error))
      logging.info("## Valid Class Error: {} ##".format(class_error))
      logging.info("## Valid ECE: {} ##".format(ece))
      logging.info("## Valid Entropy: {} ##".format(entropy))
      logging.info("## Valid NLL: {} ##".format(nll))
      logging.info("## Valid mIoU: {} ##".format(miou))
      logging.info("## Valid mean Dice: {} ##".format(mean_dice))

      results["entropy"]["valid"] = entropy
      results["ece"]["valid"] = ece
      results["global_error"]["valid"] = global_error
      results["class_error"]["valid"] = class_error
      results["nll"]["valid"] = nll
      results["miou"]["valid"] = miou
      results["mean_dice"]["valid"] = mean_dice

    if test:
      global_error, class_error, miou, mean_dice, ece, entropy, nll = _evaluate_with_loader(
          test_loader, model, args, 'test')
      logging.info("## Test Global Error: {} ##".format(global_error))
      logging.info("## Test Class Error: {} ##".format(class_error))
      logging.info("## Test ECE: {} ##".format(ece))
      logging.info("## Test Entropy: {} ##".format(entropy))
      logging.info("## Test NLL: {} ##".format(nll))
      logging.info("## Test mIoU: {} ##".format(miou))
      logging.info("## Test mean Dice: {} ##".format(mean_dice))

      results["entropy"]["test"] = entropy
      results["ece"]["test"] = ece
      results["global_error"]["test"] = global_error
      results["class_error"]["test"] = class_error
      results["nll"]["test"] = nll
      results["miou"]["test"] = miou
      results["mean_dice"]["test"] = mean_dice
