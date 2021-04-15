
import torch
import combinet.src.utils as utils
import time
import logging
from combinet.src.metrics import StreamSegMetrics, AverageMeter
from combinet.src.dataset.visualise import view_sample_predictions

class Trainer():
  def __init__(self, model, criterion, optimizer, scheduler, args):
    super().__init__()
    self.model = model
    self.criterion = criterion
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.args = args

    self.train_time = 0.0
    self.val_time = 0.0

    self.train_metrics = StreamSegMetrics(args)
    self.valid_metrics = StreamSegMetrics(args)

    self.writer = None

  def _scalar_logging(self, obj, nll, global_error, class_error, miou, mean_dice, ece, entropy, info, iteration):
    self.writer.add_scalar(info+'global_error', global_error, iteration)
    self.writer.add_scalar(info+'class_error', class_error, iteration)
    self.writer.add_scalar(info+'miou', miou, iteration)
    self.writer.add_scalar(info+'mean_dice', mean_dice, iteration)
    self.writer.add_scalar(info+'loss', obj, iteration)
    self.writer.add_scalar(info+'ece', ece, iteration)
    self.writer.add_scalar(info+'entropy', entropy, iteration)
    self.writer.add_scalar(info+'nll', nll, iteration)

  def _get_average_meters(self):
    obj = AverageMeter()
    return obj

  def train_loop(self, train_loader, valid_loader, writer=None, special_infor=""):
    self.writer = writer
    best_miou = -float('inf')
    train_global_error = train_class_error = train_miou = train_mean_dice = train_obj =  train_nll = train_ece = train_entropy = None
    val_global_error = val_class_error = val_miou = val_mean_dice = val_obj = val_nll = val_ece = val_entropy  = None

    for epoch in range(self.args.epochs):

      if self.scheduler is not None:
        lr = self.optimizer.param_groups[0]['lr']
      else:
        lr = self.args.learning_rate

      if self.writer is not None:
        writer.add_scalar('Train/learning_rate', lr, epoch)

      logging.info(
          '### Epoch: [%d/%d], Learning rate: %e', self.args.epochs,
          epoch, lr)

      train_obj, train_nll, train_global_error, train_class_error, train_miou, train_mean_dice, train_ece, train_entropy = self.train(
          epoch, train_loader)

      logging.info('#### Train | Global Error: %f, Class Error: %f,  mIoU: %f, mean Dice: %f, Train loss: %f,  Train NLL: %f, Train ECE %f, Train entropy: %f ####',
                   train_global_error, train_class_error, train_miou, train_mean_dice, train_obj,  train_nll,  train_ece, train_entropy)

      if self.writer is not None:
        self._scalar_logging(train_obj, train_nll, 
                             train_global_error, train_class_error, train_miou, train_mean_dice, train_ece, train_entropy,  "Train/", epoch)

      if valid_loader is not None:
        val_obj, val_nll,  val_global_error, val_class_error, val_miou, val_mean_dice, val_ece, val_entropy = self.infer(epoch,
                                                                                                             valid_loader, "Valid")
        logging.info('#### Valid | Global Error: %f, Class Error: %f,  mIoU: %f, mean Dice: %f, Valid loss: %f, Valid NLL: %f, Valid ECE: %f, Valid entropy: %f ####',
                     val_global_error, val_class_error, val_miou, val_mean_dice, val_obj,  val_nll, val_ece, val_entropy)

        if self.writer is not None:
          self._scalar_logging(val_obj, val_nll, val_global_error, val_class_error,
                               val_miou, val_mean_dice, val_ece, val_entropy, "Valid/", epoch)

      if self.args.save_last or val_miou > best_miou:
        if epoch == self.args.epochs-1:
          utils.save_model(self.model, self.args, special_infor+"_{}".format(epoch))
        if val_miou > best_miou:
          best_miou = val_miou
          utils.save_model(self.model, self.args, special_infor+"_best")
          logging.info(
            '### Epoch: [%d/%d], Best mIoU: %f changed! ###', self.args.epochs,
            epoch, best_miou)
        logging.info(
            '### Epoch: [%d/%d], Saving model! Current best mIoU: %f ###', self.args.epochs,
            epoch, best_miou)

      if epoch >= 1 and self.scheduler is not None:
        self.scheduler.step()

      if hasattr(self.model, 'log'):
        self.model.log(self.writer, epoch)

    return best_miou, self.train_time, self.val_time

  def _step(self, input, target, optimizer, epoch, metric_container, train_timer, view=False):
    start = time.time()
    if next(self.model.parameters()).is_cuda:
      input = input.cuda()
      target = target.cuda()

    if optimizer is not None:
      optimizer.zero_grad()

    output = self.model(input)
    output_mean = output
    obj = self.criterion(
        output_mean, target)
    if optimizer is not None and obj == obj:
      obj.backward()
      for p in self.model.parameters():
        if p.grad is not None:
          p.grad[p.grad != p.grad] = 0
      optimizer.step()
    metric_container.update(target.detach(), output_mean.detach())

    if train_timer:
      self.train_time += time.time() - start
    else:
      self.val_time += time.time() - start

    if not self.model.training and view and self.args.visualise:
      output_entropy = utils.get_entropy_output(output_mean.detach())
      view_sample_predictions(input, output_mean, output_entropy, target, self.args, epoch)
    return obj.item()

  def train(self, epoch, loader):
    self.train_metrics.reset()
    self.model.train()

    obj= self._get_average_meters()
    global_error = class_error = miou = mean_dice = ece = entropy = nll = None 

    for step, (input, target) in enumerate(loader):
      n = input.shape[0]
      _obj = self._step(
          input, target, self.optimizer, epoch, self.train_metrics, True)

      obj.update(_obj, n)

      global_error, class_error, miou, mean_dice, ece, entropy, nll = self.train_metrics.get_results()

      if step % self.args.report_freq == 0:
        logging.info('##### Train step: [%03d/%03d] | Global Error: %f, Class Error: %f, mIoU: %f, mean Dice: %f, Loss: %f,  NLL: %f, ECE: %f, Entropy: %f #####',
                     len(loader),  step, global_error, class_error, miou, mean_dice, obj.avg,  nll,   ece, entropy)
      if self.args.debug:
        break

    return obj.avg,  nll, global_error, class_error, miou, mean_dice, ece, entropy

  def infer(self, epoch, loader, dataset="Valid"):
    self.valid_metrics.reset()
    self.model.eval(mode="wa")
    
    obj = self._get_average_meters()
    global_error = class_error = miou = mean_dice = ece = entropy = nll = None 
    with torch.no_grad():
      try:
        for step, (input, target) in enumerate(loader):
          n = input.shape[0]
          _obj = self._step(
              input, target, None, epoch, self.valid_metrics, False, step == 0)

          obj.update(_obj, n)

          global_error, class_error, miou, mean_dice, ece, entropy, nll = self.valid_metrics.get_results()

          if step % self.args.report_freq == 0:
            logging.info('##### {} step: [{}/{}] | Global Error: {}, Class Error: {}, mIoU: {}, mean Dice: {}, Loss: {}, NLL: {},  ECE: {}, Entropy: {} #####'.format(
                        dataset, len(loader), step, global_error,  class_error,  miou, mean_dice, obj.avg, nll, ece, entropy))

          if self.args.debug:
            break
      except:
        pass

      return obj.avg, nll, global_error, class_error, miou, mean_dice, ece, entropy
