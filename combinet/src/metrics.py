import torch 
import torch.nn.functional as F
from combinet.src.dataset.camvid import CLASS_WEIGHT_CAMVID
from combinet.src.dataset.bacteria import CLASS_WEIGHT_BACTERIA

from combinet.src.utils import nanmean

class StreamSegMetrics():
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, args):
        self.args = args
        self.n_classes = args.n_classes
        self.confusion_matrix = torch.zeros((self.n_classes, self.n_classes))
        
        self.entropy = AverageMeter()
        self.nll = AverageMeter()
        self.ece = AverageMeter()

        if self.args.dataset == "camvid":
            self.class_weight = CLASS_WEIGHT_CAMVID
        elif self.args.dataset == "bacteria":
            self.class_weight = CLASS_WEIGHT_BACTERIA
        elif self.args.dataset == "random":
            self.class_weight = torch.ones((args.n_classes,))

    @staticmethod
    def get_predictions(output):
        bs, c, h, w = output.size()
        tensor = output.data
        _, indices = tensor.max(1)
        indices = indices.view(bs, h, w)
        return indices

    @staticmethod
    def ec_error(output, target):
        _ece = 0.0
        confidences, predictions = torch.max(output, 1)
        accuracies = predictions.eq(target)

        bin_boundaries = torch.linspace(0, 1, 10 + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * \
                confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                _ece += torch.abs(avg_confidence_in_bin -
                                    accuracy_in_bin) * prop_in_bin
        _ece = _ece if isinstance(_ece, float) else _ece.item()
        return _ece


    def update(self, labels, output_mean):
        with torch.no_grad():
            bs, c, h, w = output_mean.size()
            predictions = StreamSegMetrics.get_predictions(output_mean)
            for lt, lp in zip(labels, predictions):
                if labels.is_cuda or predictions.is_cuda and not self.confusion_matrix.is_cuda:
                    self.confusion_matrix = self.confusion_matrix.to(predictions.device)
                self.confusion_matrix += self._fast_hist(lt.view(-1), lp.view(-1))
            nll = F.nll_loss(torch.log(output_mean+1e-8), labels, weight=self.class_weight.to(predictions.device)
                            if predictions.is_cuda else self.class_weight, reduction='mean').item()
            ece =  self.ec_error(output_mean.view(-1, self.n_classes), labels.view(-1))*100
            entropy = -(torch.sum(torch.log(output_mean+1e-8)*output_mean)/(bs * h * w)).item()

            self.nll.update(nll, bs * h * w)
            self.ece.update(ece, bs * h * w)
            self.entropy.update(entropy, bs * h * w)
    
    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = torch.bincount(
            self.n_classes * label_true[mask].long() + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        with torch.no_grad():
            hist = self.confusion_matrix
            # This is to avoid computation with respect to the background class for CamVid only
            if self.args.dataset=="camvid":
                hist = hist[:-1, :-1]
            acc = torch.diag(hist).sum() / hist.sum()
            acc_cls = torch.diag(hist) / hist.sum(dim=1)
            acc_cls = nanmean(acc_cls)
            iu = torch.diag(hist) / (hist.sum(dim=1) + hist.sum(dim=0) - torch.diag(hist))
            mean_iu = nanmean(iu)
            dice = 2 * torch.diag(hist) / (hist.sum(dim=1) + hist.sum(dim=0))
            mean_dice = nanmean(dice)
        return (1.-acc.item())*100, (1.-acc_cls.item())*100, mean_iu.item()*100, mean_dice.item(), self.ece.avg, self.entropy.avg, self.nll.avg

        
    def reset(self):
        self.confusion_matrix = torch.zeros((self.n_classes, self.n_classes))
        self.entropy.reset()
        self.nll.reset()
        self.ece.reset()


class AverageMeter(object):
      def __init__(self):
        self.reset()

      def reset(self):
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0.0

      def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

