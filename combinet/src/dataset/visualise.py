import matplotlib.pyplot as plt
import numpy as np

from combinet.src.dataset.camvid import CamVid
from combinet.src.dataset.camvid import mean as camvid_mean
from combinet.src.dataset.camvid import std as camvid_std
from combinet.src.dataset.bacteria import Bacteria
from combinet.src.dataset.bacteria import mean as bacteria_mean
from combinet.src.dataset.bacteria import std as bacteria_std

def get_predictions(output_batch, ignore_index, threshold=0.5):
    bs, c, h, w = output_batch.size()
    tensor = output_batch.data
    values, indices = tensor.max(1)
    indices[values<=threshold] = ignore_index
    indices = indices.view(bs, h, w)
    return indices


def view_sample_predictions(inputs, output, output_entropy, targets, args, epoch, special_info = "", view_target_input = False):
    batch_size = output.size(0)
    colors = None 
    mean_std = None 
    ignore_index = None
    if args.dataset == "camvid":
        colors = CamVid.colors
        ignore_index = CamVid.ignore_index
        mean_std = (camvid_mean, camvid_std)
    elif args.dataset == "bacteria":
        colors = Bacteria.colors
        ignore_index = Bacteria.ignore_index
        mean_std = (bacteria_mean, bacteria_std) 

    pred = get_predictions(output, ignore_index)
    for i in range(max(1, batch_size)):
        if epoch == 0 or view_target_input:
          view_image(inputs[i], mean_std, args, "input_{}_{}_{}.pdf".format(epoch, i, special_info))
          view_annotated(targets[i], colors, args, "target_{}_{}_{}.pdf".format(epoch, i, special_info))
        view_annotated(pred[i], colors, args, "pred_{}_{}_{}.pdf".format(epoch, i, special_info))
        view_uncertain_image(
            output_entropy[i], args, "pred_entropy_{}_{}_{}.pdf".format(epoch, i, special_info))


def _view_annotated(tensor, colors):
    temp = tensor.cpu().numpy()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, len(colors)):
        r[temp == l] = colors[l][0]
        g[temp == l] = colors[l][1]
        b[temp == l]= colors[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = (r/255.0)  
    rgb[:, :, 1] = (g/255.0)
    rgb[:, :, 2] = (b/255.0)
    return rgb

def view_annotated(tensor, colors, args, name):
    rgb = _view_annotated(tensor, colors)
    plt.imshow(rgb)
    plt.tight_layout()
    plt.savefig(args.save+"/"+name)
    plt.clf()
    plt.close()
    
def decode_image(tensor, mean_std):
    inp = tensor.numpy().transpose((1, 2, 0))
    _mean = np.array(mean_std[0])
    _std = np.array(mean_std[1])
    inp = _std * inp + _mean
    return inp

def _view_image(tensor, mean_std):
    inp = decode_image(tensor.cpu(), mean_std)
    inp = np.clip(inp, 0, 1)
    return inp
    
def view_image(tensor, mean_std, args, name):
    inp = _view_image(tensor, mean_std)
    plt.imshow(inp)
    plt.tight_layout()
    plt.savefig(args.save+"/"+name)
    plt.clf()
    plt.close()

def view_uncertain_image(tensor, args, name):
    inp = tensor.cpu().numpy()
    fig, ax = plt.subplots()
    im = ax.imshow(inp, cmap='cool')
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Entropy [nats]", rotation=-90, va="bottom")
    plt.tight_layout()
    plt.savefig(args.save+"/"+name)
    plt.clf()
    plt.close()