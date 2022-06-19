import numpy as np
import torch
from torchvision.transforms import ToTensor
import sys

from utils import bttr_mapping, get_vocab

sys.path.append("./BTTRcustom/")
from bttr.lit_bttr import LitBTTR

diff_to_bttr = bttr_mapping('diff_to_bttr')

def bttr_beam_search_prob(
    gen_text: list, image: np.ndarray, bttr_model: LitBTTR, verbose=False
    )-> float:
    """_summary_

    Args:
        gen_text (list): _description_
        image (np.ndarray): _description_
        bttr_model (LitBTTR): _description_

    Returns:
        float: _description_
    """
    if(type(image) == np.ndarray):
        image = ToTensor()(image).numpy()
    logits = bttr_model.beam_search_logits(image).detach()
    diff_vocab = get_vocab('diff')
    bttr_vocab = get_vocab('bttr')

    if(verbose):
        hyp = bttr_model.beam_search(image)
        print(hyp)
        print(diff_vocab)
        print(' '.join([diff_vocab[token] for token in gen_text]))
        print(gen_text)
    
    i = 0
    preds = []
    for token in gen_text:
        if token in diff_to_bttr.keys():
            if verbose:
                print('{} -> {} | [{}] -> [{}]'.format(token, diff_to_bttr[token], diff_vocab[token], bttr_vocab[diff_to_bttr[token]]))
            preds.append(logits[i, diff_to_bttr[token]].item())
            i += 1
        else:
            if verbose:
                print('unknown token: {}'.format(token))

    if verbose:
        print(preds)

    if len(preds)==0:
        return(0, -100)
    avg = (sum(preds) / len(preds))

    # avoid log of 0
    preds = [p + 1e-8 for p in preds]
    seq = np.sum(np.log(preds))
    return (avg, seq)

def get_nonwhite_bounds(img, thres=255):
    pixels = np.where(img<thres)
    bounds = []
    for p in pixels:
        if len(p) == 0:
            return None
        bounds.append(max(p) + 1)
    if 0 in bounds:
        return None
    return bounds

def cut_off_white(img, thres=255):
    pixels = np.where(img<thres)
    bounds = []
    for p in pixels:
        if len(p) == 0:
            return None
        bounds.append(max(p) + 1)
    if 0 in bounds:
        return None
    return img[:bounds[0], :bounds[1], :bounds[2]]

def bttr_beam_search_prob_mean(
    gen_texts: list, images: list, bttr_model: LitBTTR
    )-> float:
    avgs, seqs = [], []
    for (img, text) in zip(images, gen_texts):
        img = cut_off_white(img)
        if(img is None or img.shape[0] + img.shape[1] + img.shape[2] <= 3):
            continue
        img = ToTensor()(1 - img)
        img = img[0, :, :]
        img = torch.unsqueeze(img, 0)

        (avg, seq) = bttr_beam_search_prob(text, img, bttr_model)
        avgs.append(avg)
        seqs.append(seq)
    if len(avgs)==0 or len(seqs)==0:
        return(0, -100)
    return ((sum(avgs)/len(avgs)), (sum(seqs)/len(seqs)) )
    
    