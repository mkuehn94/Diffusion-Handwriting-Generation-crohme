import numpy as np
import torch
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

    logits = bttr_model.beam_search_logits(image)

    diff_vocab = get_vocab('diff')
    bttr_vocab = get_vocab('bttr')

    if(verbose):
        hyp = bttr_model.beam_search(image)
        print(hyp)
        print(' '.join([diff_vocab[token] for token in gen_text]))
        print(gen_text)
    
    i = 0
    preds = []
    for token in gen_text:
        if token in diff_to_bttr.keys():
            if verbose:
                print('{} -> {} | [{}] -> [{}]'.format(token, diff_to_bttr[token], diff_vocab[token], bttr_vocab[diff_to_bttr[token]]))
            #print('{} => {}'.format()
            #print(torch.max(logits[i]))
            #print(torch.argmax(logits[i]))
            #print(logits[i, diff_to_bttr[token]].item())
            preds.append(logits[i, diff_to_bttr[token]].item())
            i += 1
        else:
            if verbose:
                print('unknown token: {}'.format(token))

    avg = (sum(preds) / len(preds))
    seq = np.prod(preds)
    return (avg, seq)
    
    