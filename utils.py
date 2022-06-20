import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import string
import pickle
import os
import io
import json
import math

from BTTRcustom.bttr.datamodule.vocab import CROHMEVocab

#notation clarification:
#we use the variable "alpha" for alpha_bar (cumprod 1-beta)
#the alpha in the paper is replaced with 1-beta
def explin(min, max, L):
    return tf.exp(tf.linspace(tf.math.log(min), tf.math.log(max), L))

def get_beta_set(L):
    beta_set = 0.02 + explin(1e-5, 0.4, L)
    return beta_set

s = 0.008

def cosine_f(t, T):
    c = math.cos((((t/T + s) / (1 + s)) * (math.pi/2)))
    return c * c

def cosine_alpha(t, T):
    if t == 0:
        return 1
    return (cosine_f(t, T)/cosine_f(0, T))

def cosine_beta(t, T):
    if t == 0:
        return 0
    return 1 - (cosine_alpha(t, T) / (cosine_alpha(t-1, T)))

def get_cosine_beta_set(L):
    beta_set_consine = []
    for i in range(L):
        beta_set_consine.append(cosine_beta(i, 60))
    return tf.convert_to_tensor(beta_set_consine, dtype=tf.float32)

def get_cosine_alpha_set(L):
    alpha_set_consine = []
    for i in range(L):
        alpha_set_consine.append(cosine_alpha(i, 60))
    return tf.convert_to_tensor(alpha_set_consine, dtype=tf.float32)
    
def show(strokes, name='', show_output=True, scale=1, stroke_weights=None, return_image=False):
    positions = np.cumsum(strokes, axis=0).T[:2]
    prev_ind = 0
    W, H = np.max(positions, axis=-1) - np.min(positions, axis=-1)
    fig = plt.figure(figsize=(scale * W/H, scale), frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    #if return_image:

    for ind, value in enumerate(strokes[:, 2]):
        if value > 0.5:
            if stroke_weights:
                plt.plot(positions[0][prev_ind:ind], positions[1][prev_ind:ind], color=(0.5, 0.2, 0.5))
            else:
                plt.plot(positions[0][prev_ind:ind], positions[1][prev_ind:ind], color='black')
            prev_ind = ind
        
    plt.axis('off')
    if name: plt.savefig('./' + name + '.png', bbox_inches='tight')
    if show_output:  plt.show()

    if return_image:
        plt.autoscale()
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return data

    plt.close()


def get_alphas_new(batch_size, alpha_set):
    timesteps = tf.random.uniform([batch_size, 1], maxval=len(alpha_set), dtype=tf.int32)
    alphas = tf.gather_nd(alpha_set, timesteps)
    alphas = tf.reshape(alphas, [batch_size, 1, 1])
    return alphas, timesteps

def get_alphas(batch_size, alpha_set): 
    alpha_indices = tf.random.uniform([batch_size, 1], maxval=len(alpha_set) - 1, dtype=tf.int32)
    lower_alphas = tf.gather_nd(alpha_set, alpha_indices)
    upper_alphas = tf.gather_nd(alpha_set, alpha_indices+1)
    alphas = tf.random.uniform(lower_alphas.shape, maxval=1) * (upper_alphas - lower_alphas) 
    alphas += lower_alphas
    alphas = tf.reshape(alphas, [batch_size, 1, 1])
    return alphas, alpha_indices

def standard_diffusion_step(xt, eps, beta, alpha, add_sigma=True):
    x_t_minus1 = (1 / tf.sqrt(1-beta)) * (xt - (beta * eps/(tf.sqrt(1-alpha)+0.00000001)))
    if add_sigma: 
        x_t_minus1 += tf.sqrt(beta) * (tf.random.normal(xt.shape))
    return x_t_minus1
   
def new_diffusion_step(xt, eps, beta, alpha, alpha_next):
    x_t_minus1 = (xt - tf.sqrt(1-alpha)*eps) / tf.sqrt(1-beta)
    x_t_minus1 += tf.random.normal(xt.shape) * tf.sqrt(1-alpha_next)
    return x_t_minus1

def test_diffusion_step(xt, eps, beta, alpha, add_sigma=True):
    coef1 = (1 - (1 - beta)) / (tf.sqrt(1 - alpha)+0.00000001)
    x_t_minus1 = (xt - coef1 * eps) / tf.sqrt(1 - beta)
    if add_sigma: 
        x_t_minus1 += beta * (tf.random.normal(xt.shape) * 0.5)
    return x_t_minus1

def get_stroke_bounds(strokes):
    positions = np.cumsum(strokes[0], axis=0)
    min_x = tf.reduce_min(positions[0])
    max_x = tf.reduce_max(positions[0])
    min_y = tf.reduce_min(positions[1])
    max_y = tf.reduce_max(positions[1])
    return min_x, max_x, min_y, max_y

def run_batch_inference(model, beta_set, alpha_set, text, style, tokenizer=None, time_steps=480, diffusion_mode='new', show_every=None, show_samples=True, path=None, return_image=False, return_both = False):
    if isinstance(text, str):
        text = tf.constant([tokenizer.encode(text)+[1]])
    elif isinstance(text, list) and isinstance(text[0], str):
        text_org = text.copy()
        '''
        tmp = []
        for i in text:
            tmp.append(tokenizer.encode(i)+[1])
        text = tf.constant(tmp)'''
        text = tf.constant([tokenizer.encode(text)+[1]])
    elif len(text.shape) == 1:
        text = tf.expand_dims(text, axis=0)
    bs = text.shape[0]
    L = len(beta_set)
    #alpha_set = tf.math.cumprod(1- beta_set)
    x = tf.random.normal([bs, time_steps, 2])
    
    # reverse iteration L, L-1, L-2, ... 0
    for i in range(L-1, -1, -1):
        alpha = alpha_set[i] * tf.ones([bs, 1, 1]) 
        beta = beta_set[i] * tf.ones([bs, 1, 1]) 
        a_next = alpha_set[i-1] if i>1 else 1.
        model_out, pen_lifts, att = model(x, text, tf.sqrt(alpha), style)
        if diffusion_mode == 'standard':
            x = standard_diffusion_step(x, model_out, beta, alpha, add_sigma=bool(i)) 
        elif diffusion_mode == 'new': 
            x = new_diffusion_step(x, model_out, beta, alpha, a_next)
        elif diffusion_mode == 'test':
            x = test_diffusion_step(x, model_out, beta, alpha, add_sigma=bool(i))
        
        if show_every is not None:
            if i in show_every:
                plt.imshow(att[0][0])
                plt.show()
                x_new = tf.concat([x, pen_lifts], axis=-1)
                for i in range(bs):
                    show(x_new[i], scale=1, show_output = show_samples, name=path)

    x = tf.concat([x, pen_lifts], axis=-1)
    images = []
    for i in range(bs):
        if return_image or return_both:
            img = show(x[i], scale=1, show_output = show_samples, name=path, return_image=True)
            images.append(img)
        else:
            show(x[i], scale=1, show_output = show_samples, name=path)

    '''
    print(att.shape)
    for i, token in enumerate(text_org):
        print(i, token)'''

    if return_image:
        return images
    elif return_both is False:
        return x.numpy()
    else:
        return x.numpy(), images
    
def pad_stroke_seq(x, maxlength):
    if len(x) > maxlength or np.amax(np.abs(x)) > 15: return None
    zeros = np.zeros((maxlength - len(x), 2))
    ones = np.ones((maxlength - len(x), 1))
    padding = np.concatenate((zeros, ones), axis=-1)
    x = np.concatenate((x, padding)).astype('float32')
    return x

def pad_img(img, width, height):
    pad_len = width - img.shape[1]
    padding = np.full((height, pad_len, 1), 255, dtype=np.uint8)
    img = np.concatenate((img, padding), axis=1)
    return img
	
def preprocess_data(path, max_text_len, max_seq_len, img_width, img_height, train_summary_writer=None):
    with open(path, 'rb') as f:
        ds = pickle.load(f)
        
    strokes, texts, samples = [], [], []
    unpadded = []
    for x, text, sample in ds:
        if len(text) < max_text_len:
            x = pad_stroke_seq(x, maxlength=max_seq_len)
            zeros_text = np.zeros((max_text_len-len(text), ))
            text = np.concatenate((text, zeros_text))
            h, w, _ = sample.shape
            '''
            with train_summary_writer.as_default():
                tf.summary.image("Training data", tf.expand_dims(sample, axis=0), step=0)
                print(tf.expand_dims(sample, axis=0).shape)
                from random import randrange
                #tf.summary.image("Training data {}".format(randrange(999)), [sample.astype(np.uint8)], step=0)'''

            if x is not None and sample.shape[1] < img_width: 
                unpadded.append(sample)
                sample = pad_img(sample, img_width, img_height)
                strokes.append(x)
                texts.append(text)
                samples.append(sample)
    texts = np.array(texts).astype('int32')
    samples = np.array(samples)
    return strokes, texts, samples, unpadded
    
def create_dataset(strokes, texts, samples, style_extractor, batch_size, buffer_size, num_val=0):    
    #we DO NOT SHUFFLE here, because we will shuffle later
    samples = tf.data.Dataset.from_tensor_slices(samples).batch(1)

    for count, s in enumerate(samples):
        style_vec = style_extractor(s)
        style_vec = style_vec.numpy()
        if count==0: 
            if 'BTTR' in style_extractor.__class__.__name__:
                style_vectors = np.zeros((0, style_vec.shape[1], 256))
            else:
                style_vectors = np.zeros((0, style_vec.shape[1], 1280))
        style_vectors = np.concatenate((style_vectors, style_vec), axis=0)
    style_vectors = style_vectors.astype('float32')
    
    if num_val > 0:
        dataset_val = tf.data.Dataset.from_tensor_slices((strokes[:num_val], texts[:num_val], style_vectors[:num_val]))
        dataset = tf.data.Dataset.from_tensor_slices((strokes[num_val:], texts[num_val:], style_vectors[num_val:]))
        dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
        dataset_val = dataset_val.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
        return dataset, style_vectors, dataset_val
    else:
        dataset = tf.data.Dataset.from_tensor_slices((strokes, texts, style_vectors))
        dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
        return dataset, style_vectors
    
class Tokenizer:
    def __init__(self):
        self.tokens = {}
        self.chars = {}
        self.text = '_' + string.ascii_letters + string.digits + '.?!,\'\"- '
        self.numbers = np.arange(2, len(self.text)+2)
        self.create_dict()
        self.vocab_size = len(self.text)+2
    
    def create_dict(self):
        for char, token, in zip(self.text, self.numbers):
            self.tokens[char] = token
            self.chars[token] = char
        self.chars[0], self.chars[1] = ' ', '<end>' #only for decoding

    def encode(self, text):
        tokenized = []
        for char in text:
            if char in self.text: tokenized.append(self.tokens[char])
            else: tokenized.append(2) #unknown character is '_', which has index 2
         
        tokenized.append(1) #1 is the end of sentence character
        return tokenized
    
    def decode(self, tokens):
        if isinstance(tokens, tf.Tensor): tokens = tokens.numpy()
        text = [self.chars[token] for token in tokens]
        return ''.join(text)

class CrohmeTokenizer:
    def __init__(self):
        f = open("./data/latex_histogramm.json", "r")
        hist = json.loads(f.read())
        self.vocab = {k: i+2 for i, k in enumerate(hist)}
        self.vocab[' '], self.vocab['<end>'] = 0, 1
        # reverse dict
        self.lookup = {v: k for k, v in self.vocab.items()}
    
    def encode(self, tokens):
        return [self.vocab[t] if t in self.vocab.keys() else 0 for t in tokens] + [1]

    def decode(self, tokens):
        return [self.lookup[t] for t in tokens if t in self.lookup.keys()]

# calculate frechet inception distance
def calculate_fid(act1, act2):
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

def get_vocab(mode='diff', reverse=True):
    d = {}
    if mode == 'diff':
        d = CrohmeTokenizer().vocab
    elif mode == 'bttr':
        d = CROHMEVocab().word2idx
    if reverse:
        return {v: k for k, v in d.items()}
    else:
        return d

def bttr_mapping(mode='diff_to_bttr'):
    mapping = CROHMEVocab().word2idx
    mapping2 = CrohmeTokenizer().vocab
    bttr_keys = mapping.keys()
    diff_keys = mapping2.keys()
    
    if mode == 'bttr_to_diff':
        bttr_to_diff = {}
        for bttr_key in bttr_keys:
            if bttr_key in diff_keys:
                #print('{} => {}'.format(mapping[bttr_key], mapping2[bttr_key]))
                bttr_to_diff[mapping[bttr_key]] = mapping2[bttr_key]
        return bttr_to_diff
    elif mode == 'diff_to_bttr':
        diff_to_bttr = {}
        for diff_key in diff_keys:
            if diff_key in bttr_keys:
                diff_to_bttr[mapping2[diff_key]] = mapping[diff_key]
        return diff_to_bttr