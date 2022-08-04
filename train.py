from contextlib import ExitStack
import tensorflow as tf
import torch
import numpy as np
import matplotlib.pyplot as plt
import utils
import os
import nn
import time
import random
import argparse
import datetime
import scipy
from tensorflow.keras.applications.inception_v3 import InceptionV3
import skimage
import skimage.transform
from torchvision.transforms import ToTensor
import json
import pickle

from validation import bttr_beam_search_prob
from validation import bttr_beam_search_prob_mean, cut_off_white

import sys
sys.path.append("./BTTRcustom/")
from bttr.lit_bttr import LitBTTR

SIGMA_LOSS_COEF = 0.001

#@tf.function
def train_step(x, pen_lifts, text, style_vectors, interpolate_alphas, is_val_step, glob_args):
    model, alpha_set, beta_set, bce, train_loss, optimizer, train_summary_writer, loss_type, history, importance_sampling, l0_type, ignore_last_loss = glob_args
    alpha_bars_set = tf.math.cumprod(alpha_set)
    if(interpolate_alphas):
        if ignore_last_loss:
            alpha_bars, timesteps = utils.get_alphas(len(x), alpha_bars_set[:-1])
        else:
            alpha_bars, timesteps = utils.get_alphas(len(x), alpha_bars_set)
    else:
        if ignore_last_loss:
            alpha_bars, timesteps = utils.get_alphas_new(len(x), alpha_bars_set[:-1])
        else:
            alpha_bars, timesteps = utils.get_alphas_new(len(x), alpha_bars_set)
    eps = tf.random.normal(tf.shape(x))
    x_perturbed = tf.sqrt(alpha_bars) * x 
    x_perturbed += tf.sqrt(1 - alpha_bars) * eps

    # calculate loss
    with ExitStack() as stack:
        if is_val_step == False:
            tape = stack.enter_context(tf.GradientTape())
        if loss_type == "vlb":
            batch_size = len(x)
            
            alpha_bar_set_prev = tf.concat(values=([1], alpha_bars_set[:-1]), axis=0)

            beta_bar_set = beta_set * (1.0 - alpha_bar_set_prev) / (1.0 - alpha_bars_set)
            beta_bar_set_log_clipped = tf.math.log(tf.concat(values=([beta_bar_set[1]], beta_bar_set[1:]), axis=0))

            model_log_variance = tf.gather_nd(beta_bar_set_log_clipped, timesteps) #shape: BATCH_SIZE
            model_log_variance = tf.reshape(model_log_variance, [batch_size, 1, 1])
            # [96] -> [96, 488, 2]
            #b = tf.expand_dims(model_log_variance, axis=1) # [BS, 1]
            #c = tf.expand_dims(b, axis=2)          # [BS, 1, 1]
            #model_log_variance = tf.tile(c, (1, 488, 2))

            beta_bar_set_clipped = tf.concat(values=([beta_bar_set[1]], beta_bar_set[1:]), axis=0)
            beta_bars = tf.gather_nd(beta_bar_set_clipped, timesteps)
            beta_bars = tf.reshape(beta_bars, [batch_size, 1, 1])

            beta_bars_log = tf.gather_nd(beta_bar_set_log_clipped, timesteps)
            beta_bars_log = tf.reshape(beta_bars_log, [batch_size, 1, 1])
            
            betas = tf.gather_nd(beta_set, timesteps)
            betas = tf.reshape(betas, [batch_size, 1, 1])

            min_log = tf.math.log(beta_bars)
            max_log = tf.math.log(betas)

            
            if model.learn_sigma:
                score, pl_pred, sigma_logits, att = model(x_perturbed, text, tf.sqrt(alpha_bars), style_vectors, training=True)
                sigma_logits = (sigma_logits + 1) / 2
                model_log_variance = sigma_logits * max_log + (1 - sigma_logits) * min_log
            else:
                score, pl_pred, att = model(x_perturbed, text, tf.sqrt(alpha_bars), style_vectors, training=True)
                
            vlb = nn.sigma_los_vb(x_perturbed, x, timesteps, alpha_set, alpha_bars_set, alpha_bar_set_prev, beta_set, beta_bars_log, score, model_log_variance, history, importance_sampling, train_summary_writer, step=optimizer.iterations, l0_loss=l0_type)

            pl_loss = tf.reduce_mean(bce(pen_lifts, pl_pred) * tf.squeeze(alpha_bars, -1))
            if is_val_step == False:
                with train_summary_writer.as_default():
                    tf.summary.scalar('pl_loss', pl_loss, step=optimizer.iterations)
                    tf.summary.scalar('vlb', vlb, step=optimizer.iterations)
            loss = vlb + pl_loss
        elif loss_type == "hybrid":
            # hybrid loss
            batch_size = len(x)
            betas = tf.gather_nd(beta_set, timesteps)
            betas = tf.reshape(betas, [batch_size, 1, 1])
            alpha_set_prev = tf.concat(values=([1], alpha_set[:-1]), axis=0)
            beta_bar_set = beta_set * (1.0 - alpha_set_prev) / (1.0 - alpha_set)
            beta_bar_set_clipped = tf.concat(values=([beta_bar_set[1]], beta_bar_set[1:]), axis=0)

            beta_bars = tf.gather_nd(beta_bar_set_clipped, timesteps)
            beta_bars = tf.reshape(beta_bars, [batch_size, 1, 1])

            min_log = tf.math.log(beta_bars)
            max_log = tf.math.log(betas)
            
            score, pl_pred, sigma_logits, att = model(x_perturbed, text, tf.sqrt(alphas), style_vectors, training=True)
            model_log_variance = sigma_logits * max_log + (1 - sigma_logits) * min_log
            sigma = tf.exp(model_log_variance)
            loss = nn.loss_fn(eps, score, pen_lifts, pl_pred, alphas, bce)
        else:
            # simple mse loss
            alphas = tf.gather_nd(alpha_set, timesteps)
            score, pl_pred, att = model(x_perturbed, text, tf.sqrt(alpha_bars), style_vectors, training=True)
            loss_simple = tf.reduce_mean(tf.reduce_sum(tf.square(eps - score), axis=-1))
            pl_loss = tf.reduce_mean(bce(pen_lifts, pl_pred) * tf.squeeze(alpha_bars, -1))
            if is_val_step == False:
                with train_summary_writer.as_default():
                    tf.summary.scalar('pl_loss', pl_loss, step=optimizer.iterations)
                    tf.summary.scalar('loss_simple', loss_simple, step=optimizer.iterations)
            loss = loss_simple + pl_loss
            #loss = nn.loss_fn(eps, score, pen_lifts, pl_pred, alpha_bars, bce)
        #if model.learn_sigma:
        #    loss += SIGMA_LOSS_COEF * nn.sigma_los_vb(x_perturbed, x, timesteps, alphas, betas, alpha_set, alpha_set_prev, beta_set, beta_bars, score, sigma, train_summary_writer, step=optimizer.iterations)
    
    if is_val_step == False:
        gradients = tape.gradient(loss, model.trainable_variables)  
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
    return loss


# scale an array of images to a new size
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = skimage.transform.resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return np.asarray(images_list)

def calculate_fid(model, images1, images2):
	# calculate activations
	act1 = model.predict(images1)
	act2 = model.predict(images2)
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = np.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if np.iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid


def find_stroke_center(abs_stroke):
    min_x = abs_stroke[:, 0].min()
    min_y = abs_stroke[:, 1].min()
    max_x = abs_stroke[:, 0].max()
    max_y = abs_stroke[:, 1].max()
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    return np.array([center_x, center_y])

def delta_to_abs_2(delta_strokes):
    drawn = np.concatenate(([0], delta_strokes[:,2]))
    drawn = np.expand_dims(drawn, axis=1)
    abs = np.cumsum(delta_strokes[:,0:2], axis=0)
    abs = np.concatenate(([[0, 0]], abs))

    abs = np.concatenate((abs, drawn), axis=1)
    return abs

def delta_to_abs_3(delta_strokes):
    BATCH_SIZE = delta_strokes.shape[0]
    drawn = tf.concat([tf.zeros((BATCH_SIZE,1)), delta_strokes[:,:,2]], axis=1)
    drawn = tf.expand_dims(drawn, axis=2)
    abs = tf.cumsum(delta_strokes[:,:,0:2], axis=1)
    abs = tf.concat([tf.zeros((BATCH_SIZE,1, 2)), abs], axis=1)
    abs = tf.concat([abs, drawn], axis=2)
    return abs
    

def abs_to_delta_3(abs_strokes):
    delta = abs_strokes[:,1:,0:2] - abs_strokes[:,:-1,0:2]
    draws = tf.expand_dims(abs_strokes[:,1:,2], axis=2)
    delta = tf.concat([delta, draws], axis=2)
    return delta

def abs_to_delta_2(abs_strokes):
    delta = abs_strokes[1:,0:2] - abs_strokes[:-1,0:2]
    draws = np.expand_dims(abs_strokes[1:,2], axis=1)
    delta = np.concatenate((delta, draws), axis=1)
    return delta

def shear_abs_strokes(abs_strokes, shear_factor):
    abs_strokes[:, 0:2] = np.dot(abs_strokes[:, 0:2], np.array([[1, shear_factor], [0, 1]]))
    return abs_strokes

def rotate_abs_strokes(abs_strokes, radians = np.pi/2):
    abs_strokes[:, 0:2] = np.dot(abs_strokes[:, 0:2], np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]]))
    return abs_strokes

def rotate_abs_strokes_tf(abs_strokes, radians = np.pi/2):
    drawn = tf.expand_dims(abs_strokes[:, :, 2], axis=2)
    strokes = tf.matmul(abs_strokes[:, :, 0:2], tf.constant([[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]], dtype=tf.float32))
    return tf.concat([strokes, drawn], axis=2)

def find_stroke_center_tf(abs_stroke):
    min_x = tf.reduce_min(abs_stroke[:, :, 0], axis=1)
    min_y = tf.reduce_min(abs_stroke[:, :, 1], axis=1)
    max_x = tf.reduce_max(abs_stroke[:, :, 0], axis=1)
    max_y = tf.reduce_max(abs_stroke[:, :, 1], axis=1)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    center = tf.stack([center_x, center_y], axis=1)
    return center

def rotate_delta_stroke(strokes, angle):
    # rotate the strokes
    abs_strokes_batch3 = delta_to_abs_3(strokes)
    center = find_stroke_center_tf(abs_strokes_batch3)

    # move cneter to 0
    abs_strokes = tf.transpose(abs_strokes_batch3[:, :, 0:2], perm=[1, 0, 2])
    abs_strokes = tf.subtract(abs_strokes, center)
    abs_strokes = tf.transpose(abs_strokes, perm=[1, 0, 2])
    abs_strokes_batch3 = tf.concat([abs_strokes, tf.expand_dims(abs_strokes_batch3[:, :, 2], axis=2)], axis=2)

    abs_strokes_batch3 = rotate_abs_strokes_tf(abs_strokes_batch3, angle)
    delta_strokes_batch3 = abs_to_delta_3(abs_strokes_batch3)
    '''
    abs_strokes_batch2 = np.zeros((strokes.shape[0], strokes.shape[1]+1, strokes.shape[2]))
    delta_strokes_batch2 = np.zeros_like(strokes)
    center2 = np.zeros((strokes.shape[0], 2))
    for i, stroke_groups in enumerate(strokes):
        abs_strokes2 = delta_to_abs_2(stroke_groups)
        center2[i] = find_stroke_center(abs_strokes2)
        abs_strokes2[:, 0:2] -= find_stroke_center(abs_strokes2)
        abs_strokes2 = rotate_abs_strokes(abs_strokes2, radians=angle)
        abs_strokes_batch2[i] = abs_strokes2
        delta_strokes_batch2[i] = abs_to_delta_2(abs_strokes2)
    tf.print(tf.math.reduce_sum(abs_strokes_batch3 - abs_strokes_batch2))
    tf.print(tf.math.reduce_sum(delta_strokes_batch3 - delta_strokes_batch2))
    tf.print(tf.math.reduce_sum(center - center2))'''
    return delta_strokes_batch3


def pertubate_delta_strokes(delta_strokes):
    delta_strokes0 = tf.random.normal(delta_strokes[:, :, 0].shape, delta_strokes[:, :, 0], 0.125 * abs(delta_strokes[:, :, 0]))
    delta_strokes1 = tf.random.normal(delta_strokes[:, :, 1].shape, delta_strokes[:, :, 1], 0.125 * abs(delta_strokes[:, :, 1]))
    return tf.stack([delta_strokes0, delta_strokes1], axis=2)

val_model = InceptionV3()
ckpt = './BTTRcustom/checkpoints/pretrained-2014.ckpt'
lit_model = LitBTTR.load_from_checkpoint(ckpt)
def train(dataset, iterations, model, optimizer, alpha_set, beta_set, DIFF_STEPS, print_every=1000, save_every=10000, interpolate_alphas=True, train_summary_writer = None, val_every = None, val_dataset = None, dataset_val = None, pertubate = False, rotate = False, loss_type='simple', importance_sampling=False, l0_type='nll', weights_dir='weights', ignore_last_loss=False):
    assert DIFF_STEPS == len(alpha_set) == len(beta_set)
    s = time.time()
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    train_loss = tf.keras.metrics.Mean()
    val_loss = tf.keras.metrics.Mean()
    history = np.full((DIFF_STEPS, 10), np.nan)
    #history = [[np.nan] * 10 for i in range(60)]
    for count, (strokes, text, style_vectors) in enumerate(dataset.repeat(5000)):
        if rotate:
            angle = random.uniform(-np.pi/12, np.pi/12)
            strokes = rotate_delta_stroke(strokes, angle)
        strokes, pen_lifts = strokes[:, :, :2], strokes[:, :, 2:]
        if pertubate:
            strokes = pertubate_delta_strokes(strokes)
            
        glob_args = model, alpha_set, beta_set, bce, train_loss, optimizer, train_summary_writer, loss_type, history, importance_sampling, l0_type, ignore_last_loss
        loss = train_step(strokes, pen_lifts, text, style_vectors, interpolate_alphas, False, glob_args)
        
        if optimizer.iterations%print_every==0:
            print("Iteration %d, Loss %f, Time %ds" % (optimizer.iterations, train_loss.result(), time.time()-s))
            if dataset_val is not None:
                val_loss = 0
                n_valbatch = 0
                for (val_strokes, val_text, val_style_vectors) in dataset_val:
                    n_valbatch += 1
                    val_strokes, val_pen_lifts = val_strokes[:, :, :2], val_strokes[:, :, 2:]
                    val_loss += train_step(val_strokes, val_pen_lifts, val_text, val_style_vectors, interpolate_alphas, True, glob_args).numpy()
                tf.print("val_loss:", val_loss/n_valbatch, output_stream=sys.stdout)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=optimizer.iterations)
                if dataset_val is not None:
                    tf.summary.scalar('val_loss', val_loss/n_valbatch, step=optimizer.iterations)

            train_loss.reset_states()

        if (optimizer.iterations+1) % save_every==0:
            save_path = './{}/model_step{}.h5'.format(weights_dir, optimizer.iterations+1)
            model.save_weights(save_path)
            
        if optimizer.iterations > iterations:
            model.save_weights('./{}/model.h5'.format(weights_dir))
            break

        if val_every is not None and (optimizer.iterations) % val_every==0:
            print('validation step: {}'.format(optimizer.iterations))
            seq_lengths = np.count_nonzero(val_dataset['texts'] > 0, axis=1)
            indices = np.where(seq_lengths < 15)
            indices = indices[0]
            np.random.shuffle(indices)

            #beta_set = utils.get_beta_set(DIFF_STEPS)

            # perform n inference steps
            generated_images = []
            generated_texts = []
            BATCH_SIZE = 32
            for i in range(1):
                print('val_model: ', val_model)

                #select random text from dataset
                text = val_dataset['texts'][indices[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE]]
                style = val_dataset['style_vectors'][indices[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE]]
                
                #style = tf.expand_dims(style, axis=0)
                seq_length = np.max(np.count_nonzero(text, axis=1))
                #print('seq_length: ', seq_length)
                timesteps = seq_length * 16
                timesteps = timesteps - (timesteps%8) + 8 
                
                _stroke = tf.random.normal([1, 400, 2])
                _text = tf.random.uniform([1, 40], dtype=tf.int32, maxval=50)
                _noise = tf.random.uniform([1, 1])
                _style_vector = tf.random.normal([1, 14, 1280])
                _ = model(_stroke, _text, _noise, _style_vector)

                imgs = utils.run_batch_inference(model, beta_set, alpha_set, text, style, tokenizer = utils.CrohmeTokenizer(), 
                                            time_steps=timesteps, diffusion_mode='new', 
                                            show_samples=False, path=None, show_every=None, return_image=True)

                for img in imgs:
                    generated_images.append(img)
                for text in text:
                    generated_texts.append(text)
                
                with train_summary_writer.as_default():
                    #print(tf.expand_dims(img, axis=0).shape)
                    
                    print('img shape: {} img avg: {}'.format(imgs[1].shape, np.mean(imgs[1])))
                    tf.summary.image("val_inference {}".format(i), tf.expand_dims(imgs[1], axis=0), step=optimizer.iterations)
                #print(img.shape)

            (avg, seq) = bttr_beam_search_prob_mean(generated_texts, generated_images, lit_model)
            print("avg: ", avg)
            print("seq: ", seq)
            #log random image and its prediction
            ind = random.choice(range(len(generated_texts)))
            (text, img) = (generated_texts[ind], generated_images[ind])
            img = cut_off_white(img)
            if not (img is None or img.shape[0] + img.shape[1] + img.shape[2] <= 3):
                tf.summary.image("val_img_gen", tf.expand_dims(img, axis=0), step=optimizer.iterations)
                
                img = ToTensor()(255 - img)
                img = img[0, :, :]
                img = torch.unsqueeze(img, 0)

                hyp = lit_model.beam_search(img)
                tf.summary.text("val_img_pred", str(hyp))

            images1 = scale_images(generated_images, (299,299,3))

            images2 = scale_images(val_dataset['samples'], (299,299,3))
            
            sub_dataset = np.random.choice(np.array_split(images2, 10))[:-1]
            fid_score = calculate_fid(val_model, images1, sub_dataset)
            print("fid_score: ", fid_score)
            
            with train_summary_writer.as_default():
                
                tf.summary.scalar('fid_score', fid_score, step=optimizer.iterations)
                tf.summary.scalar('avg bbtr pred', avg, step=optimizer.iterations)
                if(seq != float("-inf")):
                    tf.summary.scalar('seq bbtr pred', seq, step=optimizer.iterations)
                tf.summary.image("images1 {}".format(i), tf.expand_dims(images1[1], axis=0), step=optimizer.iterations)
                tf.summary.image("images2 {}".format(i), tf.expand_dims(sub_dataset[1], axis=0), step=optimizer.iterations)

            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='name of the dataset .p file', default='new.p', type=str)
    parser.add_argument('--num_valsamples', help='number of validation samples', default=0, type=int)
    parser.add_argument('--steps', help='number of trainsteps, default 60k', default=60000, type=int)
    parser.add_argument('--batchsize', help='default 96', default=96, type=int)
    parser.add_argument('--seqlen', help='sequence length during training, default 480', default=480, type=int)
    parser.add_argument('--textlen', help='text length during training, default 50', default=50, type=int)
    parser.add_argument('--width', help='offline image width, default 1400', default=1400, type=int)
    parser.add_argument('--warmup', help='number of warmup steps, default 10k', default=15000, type=int)
    parser.add_argument('--dropout', help='dropout rate, default 0', default=0.0, type=float)
    parser.add_argument('--num_attlayers', help='number of attentional layers at lowest resolution', default=2, type=int)
    parser.add_argument('--channels', help='number of channels in first layer, default 128', default=128, type=int)
    parser.add_argument('--print_every', help='show train loss every n iters', default=250, type=int)
    parser.add_argument('--save_every', help='save ckpt every n iters', default=2500, type=int)
    parser.add_argument('--diffusion_steps', help='number of diffusion steps', default=60, type=int)
    parser.add_argument('--tb_prefix', help='prefix for tensorboard logs', default=None, type=str)
    parser.add_argument('--val_every', help='how often to perform validation', default=None, type=int)
    parser.add_argument('--num_heads', help='number of attention heads for encoder', default=8, type=int)
    parser.add_argument('--enc_att_layers', help='number of attention layers for encoder', default=1, type=int)
    parser.add_argument('--noise_shedule', help='specifies which noise shedule to use (default or cosine)', default='cosine', type=str)
    parser.add_argument('--learn_sigma', help='learn cov matrix', action='store_true')
    parser.add_argument('--no-learn_sigma', dest='learn_sigma', action='store_false')
    parser.set_defaults(learn_sigma=False)
    parser.add_argument('--interpolate_alphas', help='interpolate alphas in training step', action='store_true')
    parser.add_argument('--no-interpolate_alphas', dest='interpolate_alphas', action='store_false')
    parser.set_defaults(interpolate_alphas=False)
    parser.add_argument('--pertubate_strokes', help='pertubate strokes', action='store_true')
    parser.add_argument('--no-pertubate_strokes', dest='pertubate_strokes', action='store_false')
    parser.set_defaults(pertubate_strokes=False)
    parser.add_argument('--rotate_strokes', help='rotate strokes by random angle', action='store_true')
    parser.add_argument('--no-rotate_strokes', dest='rotate_strokes', action='store_false')
    parser.set_defaults(rotate_strokes=False)
    parser.add_argument('--style_extractor', help='which style extractor to use (default mobilenet)', default='mobilenet', type=str)
    parser.add_argument('--loss_type', help='which loss function to use, possible: simple, vlb, hybrid (default simple)', default='simple', type=str)
    parser.add_argument('--importance_sampling', help='whether or not to perform importance sampling', action='store_true')
    parser.add_argument('--no-importance_sampling', dest='importance_sampling', action='store_false')
    parser.set_defaults(importance_sampling=False)
    parser.add_argument('--l0_loss', help='which loss function to use for l0, possible: nll, kl, mse (default nll)', default='nll', type=str)
    parser.add_argument('--weight_dir', help='specifies the directory to store the weights in', default='weights', type=str)
    parser.add_argument('--ignore_last_loss', help='whether or not to ignore last lossterm', action='store_true')
    parser.add_argument('--no-ignore_last_loss', dest='ignore_last_loss', action='store_false')
    parser.set_defaults(ignore_last_loss=False)
    parser.add_argument('--weight_decay', help='weight decay rate, default 0', default=0.0, type=float)

    
    args = parser.parse_args()

    DATASET = args.dataset
    NUM_VAL_SAMPLES = args.num_valsamples
    TB_PREFIX = args.tb_prefix
    NUM_STEPS = args.steps
    BATCH_SIZE = args.batchsize
    MAX_SEQ_LEN = args.seqlen
    MAX_TEXT_LEN = args.textlen
    WIDTH = args.width
    DROP_RATE = args.dropout
    NUM_ATTLAYERS = args.num_attlayers
    WARMUP_STEPS = args.warmup
    PRINT_EVERY = args.print_every
    SAVE_EVERY = args.save_every
    DIFF_STEPS = args.diffusion_steps
    VAL_EVERY = args.val_every
    ENCODER_NUM_HEADS = args.num_heads
    ENCODER_NUM_ATTLAYERS = args.enc_att_layers
    NOISE_SHEDULE = args.noise_shedule
    LEARN_SIGMA = args.learn_sigma
    INTERPOLATE_ALPHAS = args.interpolate_alphas
    PERTUBATE = args.pertubate_strokes
    ROTATE = args.rotate_strokes
    STYLE_EXTRACTOR = args.style_extractor
    LOSS_TYPE = args.loss_type
    IMPORTANCE_SAMPLING = args.importance_sampling
    L0_TYPE = args.l0_loss
    WEIGHTS_DIR = args.weight_dir
    IGNORE_LAST_LOSS = args.ignore_last_loss
    WEIGHT_DECAY = args.weight_decay

    if not os.path.isdir('./{}/'.format(WEIGHTS_DIR)):
        os.mkdir('./{}/'.format(WEIGHTS_DIR))

    with open('./{}/config.json'.format(WEIGHTS_DIR), 'w') as f:
        json.dump(vars(args), f)

    assert NOISE_SHEDULE in ['default', 'cosine']
    C1 = args.channels
    C2 = C1 * 3//2
    C3 = C1 * 2
    MAX_SEQ_LEN = MAX_SEQ_LEN - (MAX_SEQ_LEN%8) + 8

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if TB_PREFIX is None:
        train_log_dir = 'logs/diffusionwriter/{}/train'.format(current_time)
    else:
        train_log_dir = 'logs/diffusionwriter/{}_{}/train'.format(TB_PREFIX, current_time)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    BUFFER_SIZE = 3000
    L = DIFF_STEPS
    tokenizer = utils.Tokenizer()
    if NOISE_SHEDULE == 'cosine':
        beta_set = utils.get_cosine_beta_set(DIFF_STEPS)
        alpha_set = 1 - beta_set#utils.get_cosine_alpha_set(DIFF_STEPS)
    elif(NOISE_SHEDULE == 'default'):
        beta_set = utils.get_beta_set(DIFF_STEPS)
        alpha_set = tf.math.cumprod(1-beta_set)
    else:
        raise ValueError('Noise shedule not supported')

    print(beta_set)
    assert (beta_set > 0).all() and (beta_set <= 1).all()

    if STYLE_EXTRACTOR == 'mobilenet':
        style_extractor = nn.StyleExtractor()
    elif(STYLE_EXTRACTOR == 'bttr'):
        style_extractor = nn.StyleExctractor_BTTR_conv()
        style_extractor.set_model(lit_model)
    else:
        raise ValueError('Style extractor not supported')

    model = nn.DiffusionWriter(num_layers=NUM_ATTLAYERS, c1=C1, c2=C2, c3=C3, drop_rate=DROP_RATE, num_heads=ENCODER_NUM_HEADS, encoder_att_layers=ENCODER_NUM_ATTLAYERS, learn_sigma=LEARN_SIGMA, l2_reg=WEIGHT_DECAY)
    lr = nn.InvSqrtSchedule(C3, warmup_steps=WARMUP_STEPS)
    # plot lr
    if False:
        fig = plt.figure()
        a = []
        lrs = []
        for i in range(0, 70000, 1):
            #print(i, lr(float(i)).numpy())
            a.append(i)
            lrs.append(lr(float(i)).numpy())
        plt.plot(a, lrs, color='black')
        ax = plt.gca()
        ax.set_xlim([0, 70000])
        ax.set_ylim([0, 0.00025])
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Lernrate', fontsize=12)
        plt.show()
    optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, clipnorm=100)
    
    path = './data/{}'.format(DATASET)
    strokes, texts, samples, unpadded = utils.preprocess_data(path, MAX_TEXT_LEN, MAX_SEQ_LEN, WIDTH, 96, train_summary_writer)
    if NUM_VAL_SAMPLES == 0:
        dataset, style_vectors = utils.create_dataset(strokes, texts, samples, style_extractor, BATCH_SIZE, BUFFER_SIZE, NUM_VAL_SAMPLES)
        dataset_val = None
    else:
        dataset, style_vectors, dataset_val = utils.create_dataset(strokes, texts, samples, style_extractor, BATCH_SIZE, BUFFER_SIZE, NUM_VAL_SAMPLES)

    val_dataset = {'texts': texts, 'samples': unpadded, 'style_vectors': style_vectors}
    train(dataset, NUM_STEPS, model, optimizer, alpha_set, beta_set, DIFF_STEPS, PRINT_EVERY, SAVE_EVERY, INTERPOLATE_ALPHAS, train_summary_writer, val_every=VAL_EVERY, val_dataset=val_dataset, dataset_val=dataset_val, pertubate=PERTUBATE, rotate=ROTATE, loss_type=LOSS_TYPE, importance_sampling=IMPORTANCE_SAMPLING, l0_type=L0_TYPE, weights_dir=WEIGHTS_DIR, ignore_last_loss=IGNORE_LAST_LOSS)

if __name__ == '__main__':
    main()
