from random import random
from random import randint
import sys
import os
import math
import argparse
import pickle

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import nn
import utils

sys.path.append("./BTTRcustom/")
from bttr.lit_bttr import LitBTTR
ckpt = './BTTRcustom/checkpoints/pretrained-2014.ckpt'
lit_model = LitBTTR.load_from_checkpoint(ckpt)

def normalize_strokes(strokes):
    strokes = strokes.copy()
    min_x, max_x, min_y, max_y = utils.get_stroke_bounds(strokes)

    for point in strokes:
        point[0] = (point[0] - min_x) / (max_x - min_x)
        point[1] = (point[1] - min_y) / (max_y - min_y)
    return strokes

def plot_from_strokes(strokes, file):
    strokes = strokes.copy()
    strokes = strokes[0]
    
    positions = np.cumsum(strokes[:,0:2], axis=0)

        
    min_x = min(positions[:,0])
    max_x = max(positions[:,0])
    min_y = min(positions[:,1])
    max_y = max(positions[:,1])
    aspect_ratio = (max_x - min_x) / (max_y - min_y)

    #print('aspect_ratio', aspect_ratio)
    fig = plt.figure(figsize=(aspect_ratio*1,1))
    ax = fig.add_subplot(111)

    plt.axis('off')

    drawn = 0
    previous_point = [0, 0, 0]
    for i, point in enumerate(positions):
        #print('strokes[i,2]', strokes[i,2])
        if strokes[i,2] < 0.5:
            x_values = [previous_point[0], point[0]]
            y_values = [previous_point[1], point[1]]
            plt.plot(x_values, y_values, color='black' ,linewidth=2)
            drawn += 1
        previous_point = point

    plt.axis('equal')

    fig.savefig(file, bbox_inches='tight', transparent=False, pad_inches=0)
    plt.close()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batchsize', help='default 1', default=1, type=int)
    parser.add_argument('--dropout', help='dropout rate, default 0', default=0.0, type=float)
    parser.add_argument('--num_attlayers', help='number of attentional layers at lowest resolution', default=2, type=int)
    parser.add_argument('--channels', help='number of channels in first layer, default 128', default=128, type=int)
    parser.add_argument('--diffusion_steps', help='number of diffusion steps', default=60, type=int)
    parser.add_argument('--num_heads', help='number of attention heads for encoder', default=8, type=int)
    parser.add_argument('--enc_att_layers', help='number of attention layers for encoder', default=1, type=int)
    parser.add_argument('--noise_shedule', help='specifies which noise shedule to use (default or cosine)', default='cosine', type=str)
    parser.add_argument('--val_nsamples', help='Number of images to generate in total', default=2048, type=int)
    parser.add_argument('--output_path', help='directory of the output path to create', default='default', type=str)
    parser.add_argument('--val_path', help='name of the validation dataset to use', default='./data/val_dataset.p', type=str)
    parser.add_argument('--weight_file', help='name of the weight file to use', default='model_step70000.h5', type=str)
    parser.add_argument('--weight_dir', help='directory of the weights', default='weights', type=str)
    parser.add_argument('--style_extractor', help='which style extractor to use (default mobilenet)', default='mobilenet', type=str)
    parser.add_argument('--from_bound', help='percentage upper bound of samples to generate', default=0.0, type=float)
    parser.add_argument('--to_bound', help='percentage lower bound of samples to generate', default=1.0, type=float)

    args = parser.parse_args()

    BATCH_SIZE = args.batchsize
    DROP_RATE = args.dropout
    NUM_ATTLAYERS = args.num_attlayers
    CHANNELS = args.channels
    DIFF_STEPS = args.diffusion_steps
    ENCODER_NUM_HEADS = args.num_heads
    ENCODER_NUM_ATTLAYERS = args.enc_att_layers
    NOISE_SHEDULE = args.noise_shedule
    VAL_NSAMPLES = args.val_nsamples
    OUTPUT_PATH = args.output_path
    VAL_PATH = args.val_path
    STYLE_EXTRACTOR = args.style_extractor
    BOUND_FROM = args.from_bound
    BOUND_TO = args.to_bound
    WEIGHT_DIR = args.weight_dir
    
    WEIGHT_FILE = args.weight_file
    WEIGHT_FILE = "./{}/".format(WEIGHT_DIR) + WEIGHT_FILE

    if OUTPUT_PATH is None:
        print('Please specify an output path')
        return
    OUTPUT_PATH = "./output/{}_{}".format(OUTPUT_PATH, WEIGHT_DIR)
    if not os.path.isdir(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    VAL_NSAMPLES -= (VAL_NSAMPLES % BATCH_SIZE)

    C1 = CHANNELS
    C2 = C1 * 3//2
    C3 = C1 * 2

    if STYLE_EXTRACTOR == 'mobilenet':
        style_extractor = nn.StyleExtractor()
    elif STYLE_EXTRACTOR == 'bttr':
        style_extractor = nn.StyleExctractor_BTTR_conv()
        style_extractor.set_model(lit_model)
    else:
        print('Please specify a valid style extractor')
        return

    print('using noise shedule: {}'.format(NOISE_SHEDULE))
    if NOISE_SHEDULE == 'default':
        beta_set = utils.get_beta_set(DIFF_STEPS)
        alpha_set = tf.math.cumprod(1-beta_set)
    elif NOISE_SHEDULE == 'cosine':
        beta_set = utils.get_cosine_beta_set(DIFF_STEPS)
        alpha_set = 1 - beta_set#utils.get_cosine_alpha_set(DIFF_STEPS)
        alpha_set_bar = tf.math.cumprod(alpha_set)
    else:
        print('Noise shedule not found')
        return
    tokenizer = utils.CrohmeTokenizer()

    strokes, texts, samples, unpadded = utils.preprocess_data(VAL_PATH, 26, 480, 1400, 96)

    style_vectors = []
    for sample in samples:
        sample = tf.expand_dims(sample, axis=0)
        
        style_vec = style_extractor(sample)
        style_vec = style_vec.numpy()
        style_vectors.append(style_vec)
    
    style_vecs = np.concatenate(style_vectors, axis=0)
    
    model = nn.DiffusionWriter(num_layers=NUM_ATTLAYERS, c1=C1, c2=C2, c3=C3, drop_rate=DROP_RATE, num_heads=ENCODER_NUM_HEADS, encoder_att_layers=ENCODER_NUM_ATTLAYERS)

    _stroke = tf.random.normal([32, 400, 2])
    _text = tf.random.uniform([32, 40], dtype=tf.int32, maxval=50)
    _noise = tf.random.uniform([32, 1, 1])
    if STYLE_EXTRACTOR == 'mobilenet':
        _style_vector = tf.random.normal([32, 14, 1280])
    elif STYLE_EXTRACTOR == 'bttr':
        _style_vector = tf.random.normal([32, 60, 256])
    _ = model(_stroke, _text, _noise, _style_vector)

    model.load_weights(WEIGHT_FILE)

    indices = np.repeat(np.arange(len(samples)), math.ceil(VAL_NSAMPLES / len(samples)))
    offset = math.ceil(indices.shape[0] * BOUND_FROM)
    offset_to = math.ceil(indices.shape[0] * BOUND_TO)
    print('offset: {}, offset_to: {}'.format(offset, offset_to))
    indices = indices[offset:offset_to]
    print((offset_to - offset) / BATCH_SIZE)
    batches = np.array_split(indices, math.ceil((offset_to - offset) / BATCH_SIZE))
    j = offset

    with open('./mean_style.pkl', 'rb') as f:
        batch_style = pickle.load(f)
    batch_style = [batch_style]

    for batch in batches:
        _stroke = tf.random.normal([32, 400, 2])
        _text = tf.random.uniform([32, 40], dtype=tf.int32, maxval=50)
        _noise = tf.random.uniform([32, 1, 1])
        if STYLE_EXTRACTOR == 'mobilenet':
            _style_vector = tf.random.normal([32, 14, 1280])
        elif STYLE_EXTRACTOR == 'bttr':
            _style_vector = tf.random.normal([32, 60, 256])
        _ = model(_stroke, _text, _noise, _style_vector)
        #model.load_weights(WEIGHT_FILE)

        print(j, batch)

        batch_texts = texts[batch]
        #batch_style = [style_vecs[randint(0, len(style_vecs)-1)]]

        seq_length = np.max(np.count_nonzero(batch_texts, axis=1))
        timesteps = seq_length * 16
        timesteps = timesteps - (timesteps%8) + 8

        print(batch_texts)
        strokes, imgs = utils.run_batch_inference(model, beta_set, alpha_set_bar, batch_texts, batch_style, 
                                    tokenizer=tokenizer, time_steps=timesteps, diffusion_mode='new', 
                                    show_samples=False, path=None, show_every=None, return_both=True)


        # plot images
        for i in range(len(imgs)):
            #plt.imshow(imgs[i])
            #plt.show()
            #plt.axis('off')
            #plt.savefig(OUTPUT_PATH + '/{}.png'.format(j+i), bbox_inches='tight', pad_inches=0)
            print(OUTPUT_PATH + '/{}.png'.format(j+i))
            plot_from_strokes(strokes, OUTPUT_PATH + '/{}.png'.format(j+i))
            #plt.close()

        j += len(batch)
    return
    for batch_n in range(VAL_NSAMPLES // BATCH_SIZE):

        batchlb = batch_n * BATCH_SIZE
        batchup = batch_n * BATCH_SIZE + BATCH_SIZE
        batch_indices = indices[batchlb:batchup]
        batch_texts = texts[batch_indices]
        batch_style = style_vecs[batch_indices]

        seq_length = np.max(np.count_nonzero(batch_texts, axis=1))
        timesteps = seq_length * 16
        timesteps = timesteps - (timesteps%8) + 8

        print(batch_texts)
        imgs = utils.run_batch_inference(model, beta_set, alpha_set, batch_texts, batch_style, 
                                    tokenizer=tokenizer, time_steps=timesteps, diffusion_mode='new', 
                                    show_samples=False, path=None, show_every=None, return_image=True)

        # plot images
        for i in range(len(imgs)):
            plt.imshow(imgs[i])
            plt.savefig(OUTPUT_PATH + '/{}.png'.format(batchlb+i))
            print(OUTPUT_PATH + '/{}.png'.format(batchlb+i))
            plt.close()


if __name__ == "__main__":
    main()