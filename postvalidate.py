import sys
import argparse
import os
import re
import time

import tensorflow as tf
import numpy as np

import nn
import utils
import preprocessing

sys.path.append("./BTTRcustom/")
from bttr.lit_bttr import LitBTTR
ckpt = './BTTRcustom/checkpoints/pretrained-2014.ckpt'
lit_model = LitBTTR.load_from_checkpoint(ckpt)

def main():
    parser = argparse.ArgumentParser()

    #args.add_argument('--steps', help='number of trainsteps, default 60k', default=60000, type=int)
    #args.add_argument('--batchsize', help='default 96', default=96, type=int)
    #args.add_argument('--seqlen', help='sequence length during training, default 480', default=480, type=int)
    #args.add_argument('--textlen', help='text length during training, default 50', default=50, type=int)
    #args.add_argument('--width', help='offline image width, default 1400', default=1400, type=int)
    #args.add_argument('--warmup', help='number of warmup steps, default 10k', default=10000, type=int)
    parser.add_argument('--dropout', help='dropout rate, default 0', default=0.0, type=float)
    parser.add_argument('--num_attlayers', help='number of attentional layers at lowest resolution', default=2, type=int)
    parser.add_argument('--channels', help='number of channels in first layer, default 128', default=128, type=int)
    #args.add_argument('--print_every', help='show train loss every n iters', default=1000, type=int)
    #args.add_argument('--save_every', help='save ckpt every n iters', default=10000, type=int)
    parser.add_argument('--diffusion_steps', help='number of diffusion steps', default=60, type=int)
    #args.add_argument('--tb_prefix', help='prefix for tensorboard logs', default=None, type=str)
    #args.add_argument('--val_every', help='how often to perform validation', default=None, type=int)
    parser.add_argument('--num_heads', help='number of attention heads for encoder', default=8, type=int)
    parser.add_argument('--enc_att_layers', help='number of attention layers for encoder', default=1, type=int)
    parser.add_argument('--noise_shedule', help='specifies which noise shedule to use (default or cosine)', default='default', type=str)
    parser.add_argument('--val_nsamples', help='Number of images to generate each iteration', default=320, type=int)

    args = parser.parse_args()
    #TB_PREFIX = args.tb_prefix
    #NUM_STEPS = args.steps
    #BATCH_SIZE = args.batchsize
    #MAX_SEQ_LEN = args.seqlen
    #MAX_TEXT_LEN = args.textlen
    #WIDTH = args.width
    DROP_RATE = args.dropout
    NUM_ATTLAYERS = args.num_attlayers
    #WARMUP_STEPS = args.warmup
    #PRINT_EVERY = args.print_every
    #SAVE_EVERY = args.save_every
    DIFF_STEPS = args.diffusion_steps
    #VAL_EVERY = args.val_every
    ENCODER_NUM_HEADS = args.num_heads
    ENCODER_NUM_ATTLAYERS = args.enc_att_layers
    NOISE_SHEDULE = args.noise_shedule
    VAL_NSAMPLES = args.val_nsamples
    print(lit_model)

    C1 = args.channels
    C2 = C1 * 3//2
    C3 = C1 * 2

    style_extractor = nn.StyleExtractor()
    model = nn.DiffusionWriter(num_layers=NUM_ATTLAYERS, c1=C1, c2=C2, c3=C3, drop_rate=DROP_RATE, num_heads=ENCODER_NUM_HEADS, encoder_att_layers=ENCODER_NUM_ATTLAYERS)

    _stroke = tf.random.normal([1, 400, 2])
    _text = tf.random.uniform([1, 40], dtype=tf.int32, maxval=50)
    _noise = tf.random.uniform([1, 1])
    _style_vector = tf.random.normal([1, 14, 1280])
    _ = model(_stroke, _text, _noise, _style_vector)

    beta_set = utils.get_beta_set(DIFF_STEPS)
    alpha_set = tf.math.cumprod(1-beta_set)
    tokenizer = utils.CrohmeTokenizer()

    path = './data/crohme_strokes.p'
    strokes, texts, samples, unpadded = utils.preprocess_data(path, 50, 480, 1400, 96)
    strokes = strokes[:VAL_NSAMPLES]
    texts = texts[:VAL_NSAMPLES]
    samples = samples[:VAL_NSAMPLES]
    style_vectors = []
    samples = tf.data.Dataset.from_tensor_slices(samples).batch(320)
    for sample in samples:
        print(sample.shape)
        style_vec = style_extractor(sample)
        style_vec = style_vec.numpy()
        style_vectors.append(style_vec)
    print(style_vec[1].shape)

    ordered_weights = {}
    weight_files = []
    for file in os.listdir("./weights/"):
        if not re.match('model_step\d*.h5', file):
            continue
        i1 = file.index('p')
        i2 = file.index('.')
        step = int(file[i1+1:i2])
        #print(step)
        ordered_weights[step] = './weights/{}'.format(file)

    for k, v in sorted(ordered_weights.items()):
        weight_files.append(v)
    
    writer_img = tf.expand_dims(preprocessing.read_img('./assets/j07-370z-01.tif', 96), 0)
    style_vector = style_extractor(writer_img)
    for weight_file in weight_files:
        print(weight_file)
        model.load_weights(weight_file)
        print(model)
        start = time.time()
        for i, text in enumerate(texts):
            seq_length = np.count_nonzero(texts[0], axis=0)
            #print('seq_length: ', seq_length)
            timesteps = seq_length * 16
            timesteps = timesteps - (timesteps%8) + 8

            #print('style_vector.shape: ', style_vector.shape)
            #print('style_vec[0].shape: ', style_vec[0].shape)
            img = utils.run_batch_inference(model, beta_set, alpha_set, texts[0], tf.expand_dims(style_vec[0], axis=0), 
                                    tokenizer=tokenizer, time_steps=timesteps, diffusion_mode='new', 
                                    show_samples=False, path=None, show_every=None, return_image=True)
            print('{}/{}'.format(i, VAL_NSAMPLES))
        end = time.time()
        print('time per sample: {}'.format((end - start) / VAL_NSAMPLES))
        

if __name__ == "__main__":
    main()