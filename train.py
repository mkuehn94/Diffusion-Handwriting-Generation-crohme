from xmlrpc.client import Boolean
import tensorflow as tf
import torch
import numpy as np
import matplotlib.pyplot as plt
import utils
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

from validation import bttr_beam_search_prob
from validation import bttr_beam_search_prob_mean, cut_off_white

import sys
sys.path.append("./BTTRcustom/")
from bttr.lit_bttr import LitBTTR

SIGMA_LOSS_COEF = 0.001

@tf.function
def train_step(x, pen_lifts, text, style_vectors, interpolate_alphas, glob_args):
    model, alpha_set, beta_set, bce, train_loss, optimizer, train_summary_writer = glob_args
    if(interpolate_alphas):
        alphas, timesteps = utils.get_alphas(len(x), alpha_set)
    else:
        alphas, timesteps = utils.get_alphas_new(len(x), alpha_set)
    eps = tf.random.normal(tf.shape(x))
    x_perturbed = tf.sqrt(alphas) * x 
    x_perturbed += tf.sqrt(1 - alphas) * eps
    
    with tf.GradientTape() as tape:
        if model.learn_sigma:
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
        else:
            score, pl_pred, att = model(x_perturbed, text, tf.sqrt(alphas), style_vectors, training=True)
        loss = nn.loss_fn(eps, score, pen_lifts, pl_pred, alphas, bce)
        if model.learn_sigma:
            loss += SIGMA_LOSS_COEF * nn.sigma_los_vb(x_perturbed, x, timesteps, alphas, betas, alpha_set, alpha_set_prev, beta_set, beta_bars, score, sigma, train_summary_writer, step=optimizer.iterations)
        
    gradients = tape.gradient(loss, model.trainable_variables)  
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    return score, att


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

@tf.function
def validation_step(x, pen_lifts, text, style_vectors, interpolate_alphas, glob_args):
    model, alpha_set, beta_set, bce, train_loss, optimizer, train_summary_writer = glob_args
    if(interpolate_alphas):
        alphas, timesteps = utils.get_alphas(len(x), alpha_set)
    else:
        alphas, timesteps = utils.get_alphas_new(len(x), alpha_set)
    eps = tf.random.normal(tf.shape(x))
    x_perturbed = tf.sqrt(alphas) * x 
    x_perturbed += tf.sqrt(1 - alphas) * eps
    
    #with tf.GradientTape() as tape:
    if model.learn_sigma:
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
    else:
        score, pl_pred, att = model(x_perturbed, text, tf.sqrt(alphas), style_vectors, training=True)
    loss = nn.loss_fn(eps, score, pen_lifts, pl_pred, alphas, bce)
    if model.learn_sigma:
        loss += SIGMA_LOSS_COEF * nn.sigma_los_vb(x_perturbed, x, timesteps, alphas, betas, alpha_set, alpha_set_prev, beta_set, beta_bars, score, sigma, train_summary_writer, step=optimizer.iterations)
        
    return loss

val_model = InceptionV3()
ckpt = './BTTRcustom/checkpoints/pretrained-2014.ckpt'
lit_model = LitBTTR.load_from_checkpoint(ckpt)
def train(dataset, iterations, model, optimizer, alpha_set, beta_set, DIFF_STEPS, print_every=1000, save_every=10000, interpolate_alphas=True, train_summary_writer = None, val_every = None, val_dataset = None, dataset_val = None):
    assert DIFF_STEPS == len(alpha_set) == len(beta_set)
    s = time.time()
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    train_loss = tf.keras.metrics.Mean()
    val_loss = tf.keras.metrics.Mean()
    for count, (strokes, text, style_vectors) in enumerate(dataset.repeat(5000)):
        strokes, pen_lifts = strokes[:, :, :2], strokes[:, :, 2:]
        glob_args = model, alpha_set, beta_set, bce, train_loss, optimizer, train_summary_writer
        model_out, att = train_step(strokes, pen_lifts, text, style_vectors, interpolate_alphas, glob_args)
        
        
        if optimizer.iterations%print_every==0:
            print("Iteration %d, Loss %f, Time %ds" % (optimizer.iterations, train_loss.result(), time.time()-s))
            if dataset_val is not None:
                val_loss = validation_step(strokes, pen_lifts, text, style_vectors, interpolate_alphas, glob_args)
                tf.print("val_loss:", val_loss, output_stream=sys.stdout)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=optimizer.iterations)
                if dataset_val is not None:
                    tf.summary.scalar('val_loss', val_loss, step=optimizer.iterations)

            train_loss.reset_states()

        if (optimizer.iterations+1) % save_every==0:
            save_path = './weights/model_step%d.h5' % (optimizer.iterations+1)
            model.save_weights(save_path)
            
        if optimizer.iterations > iterations:
            model.save_weights('./weights/model.h5')
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
    parser.add_argument('--dataset', help='name of the dataset .p file', default='shuffled.p', type=str)
    parser.add_argument('--num_valsamples', help='number of validation samples', default=320, type=int)
    parser.add_argument('--steps', help='number of trainsteps, default 60k', default=60000, type=int)
    parser.add_argument('--batchsize', help='default 96', default=96, type=int)
    parser.add_argument('--seqlen', help='sequence length during training, default 480', default=480, type=int)
    parser.add_argument('--textlen', help='text length during training, default 50', default=50, type=int)
    parser.add_argument('--width', help='offline image width, default 1400', default=1400, type=int)
    parser.add_argument('--warmup', help='number of warmup steps, default 10k', default=10000, type=int)
    parser.add_argument('--dropout', help='dropout rate, default 0', default=0.0, type=float)
    parser.add_argument('--num_attlayers', help='number of attentional layers at lowest resolution', default=2, type=int)
    parser.add_argument('--channels', help='number of channels in first layer, default 128', default=128, type=int)
    parser.add_argument('--print_every', help='show train loss every n iters', default=1000, type=int)
    parser.add_argument('--save_every', help='save ckpt every n iters', default=10000, type=int)
    parser.add_argument('--diffusion_steps', help='number of diffusion steps', default=60, type=int)
    parser.add_argument('--tb_prefix', help='prefix for tensorboard logs', default=None, type=str)
    parser.add_argument('--val_every', help='how often to perform validation', default=None, type=int)
    parser.add_argument('--num_heads', help='number of attention heads for encoder', default=8, type=int)
    parser.add_argument('--enc_att_layers', help='number of attention layers for encoder', default=1, type=int)
    parser.add_argument('--noise_shedule', help='specifies which noise shedule to use (default or cosine)', default='default', type=str)
    parser.add_argument('--learn_sigma', help='learn cov matrix', default=False, type=Boolean)
    parser.add_argument('--interpolate_alphas', help='interpolate alphas in training step', default=True, type=Boolean)

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
        alpha_set = utils.get_cosine_alpha_set(DIFF_STEPS)
    elif(NOISE_SHEDULE == 'default'):
        beta_set = utils.get_beta_set(DIFF_STEPS)
        alpha_set = tf.math.cumprod(1-beta_set)

    style_extractor = nn.StyleExtractor()
    model = nn.DiffusionWriter(num_layers=NUM_ATTLAYERS, c1=C1, c2=C2, c3=C3, drop_rate=DROP_RATE, num_heads=ENCODER_NUM_HEADS, encoder_att_layers=ENCODER_NUM_ATTLAYERS, learn_sigma=LEARN_SIGMA)
    lr = nn.InvSqrtSchedule(C3, warmup_steps=WARMUP_STEPS)
    # plot lr
    if False:
        a = []
        lrs = []
        for i in range(0, 60000, 100):
            print(i, lr(float(i)).numpy())
            a.append(i)
            lrs.append(lr(float(i)).numpy())
        plt.plot(a, lrs)
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
    train(dataset, NUM_STEPS, model, optimizer, alpha_set, beta_set, DIFF_STEPS, PRINT_EVERY, SAVE_EVERY, INTERPOLATE_ALPHAS, train_summary_writer, val_every=VAL_EVERY, val_dataset=val_dataset, dataset_val=dataset_val)

if __name__ == '__main__':
    main()
