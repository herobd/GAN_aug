"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import numpy as np

def diff_images(imA,imB):
    if imA.shape[3]<imB.shape[3]:
        imT=imA
        imA=imB
        imB=imT
    diff_w = imA.shape[3]-imB.shape[3]
    offF = diff_w//2
    offB = diff_w//2 + diff_w%2
    if offB>0:
        imA = imA[...,offF:-offB]
    else:
        imA = imA[...,offF:]
    diff = (imA-imB).abs().mean().item()
    return diff

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    aux = model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = opt.load_iter                # the total number of training iterations
    start_epoch = opt.epoch_count
    if aux is not None:
        total_iters = aux['iteration']
        start_epoch = aux['epoch']
        model.set_optimizer_states(aux['optimizers'])
        model.set_scheduler_states(aux['schedulers'])
    if opt.dataset_mode == 'synthetic' and not opt.no_weight_fonts:
        if aux is not None:
            for i,prob in enumerate(aux['font_prob']):
                dataset.dataset.textGen[i].fontProbs=prob
            B_diff_EMA = aux['B_diff_EMA']
            score_fake_EMA = aux['score_fake_EMA']
            agree_score_diff = aux['agree_score_diff']
        else:
            B_diff_EMA = []
            score_fake_EMA = []
            agree_score_diff = []
        alpha=0.01


    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            print('iteration: {}'.format(total_iters), end='\r')
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += 1#opt.batch_size
            epoch_iter += 1#opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if opt.dataset_mode == 'synthetic' and not opt.no_weight_fonts:
                if total_iters>230:
                    #update exponential moving average
                    B_diff = diff_images(model.fake_A,model.real_B)
                    agree_score_diff.append(B_diff<B_diff_EMA == model.score_fake>score_fake_EMA)
                    #if B_diff<B_diff_EMA:
                    if ( (model.score_fake>score_fake_EMA and not opt.weight_fonts_with_id) or
                            (B_diff<B_diff_EMA and opt.weight_fonts_with_id) ):
                        dataset.dataset.textGen[data['B_gen']].changeFontProb(data['B_font'],opt.weight_font_step)
                    #else:
                    #    dataset.dataset.textGen[data['B_gen']].changeFontProb(data['B_font'],-2)
                    B_diff_EMA = alpha*B_diff+(1-alpha)*B_diff_EMA
                    score_fake_EMA = alpha*model.score_fake+(1-alpha)*score_fake_EMA
                elif total_iters>200:
                    B_diff = diff_images(model.fake_A,model.real_B)
                    B_diff_EMA.append(B_diff)
                    score_fake_EMA.append(model.score_fake)
                    if total_iters==230:
                        B_diff_EMA = np.mean(B_diff_EMA) #get initial average
                        score_fake_EMA = np.mean(score_fake_EMA) #get initial average


            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter and (total_iters%opt.save_by_iter_at==0) else 'latest'
                aux={   'iteration':total_iters,
                        'epoch': epoch,
                        'optimizers': model.get_optimizer_states(),
                        'schedulers': model.get_scheduler_states(),
                    }
                if opt.dataset_mode == 'synthetic' and not opt.no_weight_fonts:
                    aux['font_prob'] = [s.fontProbs for s in dataset.dataset.textGen]
                    aux['score_fake_EMA'] = score_fake_EMA
                    aux['B_diff_EMA'] = B_diff_EMA
                    aux['agree_score_diff'] = agree_score_diff
                model.save_networks(save_suffix,aux)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
