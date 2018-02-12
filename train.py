### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
from collections import OrderedDict
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import os
import numpy as np
import torch
from torch.autograd import Variable


# ----- Class to manage cosine rate decay -----
# Currently only handles new cycles from epoch boundaries
class CosLearningRateDecay():
    # Note: stateless after initialization
    def __init__(self, start_epoch, stop_epoch, iters_per_epoch, max_lr, min_lr=0.0):
        self.start_epoch = start_epoch
#        self.start_iter = start_iter

        self.max_lr = max_lr
        self.min_lr = min_lr

        self.start_iters = start_epoch * iters_per_epoch

        # spill one
        full_epochs_remaining = stop_epoch - start_epoch# + 1
        #this_epoch_iters_remaining = iters_per_epoch - start_iter

        #total number of iterations to be completed this cosine cycle
        self.total_iters_this_cycle = full_epochs_remaining * iters_per_epoch #+ (this_epoch_iters_remaining - stop_iter)
    
    """
    params:
        total_iter: total number of iterations completed since start

    formula: 
    lr_min + 0.5*(lr_max - lr_min) * (1 + cos(pi*current_total_iter/target_iter))
    """
    def get_lr(self, total_iter):
        iters_elapsed = total_iter - self.start_iters
        learning_rate = self.min_lr
        learning_rate += 0.5*(self.max_lr - self.min_lr) * (1 + np.cos(np.pi * (iters_elapsed / self.total_iters_this_cycle)))
        return learning_rate

# ----- end CosLearningRateDecay class -----


opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
else:    
    start_epoch, epoch_iter = 1, 0

if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)


total_steps = (start_epoch-1) * dataset_size + epoch_iter    


if opt.cos_decay:
    print("Using cosine decay for one lr => lr = 0 cycle")
    print("Using niter_decay as period of cycle (single lr => 0 cycle)")
    if opt.started_epoch and opt.started_epoch != start_epoch:
        # definitely NOT starting a new cosine cycle -- continuing
        print("Calculating LR from starting epoch: ", opt.started_epoch)
        

        # total_epochs_since_cycle_start = start_epoch - opt.started_epoch
        # iter_this_epoch = epoch_iter
        # total_iters_since_cycle_start = total_epochs_since_cycle_start * dataset_size  + epoch_iter
        # total_iters_this_cycle = opt.niter_decay * dataset_size
        
        # print("Continuing with lr = ", learning_rate)
        # opt.lr = learning_rate
    else:
        print("Starting new cosine learning rate decay")
    cos_decay = CosLearningRateDecay(start_epoch=opt.started_epoch, 
                                     stop_epoch=opt.niter_decay, 
                                     iters_per_epoch=dataset_size, 
                                     max_lr=opt.lr
                                    )
    lr = cos_decay.get_lr(total_steps)
    print("Initial Learning Rate = ", lr)
    opt.lr = lr




model = create_model(opt)
visualizer = Visualizer(opt)

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(dataset, start=epoch_iter):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == 0

        ############## Forward Pass ######################
        losses, generated = model(Variable(data['label']), Variable(data['inst']), 
            Variable(data['image']), Variable(data['feat']), infer=save_fake)

        # sum per device losses
        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        loss_dict = dict(zip(model.module.loss_names, losses))

        # calculate final loss scalar
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = loss_dict['G_GAN'] + loss_dict['G_GAN_Feat'] + loss_dict['G_VGG']

        ############### Backward Pass ####################
        # update generator weights
        model.module.optimizer_G.zero_grad()
        loss_G.backward()
        model.module.optimizer_G.step()

        # update discriminator weights
        model.module.optimizer_D.zero_grad()
        loss_D.backward()
        model.module.optimizer_D.step()

        #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]) 

        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == 0:
            errors = {k: v.data[0] if not isinstance(v, int) else v for k, v in loss_dict.items()}
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)

        ### display output images
        if save_fake:
            visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                                   ('synthesized_image', util.tensor2im(generated.data[0])),
                                   ('real_image', util.tensor2im(data['image'][0]))])
            visualizer.display_current_results(visuals, epoch, total_steps)

        ### save latest model
        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.module.save('latest')            
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
       
        
        if opt.cos_decay and total_steps % opt.cos_decay_update_iters == 0:
            new_lr = cos_decay.get_lr(total_steps)
            model.module.update_learning_rate(override_lr=new_lr)


    # end of epoch 
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
        model.module.save('latest')
        model.module.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        model.module.update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter and not opt.cos_decay:
        model.module.update_learning_rate()
