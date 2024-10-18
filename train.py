from pathlib import Path
import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
import pdb
import logging

# from PIL import Image, ImageOps, ImageFilter
# from dice_loss_new import DiceLoss
from monai.losses import DiceLoss
from torch import nn, optim
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from seq_KD_algo_Loss_BTF import Seq_KD_algo_loss_BTF
from eval3d_seq_KD_nnUNet_revised import eval_net
from nnUNet_custom import DynUNet

from torch.utils.tensorboard import SummaryWriter
import torchvision

import numpy as np

# from utils.customdataset3d_seq import TrainCustomDataset3D, ValCustomDataset3D
from utils.customdataset3d_BT import TrainCustomDataset3D, ValCustomDataset3D
from utils.tensorboard_img_ck import make_image_grid, tb_image_grid, make_grid
from utils.postprocessing import OnehotEncoding3D, OnehotNcombine3D, Combine3D
from utils.set_requires_grad import set_requires_grad

parser = argparse.ArgumentParser(description='Seq KD Training')
# parser.add_argument('data', type=Path, metavar='DIR',
#                     help='path to dataset')

parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('-e', '--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N',
                    help='mini-batch size')
# parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
#                     help='base learning rate for weights')
parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-3,
                        help='Learning rate')
# parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
#                     help='base learning rate for biases and batch norm parameters')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',         # Barlow Twins loss의 두번째 term에 곱해지는 weight
                    help='weight on off-diagonal terms')
# parser.add_argument('--projector', default='8192-8192-8192', type=str,
#                     metavar='MLP', help='projector MLP')
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint-dir', default='./checkpoint_KnowledgeDistillation_research/ckpt_ablation_study_seq_loss_BTF_2ch_nnUnet_flair_t1ce_btl_10dl_cel_20220718', type=Path,
                    metavar='DIR', help='path to checkpoint directory')




#-----------------------------------------------------------------------------------------------------------------------

def cleanup():
    dist.destroy_process_group()

#-----------------------------------------------------------------------------------------------------------------------



#-----------------------------------------------------------------------------------------------------------------------

def main():

    os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'
    args = parser.parse_args()
    args.ngpus = torch.cuda.device_count()

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12358'

    # single-node distributed training
    args.rank = 0
    args.world_size = args.ngpus

    
    logging.info(f'The # of GPUs we\'re using is {args.ngpus}')
    logging.info(f'''Starting training:
        Epochs:          {args.epochs}
        Batch size:      {args.batch_size}
        Learning rate:   {args.learning_rate}
        Training size:   {300}
        Device:          {os.environ['CUDA_VISIBLE_DEVICES']}
    ''')

    mp.spawn( main_worker, args=(args,), nprocs=args.ngpus, join=True )


def main_worker( gpu, args ):
    """
    main worker안에
    Distributed Data Parallel (DDP)
    set up부터 다 되어있다.
    
    """
    torch.manual_seed(0)  # trianing시에 Dataloader로 trainDataset의 index를 shuffle 할 때
                          # random으로 섞기 때문에 같은 performance를 유지하기 위해선 torch.manual_seed()가 필요하다.
    
    
    tb_comment = f'_Ablation_loss_BTF_Seq_KD_rev_2ch_nnUNet_flair_t1ce_bt1_10dl_cel_StepLR_20_w_BG_1000_patient_crop_128x128x128'  
    
    args.rank += gpu
    # dist.init_process_group(
    #     backend='nccl', init_method=args.dist_url,
    #     world_size=args.world_size, rank=args.rank)

    # initialize the process group
    dist.init_process_group( backend='nccl', world_size=args.world_size, rank=args.rank )

    # training 시작하자마자 checkpoint 만듦
    if args.rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.checkpoint_dir / 'seq_kd_stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)

    # torch.cuda.set_device(gpu)                   # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'으로 사용가능한 gpu 환경변수를 선언했기 때문에 안써도 됨
    torch.backends.cudnn.benchmark = True

    # setup teacher and student model and devices for this process    
    model_t = DynUNet( spatial_dims=3,
                       in_channels=4,
                       out_channels=4,
                       deep_supervision=True ).cuda(gpu)
    model_t = nn.SyncBatchNorm.convert_sync_batchnorm(model_t)
    model_t = DDP( model_t, device_ids=[gpu] )

    model_s = Seq_KD_algo_loss_BTF( args ).cuda(gpu) # 그냥 model 선언한 것
    model_s = nn.SyncBatchNorm.convert_sync_batchnorm(model_s)
    model_s = DDP( model_s, device_ids=[gpu] )

    
    
    # for training 4ch backboneA --------------------------------------------------------------------------#
    dice_t = DiceLoss( include_background=True, to_onehot_y=False, softmax=True, smooth_nr=1e-05, smooth_dr=1e-05 )
    optimizer_t = optim.Adam( model_t.parameters(), lr=args.learning_rate, weight_decay=1e-5 )
    scheduler_t = optim.lr_scheduler.StepLR( optimizer_t, step_size=20, gamma=0.1 )    
    #------------------------------------------------------------------------------------------------------#

    # for training 2ch backboneB --------------------------------------------------------------------------#
    optimizer_s = optim.Adam( model_s.parameters(), lr=args.learning_rate, weight_decay=1e-5 )
    scheduler_s = optim.lr_scheduler.StepLR( optimizer_s, step_size=20, gamma=0.1 )   


    # automatically resume from checkpoint if it exists
    nameofModelinfo = 'Ablation_loss_BTF_Seq_KD_rev_2ch_nnUNet_flair_t1ce_bt1_10dl_cel_StepLR_20_w_BG_1000_patient_crop_128x128x128_total_ckpt.pth'
    
    if (args.checkpoint_dir / nameofModelinfo / 'a' ).is_file():
        ckpt = torch.load(args.checkpoint_dir / nameofModelinfo / 'a',
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        model_t.load_state_dict(ckpt['model_t'])
        model_s.load_state_dict(ckpt['model_s'])
        optimizer_t.load_state_dict(ckpt['optimizer_t'])
        optimizer_s.load_state_dict(ckpt['optimizer_s'])
        args.learning_rate = ckpt['lr']
    else:
        start_epoch = 0
    #------------------------------------------------------------------------------------------------------#
    
    # Data sorting process
    dir_learning = 'data/input_BraTS21_4ch_1000_251/learning/'
    dir_testing = 'data/input_BraTS21_4ch_1000_251/testing/'
    args.train_percent = 1.0

    patient_lists = os.listdir( dir_learning )
    patient_lists.sort()
    patient_lists = [ x for x in patient_lists if x.endswith('s') == False ]

    train_patient_lists = random.sample( patient_lists, int(len(patient_lists) * args.train_percent) )
    train_patient_lists.sort()

    val_patient_lists = os.listdir( dir_testing )
    val_patient_lists.sort()
    val_patient_lists = [ x for x in val_patient_lists if x.endswith('s') == False ]

    # pdb.set_trace()

    # Remove overlapped patient in train_patient list
    # for nb in train_patient_lists:
    #     val_patient_lists.remove( nb )

    trainDataset = TrainCustomDataset3D( dir_learning, train_patient_lists[:4], img1='flair', img2='t1ce' )
    valDataset   = ValCustomDataset3D( dir_testing, val_patient_lists[:2], img1='flair', img2='t1ce' )

    trainSampler = DistributedSampler( trainDataset ) # torch.utils.data.distributed.DistributedSampler는 기본 default로 Dataset의 index를 shuffle하게 되어있다.
    valSampler  = DistributedSampler( valDataset )   # torch.utils.data.distributed.DistributedSampler는 기본 default로 Dataset의 index를 shuffle하게 되어있다.
    
    assert args.batch_size % args.world_size == 0  # batch_size를 사용하는 GPU 개수로 나눴을 때 나머지가 0이 아니면 에러뜸

    per_device_batch_size = args.batch_size // args.world_size

    trainLoader = DataLoader( 
        trainDataset, batch_size=per_device_batch_size, num_workers=args.workers,
        pin_memory=True, sampler=trainSampler )

    valLoader = DataLoader( 
        valDataset, batch_size=per_device_batch_size, num_workers=args.workers,
        pin_memory=True, sampler=valSampler )

    # start_time = time.time()
    scaler  = torch.cuda.amp.GradScaler()
    n_train = len(trainDataset)
    # n_val   = len(valDataset)
    
    # Tensorboard setting
    writer = SummaryWriter( comment=tb_comment )

    # Start traing process ---------------------------------------------------------------------------
    for epoch in range( start_epoch, args.epochs ):
        
        model_t.train()
        model_s.train()

        loss_t_train = 0
        loss_s_train = 0
        loss_s_bt = 0
        loss_s_ce = 0
        """
        In distributed mode, calling the set_epoch() method at the beginning of each epoch
        before creating the DataLoader iterator is necessary to make shuffling work properly across multiple epochs.
        Otherwise, the same ordering will be always used.        
        """
        trainSampler.set_epoch( epoch )    # Dataloader iterator 들어가기 전에 set_epoch( epoch )
                                           # 선언 안해주면 Dataset의 index shuffle 안해줌        
        # running_loss_4ch = 0
        # running_loss_seq = 0        
        epoch_loss_t = 0
        epoch_loss_s = 0
        epoch_loss_s_bt = 0
        epoch_loss_s_ce = 0
               
        # Manually control on tqdm() updates by using a with statement
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{args.epochs}', unit='patient' ) as pbar:
            for step, (y1, y2) in enumerate( trainLoader, start=epoch * len(trainLoader)):
                # step == iteration

                y1['imgs']  = y1['imgs'].cuda( gpu, non_blocking=True )
                y1['masks'] = y1['masks'].cuda( gpu, non_blocking=True )
            
                y2['imgs']  = y2['imgs'].cuda( gpu, non_blocking=True )
                y2['masks'] = y2['masks'].cuda( gpu, non_blocking=True )
            
                # print('shape of y1_masks : ', y1['masks'].shape)
                # print('shape of y2_masks : ', y2['masks'].shape)

                # pdb.set_trace()

#-------------------------------------------------------------------------------------------------------------#                

                # Forward pass
                set_requires_grad( model_t, requires_grad=True )    # teacher network의 parameter에 대한 gradient를 학습시킨다는 것을 확실하게 설정 --> update teacher network parameter 
                optimizer_t.zero_grad()                
                with torch.cuda.amp.autocast():  # autocast는 forward pass에서만 사용
                    
                    # loop 안에서 일단 backboneA의 parameter 먼저 학습
                    output_t = model_t( y1['imgs'] ) 
                    
                    detached_output_t = {
                    'pred' : output_t['pred'].detach(),
                    'bottleneck_feature_map' : output_t['bottleneck_feature_map'].detach()
                    }

                    # print( output_t['pred'][:,0].shape )

                    # print( 'Before_update_4ch_pred = ', list(model.module.backboneA.parameters())[0][0] )
                    # torch.save(output_4ch, 'before_update_4ch_output.pth')
                    
                    # Dice loss applying Deep supervision concept for teacher network 
                    diceloss_t_1 = dice_t( output_t['pred'][:,0], y1['masks'] ) # outputA['pred'].shape = B x 3 x 4 x 128 x 128 x 128
                    diceloss_t_2 = dice_t( output_t['pred'][:,1], y1['masks'] )
                    diceloss_t_3 = dice_t( output_t['pred'][:,2], y1['masks'] )

                    diceloss_t_1.div_(args.batch_size)
                    dist.all_reduce(diceloss_t_1)
                    diceloss_t_2.div_(args.batch_size)
                    dist.all_reduce(diceloss_t_2)
                    diceloss_t_3.div_(args.batch_size)
                    dist.all_reduce(diceloss_t_3)
                    
                    print(f'dl_t_1/iter : {diceloss_t_1:.4f} || dl_t_2/iter : {diceloss_t_2:.4f} || dl_t_3/iter : {diceloss_t_3:.4f}')

                    diceloss_t_total =  10*( diceloss_t_1 + 0.5*diceloss_t_2 + 0.25*diceloss_t_3 )

                    # for p in model.module.backboneA.parameters():
                    #     print(p)
                    # pdb.set_trace()

                # Backward teacher model                
                scaler.scale( diceloss_t_total ).backward()
                scaler.step( optimizer_t )
                scaler.update() 

                # print( 'After_update_4ch_pred = ', list(model_t.parameters())[0][0] )
                
                # Backward student model
                set_requires_grad( model_t, requires_grad=False )   # Do not update teacher network parameter 
                optimizer_s.zero_grad()
                
                with torch.cuda.amp.autocast():
                    output_s = model_s( detached_output_t, y2 )
                    loss_s_total = 0.0007 * (output_s['btl_f'] + output_s['btl_z'] ) + 10*( output_s['dl_s'] ) + 0.7*(output_s['cel'])
                                
                scaler.scale( loss_s_total ).backward()
                scaler.step( optimizer_s )
                scaler.update()
                
                loss_t_train += diceloss_t_total
                loss_s_train += loss_s_total
                loss_s_bt += 0.0007 * (output_s['btl_f'] + output_s['btl_z'] )
                loss_s_ce += 0.7*(output_s['cel'])

                # print( 'After_update_4ch_n_2ch_pred = ', list(model_t.parameters())[0][0] )                
                                
#-------------------------------------------------------------------------------------------------------------#

                pbar.update(y1['imgs'].shape[0]*args.ngpus) # tqdm의 progress bar를 input data의 (batch_size) * (사용하는 gpu개수) 만큼 
                                                            # 업데이트 해준다.
                
            if args.rank == 0 and (epoch+1) % 1 == 0:

                # save checkpoint
                state = dict(epoch=epoch + 1,
                             model_t=model_t.state_dict(),      # epoch이 한 번 돌때마다 dictation type으로
                             model_s=model_s.state_dict(),
                             optimizer_t=optimizer_t.state_dict(),
                             optimizer_s=optimizer_s.state_dict(),
                             lr = optimizer_s.param_groups[0]['lr'] )               # epoch, model의 parameter, 그리고 optimizer까지 저장한다.
                torch.save( state, args.checkpoint_dir / nameofModelinfo )
            
            epoch_loss_t = loss_t_train / n_train
            epoch_loss_s = loss_s_train / n_train
            epoch_loss_s_bt = loss_s_bt / n_train
            epoch_loss_s_ce = loss_s_ce / n_train
            
            if args.rank == 1:
                
                # preprocessing for tensorboard visualization
                pred1_cp              = output_t['pred'][:,0].clone() # pred1_cp.shape = B x 3 x 4 x 192 x 192 x 128
                
                # print(pred1_cp.shape)
                
                pred1_onehotNcombinde = OnehotNcombine3D( nn.Softmax(dim=1)(pred1_cp) )[0].unsqueeze(0).detach().cpu() # 채널차원으로 combine 되었기때문에 --> H x W x D --> 1 x H x W x D
                pred2_cp              = output_s['pred_s'].clone()
                pred2_onehotNcombinde = OnehotNcombine3D( nn.Softmax(dim=1)(pred2_cp) )[0].unsqueeze(0).detach().cpu() # 채널차원으로 combine 되었기때문에 --> H x W x D --> 1 x H x W x D
                y2_mask_combine       = Combine3D(y2['masks'])[0].unsqueeze(0).detach().cpu()                          # 채널차원으로 combine 되었기때문에 --> H x W x D --> 1 x H x W x D
                
                featureMap1_cp = output_s['f_t'].clone()                # B x 512 x 2 x 2 x 2
                featureMap1_cp = featureMap1_cp[0,0:8].detach().cpu()    # 8 x 2 x 2 x 2
                featureMap2_cp = output_s['f_s'].clone()                # B x 512 x 2 x 2 x 2
                featureMap2_cp = featureMap2_cp[0,0:8].detach().cpu()    # 8 x 2 x 2 x 2
                                
                # pdb.set_trace()
                
                # pred1이나 pred2나 combine 후 size는 똑같음 / f1,f2도 마찬가지 이유
                size_p = pred2_onehotNcombinde.size()  # C x H x W x D = 1 x 192 x 192 x 128
                size_f = featureMap1_cp.size()         # C x H x W x D = 8 x 2 x 2 x 2
                
                pred1_onehotNcombinde_depth_to_batch = torch.zeros( size_p[-1], size_p[0], size_p[1], size_p[2] )  # D x 1 x H x W
                pred2_onehotNcombinde_depth_to_batch = torch.zeros( size_p[-1], size_p[0], size_p[1], size_p[2] )  # D x 1 x H x W
                y2_mask_combine_dim_to_batch         = torch.zeros( size_p[-1], size_p[0], size_p[1], size_p[2] )  # D x 1 x H x W
                
                
                f1 = torch.zeros( size_f[1], size_f[1], size_f[0]*size_f[-1] ) # 2 x 2 x (8x2)
                f2 = torch.zeros( size_f[1], size_f[1], size_f[0]*size_f[-1] ) # 2 x 2 x (8x2)
                i  = 0
                
                for ch in range( featureMap1_cp.size(0) ): # 8
                    for slice_nb in range( featureMap1_cp.size(-1) ): # 2
                        f1[:,:,i] = featureMap1_cp[ch,:,:,slice_nb]
                        f2[:,:,i] = featureMap2_cp[ch,:,:,slice_nb]
                        i += 1
                
                f1 = f1.unsqueeze(0) # B x H x W x D = 1 x 2 x 2 x 16
                f2 = f2.unsqueeze(0) # B x H x W x D = 1 x 2 x 2 x 16
                
                size_cf = f1.size()
                
                f1_depth_to_batch = torch.zeros( size_cf[-1], size_cf[0], size_cf[1], size_cf[2] )  # D x 1 x H x W = 128 x 1 x 16 x 16
                f2_depth_to_batch = torch.zeros( size_cf[-1], size_cf[0], size_cf[1], size_cf[2] )  # D x 1 x H x W = 128 x 1 x 16 x 16
                
                for idx in range( pred2_onehotNcombinde.size(-1) ): # 같은 맥락으로 pred1이나 pred2 모두 slice는 128로 동일하다
                    
                    pred1_onehotNcombinde_depth_to_batch[idx,:,:,:] = pred1_onehotNcombinde[:,:,:,idx]
                    pred2_onehotNcombinde_depth_to_batch[idx,:,:,:] = pred2_onehotNcombinde[:,:,:,idx]
                    y2_mask_combine_dim_to_batch[idx,:,:,:]         = y2_mask_combine[:,:,:,idx]
                    
                for idx in range( f1.size(-1) ): # 같은 맥락으로 f1이나 f2 모두 slice는 128으로 동일하다
                    
                    f1_depth_to_batch[idx,:,:,:] = f1[:,:,:,idx]
                    f2_depth_to_batch[idx,:,:,:] = f2[:,:,:,idx]

                # print(y2_mask_combine_dim_to_batch.shape)
                # print(type(y2_mask_combine_dim_to_batch))
                
                
                
                # tensorboard visualization
                
                """ngrid = nrow --> The Final grid size is (B / nrow, nrow)"""
                 
                writer.add_image('Seq_KD_gt/train', tb_image_grid(y2_mask_combine_dim_to_batch, ngrid=8), epoch)   
                writer.add_image('Seq_KD_output_teacher/train', tb_image_grid(pred1_onehotNcombinde_depth_to_batch, ngrid=8), epoch)
                writer.add_image('Seq_KD_output_student/train', tb_image_grid(pred2_onehotNcombinde_depth_to_batch, ngrid=8), epoch)
                writer.add_image('Seq_KD_f_teacher/train', tb_image_grid(f1_depth_to_batch, ngrid=4), epoch)
                writer.add_image('Seq_KD_f_student/train', tb_image_grid(f2_depth_to_batch, ngrid=4), epoch)
                
                writer.add_scalar('Loss[epoch]/train_teacher', epoch_loss_t.item(), epoch )   # drawing title : 'Loss/train' graph
                                                                                    # (x, y) = (global_step, loss.item())
                writer.add_scalar('Loss[epoch]/train_student', epoch_loss_s.item(), epoch )
                writer.add_scalar('Loss[epoch]/train_s_bt', epoch_loss_s_bt.item(), epoch )
                writer.add_scalar('Loss[epoch]/train_s_ce', epoch_loss_s_ce.item(), epoch )
                                                                                                                
                writer.add_scalar('lr[epoch]/train',  optimizer_t.param_groups[0]['lr'], epoch )
        
            # Validation phase
            if (epoch + 1) % 1 == 0:
                val_loss = eval_net( model_s, valSampler, valLoader, gpu, args, writer, epoch ) 
        
        scheduler_t.step()
        scheduler_s.step()
                
        if args.rank == 0 and (epoch+1) % 5 == 0:
            # save final model
            torch.save(model_t.state_dict(),
                       args.checkpoint_dir / f'Seq_KD_teacher_epoch_{epoch+1}.pth')
            torch.save(model_s.module.backboneB.state_dict(),
                       args.checkpoint_dir / f'Seq_KD_student_epoch_{epoch+1}.pth')
            # torch.save(model_s.module.projector1.state_dict(),
            #            args.checkpoint_dir / f'Seq_KD_projector1_epoch_{epoch+1}.pth')
            # torch.save(model_s.module.projector2.state_dict(),
            #            args.checkpoint_dir / f'Seq_KD_projector2_epoch_{epoch+1}.pth')
            # torch.save(model_s.module.projector3.state_dict(),
            #            args.checkpoint_dir / f'Seq_KD_projector3_epoch_{epoch+1}.pth')
            # torch.save(model_s.module.projector4.state_dict(),
            #            args.checkpoint_dir / f'Seq_KD_projector4_epoch_{epoch+1}.pth')
        
        # # Validation phase
        # if epoch % 2 == 0:

        
    writer.close()
#-----------------------------------------------------------------------------------------------------------



def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * args.learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * args.learning_rate_biases

if __name__ == '__main__':
    logging.basicConfig( level=logging.INFO, format='%(levelname)s: %(message)s' )
    main()
