import importlib
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.SCG_TransNet import SwinDeepLab
from nets.deeplabv3_training import (get_lr_scheduler, set_optimizer_lr,
                                     weights_init)
from utils.callbacks import LossHistory
from utils.dataloader import DeeplabDataset, deeplab_dataset_collate
from utils.utils import download_weights, show_config
from utils.utils_fit import fit_one_epoch



'''
训练自己的语义分割模型一定需要注意以下几点：
1、训练前仔细检查自己的格式是否满足要求，该库要求数据集格式为VOC格式，需要准备好的内容有输入图片和标签
   输入图片为.jpg图片，无需固定大小，传入训练前会自动进行resize。
   灰度图会自动转成RGB图片进行训练，无需自己修改。
   输入图片如果后缀非jpg，需要自己批量转成jpg后再开始训练。

   标签为png图片，无需固定大小，传入训练前会自动进行resize。
   由于许多同学的数据集是网络上下载的，标签格式并不符合，需要再度处理。一定要注意！标签的每个像素点的值就是这个像素点所属的种类。
   网上常见的数据集总共对输入图片分两类，背景的像素点值为0，目标的像素点值为255。这样的数据集可以正常运行但是预测是没有效果的！
   需要改成，背景的像素点值为0，目标的像素点值为1。

2、损失值的大小用于判断是否收敛，比较重要的是有收敛的趋势，即验证集损失不断下降，如果验证集损失基本上不改变的话，模型基本上就收敛了。
   损失值的具体大小并没有什么意义，大和小只在于损失的计算方式，并不是接近于0才好。如果想要让损失好看点，可以直接到对应的损失函数里面除上10000。
   训练过程中的损失值会保存在logs文件夹下的loss_%Y_%m_%d_%H_%M_%S文件夹中
   
3、训练好的权值文件保存在logs文件夹中，每个训练世代（Epoch）包含若干训练步长（Step），每个训练步长（Step）进行一次梯度下降。
   如果只是训练了几个Step是不会保存的，Epoch和Step的概念要捋清楚一下。
'''
if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #Cuda Whether to use Cuda, no GPU can be set to False
    Cuda = True

    distributed = False
    # ---------------------------------------------------------------------#
    #   sync_bn Whether to use sync_bn, DDP mode multi-card available
    # ---------------------------------------------------------------------#
    sync_bn = False
    # ---------------------------------------------------------------------#
    #   fp16        Whether to use mixed precision training
    #               It can reduce the video memory by about half, and requires pytorch1.7.1 or above
    # ---------------------------------------------------------------------#
    fp16 = False
    # -----------------------------------------------------#
    #   num_classes  The number of categories you need
    # -----------------------------------------------------#
    num_classes = 6

    # ----------------------------------------------------------------------------------------------------------------------------#
    #   pretrained      Whether to use the pre-trained weights of the backbone network. Here, the weights of the backbone are used, so they are loaded when the model is built.
    #                   If model_path is set, the weight of the backbone does not need to be loaded, and the value of pretrained is meaningless.
    #                   If you do not set model_path, pretrained=True, only the backbone is loaded to start training at this time.
    #                   If you do not set model_path, pretrained = False, Freeze_Train = Fasle, the training starts from 0 at this time, and there is no process of freezing the backbone.
    # ----------------------------------------------------------------------------------------------------------------------------#
    pretrained = False
    model_path = ""

    # ------------------------------#
    #   Enter the image size
    # ------------------------------#
    input_shape = [256, 256]

    # ----------------------------------------------------------------------------------------------------------------------------#
    #   The training is divided into two phases, namely the freezing phase and the unfreezing phase. The freezing stage is set to meet the training needs of students with insufficient machine performance.
    #   Freeze training requires less video memory, and if the graphics card is very poor, you can set Freeze_Epoch equal to UnFreeze_Epoch, and only freeze training will be performed at this time.
    #
    #   Here are some suggestions for setting parameters, and the trainers can flexibly adjust them according to their own needs:
    #   （一）Start training from the pretrained weights of the entire model：
    #       Adam：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'adam'，Init_lr = 5e-4，weight_decay = 0。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'adam'，Init_lr = 5e-4，weight_decay = 0。（不冻结）
    #       SGD：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'sgd'，Init_lr = 7e-3，weight_decay = 1e-4。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'sgd'，Init_lr = 7e-3，weight_decay = 1e-4。（不冻结）
    #       Among them: UnFreeze_Epoch can be adjusted between 100-300。
    #   （二）Start training from the pretrained weights of the backbone network：
    #       Adam：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'adam'，Init_lr = 5e-4，weight_decay = 0。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'adam'，Init_lr = 5e-4，weight_decay = 0。（不冻结）
    #       SGD：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 120，Freeze_Train = True，optimizer_type = 'sgd'，Init_lr = 7e-3，weight_decay = 1e-4。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 120，Freeze_Train = False，optimizer_type = 'sgd'，Init_lr = 7e-3，weight_decay = 1e-4。（不冻结）
    #       Among them: Since the training starts from the pre-trained weights of the backbone network, the weights of the backbone are not necessarily suitable for semantic segmentation, and more training is required to jump out of the local optimal solution.
    #             UnFreeze_Epoch can be adjusted between 120-300.
    #             Adam converges faster than SGD. Therefore, UnFreeze_Epoch can theoretically be smaller, but more Epochs are still recommended.

    # ------------------------------------------------------------------#
    #   Freezing Phase Training Parameters
    #   At this time, the backbone of the model is frozen, and the feature extraction network does not change
    #   Occupies less video memory, only fine-tuning the network
    #   Init_Epoch          The current training generation of the model, its value can be greater than Freeze_Epoch, such as setting:
    #                       Init_Epoch = 60、Freeze_Epoch = 50、UnFreeze_Epoch = 100
    #                       The freezing stage will be skipped, and the corresponding learning rate will be adjusted directly from generation 60.
    #                       (used when resuming practice)
    #   Freeze_Epoch        Freeze_Epoch for model freeze training
    #                      (Failed when Freeze_Train=False)
    #   Freeze_batch_size   Batch_size of model freezing training
    #                      (Failed when Freeze_Train=False)
    # ------------------------------------------------------------------#
    Init_Epoch = 0
    Freeze_Epoch = 0
    Freeze_batch_size = 6
    # ------------------------------------------------------------------#
    #   UnFreeze_Epoch          The total training epoch of the model
    #   Unfreeze_batch_size     The batch_size of the model after unfreezing
    # ------------------------------------------------------------------#
    UnFreeze_Epoch = 150
    Unfreeze_batch_size = 16
    # ------------------------------------------------------------------#
    #   Freeze_Train    Whether to freeze training
    #                   By default, the backbone training is frozen first and then unfreezes the training.
    # ------------------------------------------------------------------#
    Freeze_Train = False


    # ------------------------------------------------------------------#
    #   Init_lr         The maximum learning rate of the model
    #   Min_lr          The minimum learning rate of the model, the default is 0.01 of the maximum learning rate
    # ------------------------------------------------------------------#
    # Init_lr = 5e-4
    Init_lr = 1e-3
    Min_lr = Init_lr * 0.01
    # ------------------------------------------------------------------#
    #   optimizer_type  The type of optimizer used, optional adam, sgd

    #   momentum        The momentum parameter used internally by the optimizer
    #   weight_decay    weight decay
    # ------------------------------------------------------------------#
    optimizer_type = "adam"
    momentum = 0.9
    #weight_decay = 1e-4
    weight_decay = 0
    # ------------------------------------------------------------------#
    #   lr_decay_type   The learning rate drop method to use, the options are 'step', 'cos'
    # ------------------------------------------------------------------#
    lr_decay_type = 'step'
    # ------------------------------------------------------------------#
    #   save_period     How many epochs save a weight
    # ------------------------------------------------------------------#
    save_period = 1
    # ------------------------------------------------------------------#
    #   save_dir        Folder where weights and log files are saved
    # ------------------------------------------------------------------#
    # save_dir = './logs_fuxian1/'
    save_dir = './logs/1/'

    # ------------------------------------------------------------------#
    #   VOCdevkit_path  数据集路径
    # ------------------------------------------------------------------#
    VOCdevkit_path = '../SCG-TransNet/VOCdevkit_vaihingen'

    # ------------------------------------------------------------------#
    dice_loss = True
    focal_loss = False
    # ------------------------------------------------------------------#
    #   Whether to assign different loss weights to different types, the default is balanced.
    #   If you set it, pay attention to setting it in numpy form, and the length is the same as num_classes.
    #   like:
    #   num_classes = 3
    #   cls_weights = np.array([1, 2, 3], np.float32)
    # ------------------------------------------------------------------#
    cls_weights = np.ones([num_classes], np.float32)
    # ------------------------------------------------------------------#
    num_workers = 8
    # ------------------------------------------------------#
    #   Set the graphics card used
    # ------------------------------------------------------#
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0

    # ----------------------------------------------------#
    #   Download pretrained weights
    # ----------------------------------------------------#
    # if pretrained:
    #     if distributed:
    #         if local_rank == 0:
    #             download_weights(backbone)
    #         dist.barrier()
    #     else:
    #         download_weights(backbone)

    # model = DeepLab(num_classes=num_classes, backbone=backbone, downsample_factor=downsample_factor,
    #                 pretrained=pretrained)
    model_config = importlib.import_module(f'model.configs.swin_224_7_4level')
    model = SwinDeepLab(
        model_config.EncoderConfig,
        model_config.ASPPConfig,
        model_config.DecoderConfig
    )
    if not pretrained:
        weights_init(model)
    if model_path != '':
        # ------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        # ------------------------------------------------------#
        #   Load according to the Key of the pre-trained weight and the Key of the model
        # ------------------------------------------------------#
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        # ------------------------------------------------------#
        #   It shows that there is no matching key
        # ------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")


    if local_rank == 0:
        loss_history = LossHistory(save_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()

    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:

            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank],
                                                                    find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()


    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        show_config(
            num_classes=num_classes, model_path=model_path, input_shape=input_shape, \
            Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
            Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train, \
            Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
            lr_decay_type=lr_decay_type, \
            save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
        )
        # ---------------------------------------------------------#
        # Total training generations refers to the total number of times to traverse all data
        # The total training step size refers to the total number of gradient descents
        # Each training generation contains several training steps, and each training step performs a gradient descent.
        # Only the minimum training generation is recommended here, there is no upper limit, and only the unfreezing part is considered in the calculation
        # ----------------------------------------------------------#
        wanted_step = 1.5e4 if optimizer_type == "sgd" else 0.5e4
        total_step = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m" % (optimizer_type, wanted_step))
            print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m" % (
            num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
            print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m" % (
            total_step, wanted_step, wanted_epoch))

    # ------------------------------------------------------#
    # Backbone feature extraction network features are common, freezing training can speed up training
    # It can also prevent the weight from being destroyed in the early stage of training.
    # Init_Epoch is the starting generation
    # Interval_Epoch is the generation of frozen training
    # Epoch total training generations
    # Prompt OOM or insufficient video memory, please reduce Batch_size
    # ------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        # ------------------------------------#
        # Freeze a certain part of the training
        # ------------------------------------#
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        # -------------------------------------------------------------------#
        # If you don't freeze training, directly set batch_size to Unfreeze_batch_size
        # -------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        # -------------------------------------------------------------------#
        # Judge the current batch_size, adaptively adjust the learning rate
        # -------------------------------------------------------------------#
        nbs = 16
        lr_limit_max = 5e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
        # if backbone == "xception":
        #     lr_limit_max = 1e-4 if optimizer_type == 'adam' else 1e-1
        #     lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        # ---------------------------------------#
        # Select the optimizer according to optimizer_type
        # ---------------------------------------#
        optimizer = {
            'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
            'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True,
                             weight_decay=weight_decay)
        }[optimizer_type]

        # -----------------------------------------#
        # Get the formula for learning rate drop
        # -----------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        # -----------------------------------------#
        # Determine the length of each generation
        # -----------------------------------------#
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        train_dataset = DeeplabDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset = DeeplabDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)

        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, )
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, )
            batch_size = batch_size // ngpus_per_node
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True

        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True,
                         drop_last=True, collate_fn=deeplab_dataset_collate, sampler=train_sampler)
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=deeplab_dataset_collate, sampler=val_sampler)

        # -----------------------------------------#
        # start model training
        # -----------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            # -----------------------------------------#
            # If the model has a frozen learning part
            # Unfreeze and set parameters
            # -----------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                # -------------------------------------------------- ------------------#
                # Judge the current batch_size, adaptively adjust the learning rate
                # -------------------------------------------------- ------------------#
                nbs = 16
                lr_limit_max = 5e-4 if optimizer_type == 'adam' else 1e-1
                lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
                # if backbone == "xception":
                #     lr_limit_max = 1e-4 if optimizer_type == 'adam' else 1e-1
                #     lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                # ---------------------------------------#
                #   获得学习率下降的公式
                # ---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True, collate_fn=deeplab_dataset_collate, sampler=train_sampler)
                gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True,
                                     drop_last=True, collate_fn=deeplab_dataset_collate, sampler=val_sampler)

                UnFreeze_flag = True

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss_history, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, dice_loss, focal_loss,
                          cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)

        if local_rank == 0:
            loss_history.writer.close()
