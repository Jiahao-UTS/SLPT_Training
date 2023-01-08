import argparse

from Config import cfg
from Config import update_config

from utils import create_logger
from model import Sparse_alignment_network
from Dataloader import WFLW_Dataset, W300_Dataset
from utils import AverageMeter


from tensorboardX import SummaryWriter

import torch
import cv2
import numpy as np
import pprint
import os

import torchvision.transforms as transforms


def parse_args():
    parser = argparse.ArgumentParser(description='Train Sparse Facial Network')

    # philly
    parser.add_argument('--modelDir', help='model directory', type=str, default='./Checkpoint')
    parser.add_argument('--logDir', help='log directory', type=str, default='./log')
    parser.add_argument('--dataDir', help='data directory', type=str, default='./')
    parser.add_argument('--target', help='targeted branch (alignmengt, emotion or pose)',
                        type=str, default='alignment')
    parser.add_argument('--prevModelDir', help='prev Model directory', type=str, default=None)

    args = parser.parse_args()

    return args

def calcuate_loss(name, pred, gt, trans):

    pred = (pred - trans[:, 2]) @ np.linalg.inv(trans[:, 0:2].T)

    if name == 'WFLW':
        norm = np.linalg.norm(gt[60, :] - gt[72, :])
    elif name == '300W':
        norm = np.linalg.norm(gt[36, :] - gt[45, :])
    elif name == 'COFW':
        norm = np.linalg.norm(gt[17, :] - gt[16, :])
    else:
        raise ValueError('Wrong Dataset')

    error_real = np.mean(np.linalg.norm((pred - gt), axis=1) / norm)

    return error_real

def main_function():
    args = parse_args()
    update_config(cfg, args)
    logger, final_output_dir, tb_log_dir = create_logger(cfg, cfg.TARGET)
    logger.info(pprint.pformat(args))
    logger.info(cfg)

    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    if cfg.DATASET.DATASET == '300W':
        model = Sparse_alignment_network(cfg.W300.NUM_POINT, cfg.MODEL.OUT_DIM,
                                    cfg.MODEL.TRAINABLE, cfg.MODEL.INTER_LAYER,
                                    cfg.MODEL.DILATION, cfg.TRANSFORMER.NHEAD,
                                    cfg.TRANSFORMER.FEED_DIM, cfg.W300.INITIAL_PATH, cfg)
    elif cfg.DATASET.DATASET == 'WFLW':
        model = Sparse_alignment_network(cfg.WFLW.NUM_POINT, cfg.MODEL.OUT_DIM,
                                    cfg.MODEL.TRAINABLE, cfg.MODEL.INTER_LAYER,
                                    cfg.MODEL.DILATION, cfg.TRANSFORMER.NHEAD,
                                    cfg.TRANSFORMER.FEED_DIM, cfg.WFLW.INITIAL_PATH, cfg)

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }


    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    if cfg.DATASET.DATASET == '300W':
        valid_dataset = W300_Dataset(
            cfg, cfg.W300.ROOT,  False,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
    elif cfg.DATASET.DATASET == 'WFLW':
        valid_dataset = WFLW_Dataset(
            cfg, cfg.WFLW.ROOT, False,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
    else:
        raise ValueError('Wrong Dataset')

    # 验证数据迭代器
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    checkpoint_file = "./WFLW_6_layer.pth"
    checkpoint = torch.load(checkpoint_file)

    model.module.load_state_dict(checkpoint)

    error_list = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, meta) in enumerate(valid_loader):
            Annotated_Points = meta['Annotated_Points'].numpy()[0]
            Trans = meta['trans'].numpy()[0]

            outputs_initial = model(input.cuda())

            output = outputs_initial[2][0, -1, :, :].cpu().numpy()

            error = calcuate_loss(cfg.DATASET.DATASET, output * cfg.MODEL.IMG_SIZE, Annotated_Points, Trans)
            error_list.update(error, input.size(0))

            msg = 'Epoch: [{0}/{1}]\t' \
                  'NME: {error:.3f}%\t'.format(
                i, len(valid_loader), error=error_list.avg * 100.0)

            print(msg)

        print("finished")
        print("Mean Error: {:.3f}".format(error_list.avg * 100.0))

if __name__ == '__main__':
    main_function()

