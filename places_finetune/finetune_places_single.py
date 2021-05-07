from data import DoomImage
import numpy as np

import time
from torch.utils.data import DataLoader
import tensorflow as tf
from tqdm import tqdm
from imageio import imwrite as imsave
from data import Places, PlacesRoom, PlacesOutdoor
from habitat_baselines.rl.models.resnet import ResNetEncoder
import habitat_baselines.rl.models.resnet as resnet
import torch
from gym.spaces import Box
from gym import spaces
import torch.nn.functional as F
import torch
torch.manual_seed(1)
from util import adjust_learning_rate, AverageMeter, accuracy
from tensorflow.python.platform import flags
import torch.nn as nn
import sys
import torch.optim as optim
import tensorboard_logger as tb_logger
import torchvision.models as models
import torchvision.models as models

FLAGS = flags.FLAGS

flags.DEFINE_bool('places_full', False, 'use all of places')
flags.DEFINE_float('learning_rate', 0.1, 'learning rate')
flags.DEFINE_list('lr_decay_epochs', [30, 40, 50], 'epochs to decay learning rate')
flags.DEFINE_string('mode', 'crl', 'type of model to load')
flags.DEFINE_bool('policy', False, 'whether to use model or policy')


class PlacesLinear(nn.Module):
    def __init__(self, classes):
        super(PlacesLinear, self).__init__()

        self.fc = nn.Linear(2048, classes)


    def forward(self, inp):
        logits = self.fc(inp)
        return logits


class TorchVisionResNet50(nn.Module):
    r"""
    Takes in observations and produces an embedding of the rgb component.

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
        device: torch.device
    """

    def __init__(
        self, pretrained=True, spatial_output: bool = False
    ):
        super().__init__()
        self.device = torch.device('cuda')
        self.resnet_layer_size = 2048
        linear_layer_input_size = 0

        self.cnn = models.resnet50(pretrained=pretrained)
        self.layer_extract = self.cnn._modules.get("avgpool")


    def forward(self, observations):
        r"""Sends RGB observation through the TorchVision ResNet50 pre-trained
        on ImageNet. Sends through fully connected layer, activates, and
        returns final embedding.
        """

        def resnet_forward(observation):
            resnet_output = torch.zeros(1, dtype=torch.float32, device=self.device)

            def hook(m, i, o):
                resnet_output.set_(o)

            # output: [BATCH x RESNET_DIM]
            h = self.layer_extract.register_forward_hook(hook)
            self.cnn(observation)
            h.remove()
            return resnet_output

        # permute tensor to dimension [BATCH x CHANNEL x HEIGHT x WIDTH]
        rgb_observations = observations["rgb"].permute(0, 3, 1, 2)
        rgb_observations = rgb_observations / 255.0  # normalize RGB
        resnet_output = resnet_forward(rgb_observations.contiguous())

        return resnet_output


def set_optimizer(args, classifier):
    # optimizer = optim.SGD(classifier.parameters(),
    #                       lr=args.learning_rate,
    #                       momentum=0.9,
    #                       weight_decay=0.0)

    # if args.policy:
    #     optimizer = optim.Adam(classifier.parameters(),
    #                           lr=1e-4)
    # else:
    optimizer = optim.Adam(classifier.parameters(),
                          lr=1e-3)
    return optimizer

def train(epoch, train_loader, model, classifier, criterion, optimizer):
    """
    one epoch training
    """
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.float()
        input = input.cuda()
        target = target.cuda().long()

        # ===================forward=====================
        with torch.no_grad():
            im = input * 255
            im = im.float()
            im = im.cuda()
            feat = model({'rgb': im})
            feat = feat.mean(dim=2).mean(dim=2)

        output = classifier(feat)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    return top1.avg, top5.avg, losses.avg


def validate(val_loader, model, classifier, criterion):
    """
    evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    classifier.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            input = input.cuda()
            target = target.cuda().long()

            im = input * 255
            im = im.float()
            im = im.cuda()
            feat = model({'rgb': im})
            feat = feat.mean(dim=2).mean(dim=2)

            output = classifier(feat)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % 10 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg

if __name__ == "__main__":
    resnet_baseplanes = 32
    backbone = "resnet50"


    if FLAGS.places_full:
        classes = 205
        data_fn = Places
    else:
        classes = 59
        data_fn = PlacesRoom

    rgb_box = Box(0, 255, (256, 256, 3))

    model = ResNetEncoder(
        spaces.Dict({"rgb": rgb_box}),
        baseplanes=resnet_baseplanes*2,
        ngroups=resnet_baseplanes // 2,
        make_backbone=getattr(resnet, backbone),
        normalize_visual_inputs=True,
        obs_transform=None,
        backbone_only=True,
        dense=True
    )

    # ckpt = torch.load("/private/home/yilundu/sandbox/habitat/habitat-lab/checkpoints/curiosity_pointnav_pretrain_resnet50_301_resume/curiosity_pointnav_pretrain/curiosity_pointnav_pretrain.16.pth")

    if FLAGS.mode != "imagenet":

        if FLAGS.mode == "crl":
            ckpt = torch.load("/home/gridsan/yilundu/my_files/habitat_lab/checkpoints_single/curiosity_pointnav_mp3d_single/curiosity_pointnav_pretrain/curiosity_pointnav_pretrain.2.pth")
        elif FLAGS.mode == "learned_count":
            ckpt = torch.load("/home/gridsan/yilundu/my_files/habitat_lab/checkpoints_single/learned_count_pointnav_mp3d_single/learned_count_pointnav_pretrain/learned_count_pointnav_pretrain.2.pth")
        elif FLAGS.mode == "random":
            ckpt = torch.load("/home/gridsan/yilundu/my_files/habitat_lab/checkpoints_single/random_pointnav_mp3d_single/random_pointnav_pretrain/random_pointnav_pretrain.2.pth")

        state_dict = ckpt['state_dict']
        weights_new = {}

        for k, v in state_dict.items():
            split_layer_name = k.split(".")[2:]

            if len(split_layer_name) == 0:
                continue

            if FLAGS.policy:
                if "visual_resnet" == split_layer_name[0]:
                    layer_name = ".".join(split_layer_name[1:])
                    weights_new[layer_name] = v
            else:
                if "model_encoder" == split_layer_name[0]:
                    layer_name = ".".join(split_layer_name[1:])
                    weights_new[layer_name] = v

        model.load_state_dict(weights_new, strict=False)
    else:
        model = TorchVisionResNet50()

    model = model.cuda()
    model = model.eval()

    classifier = PlacesLinear(classes).cuda()
    optimizer = set_optimizer(FLAGS, classifier)

    train_data = PlacesRoom(train=True)
    test_data = PlacesRoom(train=False)

    train_dataloader = DataLoader(train_data, batch_size=256, num_workers=6, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(test_data, batch_size=256, num_workers=6, shuffle=True, drop_last=False)

    criterion = nn.CrossEntropyLoss().cuda()
    logger = tb_logger.Logger(logdir="linear_finetune", flush_secs=2)

    for epoch in range(60):
        # adjust_learning_rate(epoch, FLAGS, optimizer)

        print("==> training...")

        train_acc, train_acc5, train_loss = train(epoch, train_dataloader, model, classifier, criterion, optimizer)

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_acc5', train_acc5, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        print("==> testing...")

        test_acc, test_acc5, test_loss = validate(val_dataloader, model, classifier, criterion)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_acc5', test_acc5, epoch)
        logger.log_value('test_loss', test_loss, epoch)

