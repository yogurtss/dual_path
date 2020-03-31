import os
import torch
import numpy as np
from math import floor
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from models.dual_path import LowHighNetwork
from dataset.audioDatasetHdf import AudioDataset
from torch.utils.tensorboard import SummaryWriter

import argparse
import datetime
from dataset.util import *
from losses.conv_tas_criterion import ConvTasCriterion

nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H')
result_dir = './result/{}'.format(nowTime)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)


def save_model(model, optimizer, step, path):
    if len(os.path.dirname(path)) > 0 and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
    }, path)


def load_model(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        checkpoint['optimizer_state_dict']['param_groups'][0]['lr'] = 0.0001
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']
    return step


def cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=bool, default=True,
                        help='use gpu, default True')
    parser.add_argument('--model_path', type=str, default='{}/model_'.format(result_dir),
                        help='Path to save model')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--output_size', type=float, default=2.0,
                        help='Output duration')
    parser.add_argument('--sr', type=int, default=16000,
                        help='Sampling rate')
    parser.add_argument('--length', type=float, default=2.0,
                        help='Duration of input audio')
    parser.add_argument('--loss', type=str, default="L1",
                        help="L1 or L2")
    parser.add_argument('--channels', type=int, default=1,
                        help="Input channel, mono or sterno, default mono")
    parser.add_argument('--h5_dir', type=str, default='../WaveUNet/H5/',
                        help="Path of hdf5 file")
    parser.add_argument("--load", type=bool, default=None)
    parser.add_argument("--half_step", type=int, default=3,
                        help="Epochs of half lr")
    parser.add_argument("--hold_step", type=int, default=20,
                        help="Epochs of hold step")
    parser.add_argument("--example_freq", type=int, default=200,
                        help="write an audio summary into Tensorboard logs")
    parser.add_argument("--load_model", type=str, default='result/2020-03-08-00/model_best.pth')
    return parser.parse_args()


def valiadate(model, criterion, val_loader):
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        for i, (x, targets) in enumerate(val_loader):
            x = x.cuda()
            targets = targets.cuda()
            output = model(x)
            loss = criterion(output, targets)
            total_loss += loss.item()
    return total_loss


def main():
    args = cfg()
    writer = SummaryWriter(result_dir)
    shapes = {'length': 31992}
    args.load = True

    INSTRUMENTS = {"bass": True,
                   "drums": True,
                   "other": True,
                   "vocals": True,
                   "accompaniment": True}
    augment_func = lambda mix, targets: random_amplify_raw(mix, targets, 0.7, 1.0)
    # crop_func = lambda mix, targets: crop(mix, targets, shapes)
    train_dataset = AudioDataset('train', INSTRUMENTS, args.sr, args.channels, 1, True, args.h5_dir, shapes, augment_func)
    val_dataset = AudioDataset('val', INSTRUMENTS, args.sr, args.channels, 1, False, args.h5_dir, shapes)
    # test_dataset = AudioDataset('test', INSTRUMENTS, args.sr, args.channels, 2, False, args.h5_dir, shapes, crop_func)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = LowHighNetwork(1, 256, 512, 32, 64)
    # model = WaveUNetRaw()
    model = model.cuda()
    # model.initialize()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # criterion = ConvTasCriterion()
    criterion = nn.MSELoss()

    # Set up training state dict that will also be saved into checkpoints
    state = {"step": 0,
             "worse_epochs": 0,
             "epochs": 0,
             "improve_epochs": 0,
             "best_loss": np.Inf}

    if args.load is not None:
        print("Continuing training full model from checkpoint " + str(args.load_model))
        state = load_model(model, optimizer, args.load_model)

    print('Start training...')
    while state["worse_epochs"] < args.hold_step:
        # 如果3个epoch性功能还未提升，则lr降低一半
        if state["improve_epochs"] == 3:
            optim_state = optimizer.state_dict()
            optim_state['param_groups'][0]['lr'] = \
                optim_state['param_groups'][0]['lr'] / 2.0
            optimizer.load_state_dict(optim_state)
            print('Learning rate adjusted to: {lr:.6f}'.format(
                lr=optim_state['param_groups'][0]['lr']))
            state["improve_epochs"] = 0

        print("Training one epoch from iteration " + str(state["step"]))
        model.train()
        for i, (x, targets) in enumerate(train_loader):
            x = x.cuda()
            targets = targets.cuda()
            cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
            writer.add_scalar("learning_rate", cur_lr, state['step'])
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, targets)
            loss.backward()
            writer.add_scalar("training_loss", loss, state['step'])
            optimizer.step()
            # clr.step()
            state['step'] += 1

            if i % args.example_freq == 0:
                input_centre = x[0, :]
                writer.add_audio("input", input_centre, state["step"], sample_rate=args.sr)
                pred_ = outputs[0, 0, :].view(1, -1)
                targets_ = targets[0, 0, :].view(1, -1)

                writer.add_audio("pred", pred_, state['step'], sample_rate=args.sr)
                writer.add_audio("target", targets_, state['step'], sample_rate=args.sr)
            if i % 50 == 0:
                print("{:4d}/{:4d} --- Loss: {:.6f} with learnig rate {:.6f}".format(i, len(train_dataset)//args.batch_size, loss, cur_lr))

        val_loss = valiadate(model, criterion, val_loader)
        val_loss /= len(val_dataset)//args.batch_size
        print("Valiadation loss" + str(val_loss))
        writer.add_scalar("val_loss", val_loss, state['step'])

        writer.add_scalar("val_loss", val_loss, state["step"])

        # EARLY STOPPING CHECK
        checkpoint_path = args.model_path + str(state['step']) + '.pth'
        print("Saving model...")
        if val_loss >= state["best_loss"]:
            state["worse_epochs"] += 1
            state["improve_epochs"] += 1
        else:
            print("MODEL IMPROVED ON VALIDATION SET!")
            state["worse_epochs"] = 0
            state["best_loss"] = val_loss
            state["best_checkpoint"] = checkpoint_path
            state["improve_epochs"] = 0
            best_checkpoint_path = args.model_path + 'best.pth'
            save_model(model, optimizer, state, best_checkpoint_path)
        print(state)
        state["epochs"] += 1
        if state["epochs"] % 5 == 0:
            save_model(model, optimizer, state, checkpoint_path)
    last_model = args.model_path + 'last_model.pth'
    save_model(model, optimizer, state, last_model)
    print("Training finished")
    writer.close()


if __name__ == '__main__':
    main()
