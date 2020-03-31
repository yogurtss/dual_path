import os
import torch
import numpy as np
from math import floor
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from dataset.audioDatasetRaw import AudioDatasetRaw
from torch.utils.tensorboard import SummaryWriter
import time
from models.wave_net_raw import WaveUNetRaw
import argparse
import datetime
from dataset.util import *

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


def cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=bool, default=True,
                        help='use gpu, default True')
    parser.add_argument('--model_path', type=str, default='{}/model_'.format(result_dir),
                        help='Path to save model')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='initial learning rate')
    parser.add_argument('--max_lr', type=float, default=1e-3,
                        help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output_size', type=float, default=2.0,
                        help='Output duration')
    parser.add_argument('--sr', type=int, default=22050,
                        help='Sampling rate')
    parser.add_argument('--length', type=float, default=2.0,
                        help='Duration of input audio')
    parser.add_argument('--loss', type=str, default="L1",
                        help="L1 or L2")
    parser.add_argument('--channels', type=int, default=1,
                        help="Input channel, mono or sterno, default mono")
    parser.add_argument('--h5_dir', type=str, default='H5/',
                        help="Path of hdf5 file")
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--hold_step", type=int, default=20,
                        help="Epochs of hold step")
    parser.add_argument("--example_freq", type=int, default=200,
                        help="write an audio summary into Tensorboard logs")
    return parser.parse_args()


def valiadate(model, criterion, val_loader):
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        for i, (x, targets) in enumerate(val_loader):
            batch_s = x.size()[0]
            x = x.cuda()
            targets = targets.cuda()
            output = model(x)
            loss = criterion(output, targets)/batch_s
            total_loss += loss.item()
    return total_loss


def main():
    args = cfg()
    writer = SummaryWriter(result_dir)
    shapes = {'length': 16384*2}

    INSTRUMENTS = {"bass": False,
                   "drums": False,
                   "other": False,
                   "vocals": True,
                   "accompaniment": True}
    augment_func = lambda mix, targets: random_amplify_raw(mix, targets, 0.7, 1.0)
    # crop_func = lambda mix, targets: crop(mix, targets, shapes)
    train_dataset = AudioDatasetRaw('train', INSTRUMENTS, args.sr, args.channels, 1, True, args.h5_dir, shapes, augment_func)
    val_dataset = AudioDatasetRaw('val', INSTRUMENTS, args.sr, args.channels, 1, False, args.h5_dir, shapes)
    # test_dataset = AudioDataset('test', INSTRUMENTS, args.sr, args.channels, 2, False, args.h5_dir, shapes, crop_func)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = WaveUNetRaw()
    # model = WaveUNetRaw()
    model = model.cuda()
    # model.initialize()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    cy_len = floor(len(train_dataset) /args.batch_size // 2)
    clr = lr_scheduler.CyclicLR(optimizer, args.lr, args.max_lr, cy_len, cycle_momentum=False)
    args.loss = 'L2'
    if args.loss == 'L1':
        criterion = nn.L1Loss()
    elif args.loss == 'L2':
        criterion = nn.MSELoss()

    # Set up training state dict that will also be saved into checkpoints
    state = {"step": 0,
             "worse_epochs": 0,
             "epochs": 0,
             "best_loss": np.Inf}

    if args.load is not None:
        print("Continuing training full model from checkpoint " + str(args.load_model))
        state = load_model(model, optimizer, args.load_model)

    print('Start training...')
    while state["worse_epochs"] < args.hold_step:
        print("Training one epoch from iteration " + str(state["step"]))
        model.train()
        for i, (x, targets) in enumerate(train_loader):
            batch_s = x.size()[0]
            x = x.cuda()
            targets = targets.cuda()
            cur_lr = clr.get_lr()[0]
            writer.add_scalar("learning_rate", cur_lr, state['step'])
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, targets)/batch_s
            loss.backward()
            writer.add_scalar("training_loss", loss, state['step'])
            optimizer.step()
            clr.step()
            state['step'] += 1

            if i % args.example_freq == 0:
                input_centre = x[0, :, :]
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
        else:
            print("MODEL IMPROVED ON VALIDATION SET!")
            state["worse_epochs"] = 0
            state["best_loss"] = val_loss
            state["best_checkpoint"] = checkpoint_path
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
