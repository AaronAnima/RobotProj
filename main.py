import numpy as np
import torch
from network import Classifier
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
import random

from data import DataLoader, Args
from utils import exists_or_mkdir
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_networks(network, dataset_eval, logger, suffix=''):
    batch_img_whole, batch_steer_whole = dataset_eval.load_whole()
    steer_pred_whole = network(batch_img_whole)
    pred_loss_eval = torch.nn.L1Loss()(steer_pred_whole, batch_steer_whole)
    # print(steer_pred_whole[0:8])
    # print(batch_steer_whole[0:8])
    # input()
    # write eval loss
    logger.add_scalar('Losses/eval_loss' + suffix, pred_loss_eval, epoch)


if __name__ == "__main__":
    # setup args
    args = Args()
    args = args.parser
    
    # dataset path collected
    c1_path = './Replays/corner1'
    c2_path = './Replays/corner2'
    ca1_path = './Replays/corneranti1'
    ca2_path = './Replays/corneranti2'
    s1_path = './Replays/straight1'
    s2_path = './Replays/straight2'
    sa_path = './Replays/straightanti'
    
    # setup logger
    exp_path = './logs/' + args.exp_name + '/'
    log_path = './logs/' + args.exp_name + '/eval/'
    exists_or_mkdir('./logs')
    exists_or_mkdir(exp_path)
    exists_or_mkdir(log_path)
    writer = SummaryWriter(log_path)
    
    # setup dataloader, network, and optimizer
    kwargs_dataset = {"data_path": None,
                      "device": device,
                      "batchsize": args.batchsize,
                      "shape": (args.input_h, args.input_w),
                      "order": args.order,
                      "img_c": 1 if args.is_grey else 3}
    kwargs_dataset["data_path"] = c1_path
    data_c1 = DataLoader(**kwargs_dataset)
    
    kwargs_dataset["data_path"] = c2_path
    data_c2 = DataLoader(**kwargs_dataset)
    
    # kwargs_dataset["data_path"] = ca1_path
    # data_ca1 = DataLoader(**kwargs_dataset)
    #
    # kwargs_dataset["data_path"] = ca2_path
    # data_ca2 = DataLoader(**kwargs_dataset)
    
    # kwargs_dataset["data_path"] = s1_path
    # data_s1 = DataLoader(**kwargs_dataset)
    #
    # kwargs_dataset["data_path"] = s2_path
    # data_s2 = DataLoader(**kwargs_dataset)
    
    # kwargs_dataset["data_path"] = sa_path
    # data_sa = DataLoader(**kwargs_dataset)
    
    print('############# Load Data Finish #################')
    kwargs_actor = {"max_action": np.pi,
                    "h": args.input_h,
                    "w": args.input_w,
                    "action_dim": 1,
                    "order": args.order,
                    "neu_dim": 128,
                    "ch": 32,
                    "depth": 4,
                    "img_c": 1 if args.is_grey else 3}
    net = Classifier(**kwargs_actor).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda t: args.decay_rate ** t)
    print('############# Setup Networks Finish #################')
    
    # define training batches
    # training_batches = data_c1.batches + data_s1.batches
    training_batches = data_c1.batches
    random.shuffle(training_batches)
    # main loop
    for epoch in trange(args.epoch_num):
        for step, batch in enumerate(training_batches):
            # train
            batch_img, batch_steer = batch
            steer_pred = net(batch_img)
            pred_loss = torch.nn.L1Loss()(steer_pred, batch_steer)
            
            optimizer.zero_grad()
            pred_loss.backward()
            optimizer.step()
            # write train loss
            writer.add_scalar('Losses/training_loss', pred_loss, len(training_batches) * epoch + step)

        writer.add_scalar('Losses/lr_rate', args.decay_rate ** epoch, epoch)
        # write training res, wait for kai long
        # data_test.eval(net, exp_path, 'cw_color')
        # eval and step lr-scheduler
        eval_networks(net, data_c2, writer, suffix='corner')
        # eval_networks(net, data_s2, writer, suffix='straight')
        scheduler.step()
