import numpy as np
import torch
from network import Classifier
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from data import DataLoader, Args
from utils import exists_or_mkdir
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_networks(network, dataset_eval, logger):
    batch_img_whole, batch_steer_whole = dataset_eval.load_whole()
    steer_pred_whole = network(batch_img_whole)
    pred_loss_eval = torch.nn.L1Loss()(steer_pred_whole, batch_steer_whole)
    # print(steer_pred_whole[0:8])
    # print(batch_steer_whole[0:8])
    # input()
    # write eval loss
    logger.add_scalar('Losses/eval_loss', pred_loss_eval, epoch)


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
    
    # temp test for debug
    test_path_corner = './Replays/cw_color'
    test_path_straight = './Replays/cw_st_color'
    
    # setup logger
    exp_path = './logs/' + args.exp_name + '/'
    log_path = './logs/' + args.exp_name + '/eval/'
    exists_or_mkdir('./logs')
    exists_or_mkdir(exp_path)
    exists_or_mkdir(log_path)
    writer = SummaryWriter(log_path)
    
    # setup dataloader, network, and optimizer
    kwargs_dataset = {"data_path": test_path_corner,
                      "device": device,
                      "batchsize": args.batchsize,
                      "shape": (args.input_h, args.input_w),
                      "order": args.order}
    
    data_test = DataLoader(**kwargs_dataset)
    print('############# Load Data Finish #################')
    kwargs_classifier = {"max_action": np.pi,
                         "h": args.input_h,
                         "w": args.input_w,
                         "action_dim": 1,
                         "order": args.order,
                         "neu_dim": 128,
                         "ch": 32,
                         "depth": 4,
                         "img_c": 1 if args.is_grey else 3 * args.order}
    net = Classifier(**kwargs_classifier).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda t: 0.995 ** t)
    print('############# Setup Networks Finish #################')
    
    epoch_num = len(data_test)
    # main loop
    for epoch in trange(args.epoch_num):
        for step in range(len(data_test)):
            # train
            batch_img, batch_steer = data_test.load()
            steer_pred = net(batch_img)
            pred_loss = torch.nn.L1Loss()(steer_pred, batch_steer)
            
            optimizer.zero_grad()
            pred_loss.backward()
            optimizer.step()
            # write train loss
            writer.add_scalar('Losses/training_loss', pred_loss, epoch_num * epoch + step)
        
        # write training res
        data_test.eval(net, exp_path, 'cw_color')
        # eval and step lr-scheduler
        eval_networks(net, data_test, writer)
        scheduler.step()
