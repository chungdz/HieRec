import os
import argparse
import json
import pickle
from tqdm import tqdm
import time
import torch
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import logging
from torch.utils.data import DataLoader
import math
from datasets.recodata import RecoData
from datasets.config import ModelConfig
from gather import gather as gather_all
from models.hieReco import HieRec
from utils.train_util import set_seed
from utils.train_util import save_checkpoint_by_epoch

def run(cfg, rank, device, finished, train_dataset_path, valid_dataset_file, user_emb):
    """
    train and evaluate
    :param args: config
    :param rank: process id
    :param device: device
    :param train_dataset: dataset instance of a process
    :return:
    """
    
    set_seed(7)
    print("Worker %d is setting dataset ... " % rank)
    # Build Dataloader
    train_dataset = RecoData(cfg.mc, np.load(train_dataset_path), user_emb)
    train_data_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    valid_dataset = RecoData(cfg.mc, valid_dataset_file, user_emb)
    valid_data_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False)

    # # Build model.
    model = HieRec(cfg.mc)
    model.to(device)
    # Build optimizer.
    steps_one_epoch = len(train_data_loader)
    train_steps = cfg.epoch * steps_one_epoch
    print("Total train steps: ", train_steps)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.lr)
    print("Worker %d is working ... " % rank)
    # Fast check the validation process
    if (cfg.gpus < 2) or (cfg.gpus > 1 and rank == 0):
        validate(cfg, -1, model, device, rank, valid_data_loader, fast_dev=True)
        logging.warning(model)
        gather_all(cfg.result_path, 1, validate=True, save=False)
    
    # Training and validation
    for epoch in range(cfg.epoch):
        # print(model.match_prediction_layer.state_dict()['2.bias'])
        train(cfg, epoch, rank, model, train_data_loader,
              optimizer, steps_one_epoch, device)
    
        validate(cfg, epoch, model, device, rank, valid_data_loader)
        # add finished count
        finished.value += 1

        if (cfg.gpus < 2) or (cfg.gpus > 1 and rank == 0):
            save_checkpoint_by_epoch(model.state_dict(), epoch, cfg.checkpoint_path)

            while finished.value < cfg.gpus:
                time.sleep(1)
            gather_all(cfg.result_path, cfg.gpus, validate=True, save=False)
            finished.value = 0

def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size


def train(cfg, epoch, rank, model, loader, optimizer, steps_one_epoch, device):
    """
    train loop
    :param args: config
    :param epoch: int, the epoch number
    :param gpu_id: int, the gpu id
    :param rank: int, the process rank, equal to gpu_id in this code.
    :param model: gating_model.Model
    :param loader: train data loader.
    :param criterion: loss function
    :param optimizer:
    :param steps_one_epoch: the number of iterations in one epoch
    :return:
    """
    model.train()

    model.zero_grad()

    enum_dataloader = enumerate(loader)
    if ((cfg.gpus < 2) or (cfg.gpus > 1 and rank == 0)):
        enum_dataloader = enumerate(tqdm(loader, total=len(loader), desc="EP-{} train".format(epoch)))

    for i, data in enum_dataloader:
        if i >= steps_one_epoch:
            break
        # data = {key: value.to(device) for key, value in data.items()}
        data = data.to(device)

        batch_size = data.size(0)
        start_point = torch.arange(batch_size)
        start_point = start_point.unsqueeze(-1)
        start_point = start_point.to(device)
        # 1. Forward
        pred = model(data[:, 2:], start_point).squeeze()
        loss = F.cross_entropy(pred, data[:, 1])

        # 3.Backward.
        loss.backward()

        if cfg.gpus > 1:
            average_gradients(model)
        
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        # scheduler.step()
        model.zero_grad()

    # if (not args.dist_train) or (args.dist_train and rank == 0):
    #     util.save_checkpoint_by_epoch(
    #         model.state_dict(), epoch, args.checkpoint_path)


def validate(cfg, epoch, model, device, rank, valid_data_loader, fast_dev=False, top_k=20):
    model.eval()

    # Setting the tqdm progress bar
    if rank == 0:
        data_iter = tqdm(enumerate(valid_data_loader),
                        desc="EP_test:%d" % epoch,
                        total=len(valid_data_loader),
                        bar_format="{l_bar}{r_bar}")
    else:
        data_iter = enumerate(valid_data_loader)
                        
    with torch.no_grad():
        preds, truths, imp_ids = list(), list(), list()
        for i, data in data_iter:
            if fast_dev and i > 10:
                break

            imp_ids += data[:, 0].cpu().numpy().tolist()
            data = data.to(device)
            batch_size = data.size(0)
            start_point = torch.arange(batch_size)
            start_point = start_point.unsqueeze(-1)
            start_point = start_point.to(device)

            # 1. Forward
            pred = model(data[:, 2:], start_point, test_mode=True)
            if pred.dim() > 1:
                pred = pred.squeeze()
            try:
                preds += pred.cpu().numpy().tolist()
            except:
                preds.append(int(pred.cpu().numpy()))
            truths += data[:, 1].long().cpu().numpy().tolist()

        tmp_dict = {}
        tmp_dict['imp'] = imp_ids
        tmp_dict['labels'] = truths
        tmp_dict['preds'] = preds

        with open(cfg.result_path + 'tmp_{}.json'.format(rank), 'w', encoding='utf-8') as f:
            json.dump(tmp_dict, f)
        f.close()


def init_processes(cfg, local_rank, vocab, dataset, valid_dataset, user_emb, finished, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    addr = "localhost"
    port = cfg.port
    os.environ['MASTER_ADDR'] = addr
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group(backend, rank=0 + local_rank,
                            world_size=cfg.gpus)

    device = torch.device("cuda:{}".format(local_rank))

    fn(cfg, local_rank, device, finished, train_dataset_path=dataset, valid_dataset_file=valid_dataset, user_emb=user_emb)


def split_dataset(dataset, gpu_count):
    sub_len = len(dataset) // gpu_count
    if len(dataset) != sub_len * gpu_count:
        len_a, len_b = sub_len * gpu_count, len(dataset) - sub_len * gpu_count
        dataset, _ = torch.utils.data.random_split(dataset, [len_a, len_b])

    return torch.utils.data.random_split(dataset, [sub_len, ] * gpu_count)

def split_valid_dataset(dataset, gpu_count):
    sub_len = math.ceil(len(dataset) / gpu_count)
    data_list = []
    for i in range(gpu_count):
        s = i * sub_len
        e = (i + 1) * sub_len
        data_list.append(dataset[s: e])

    return data_list

def main(cfg):
    
    set_seed(7)
    
    print('load dev')
    validate_dataset = np.load("{}/raw/dev-all.npy".format(cfg.root))
    print('load config')
    model_cfg = ModelConfig(cfg.root)
    user_emb = np.load('{}/user_emb.npy'.format(cfg.root))

    cfg.mc = model_cfg
    cfg.result_path = '{}/result/'.format(cfg.root)
    cfg.checkpoint_path = '{}/checkpoint/'.format(cfg.root)
    finished = mp.Value('i', 0)

    assert(cfg.gpus > 1)
    valid_dataset_list = split_valid_dataset(validate_dataset, cfg.gpus)

    processes = []
    for rank in range(cfg.gpus):
        p = mp.Process(target=init_processes, args=(
            cfg, rank, None, "{}/raw/train-{}-new.npy".format(cfg.root, rank), valid_dataset_list[rank], user_emb, finished, run, "nccl"))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--gpus', type=int, default=2, help='gpu_num')
    parser.add_argument('--epoch', type=int, default=10, help='the number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--port', type=int, default=9337)
    parser.add_argument("--root", default="data", type=str)
    opt = parser.parse_args()
    logging.warning(opt)

    main(opt)
