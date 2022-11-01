import pickle
import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from torch.utils.data.dataloader import default_collate

import argparse
import logging


# 将p_lidx中的PAD=0，根据batch修改idx
def collate_fn(batch):
    new_batch = []
    n = batch[0][0].size(0)
    for i,s in enumerate(batch):
        p_lidx = []
        for idx in s[2]:
            if idx == -1:
                p_lidx.append(0)
            else:
                p_lidx.append(i*n + idx + 1)
        p_lidx = torch.LongTensor(p_lidx)
        new_batch.append((s[0], s[1], p_lidx, s[3], s[4]))
    return default_collate(new_batch)

# def collate_fn(batch):
#     n_link = [0]
#     n_path = [0]
#     for i, s in enumerate(batch[:-1]):
#         n_link.append(n_link[i] + len(s[0]['bandwidth']))
#         n_path.append(n_path[i] + len(s[0]['package']))

#     sample = {
#         'package': torch.cat([s[0]['package'] for s in batch], dim=0),
#         'bandwidth': torch.cat([s[0]['bandwidth'] for s in batch], dim=0),
#         'path': torch.cat([s[0]['path'] + n for s, n in zip(batch, n_link)], dim=0),
#         'row': torch.cat([s[0]['row'] + n for s, n in zip(batch, n_path)], dim=0),
#         'col': torch.cat([s[0]['col'] for s in batch], dim=0),
#         'link_idx': torch.cat([s[0]['link_idx'] + n for s, n in zip(batch, n_link)], dim=0),
#         'path_idx': torch.cat([s[0]['row'] for s in batch], dim=0)
#     }
#     target = torch.cat([s[1] for s in batch], dim=0)
#     new_batch = [(sample, target)]
#     return default_collate(new_batch)


class NetDataset(Dataset):
    def __init__(self, sample_list, label_str):
        super().__init__()
        self.sample_list = sample_list
        if label_str == 'delay':
            self.delay = True
        elif label_str == 'jitter':
            self.delay = False

    def __getitem__(self, index):
        s = self.sample_list[index]
        bw, tr, p_lidx, mask, delay, jitter = s
        if self.delay:
            y = delay
        else:
            y = jitter
        # del s['delay']
        # del s['jitter']
        return bw, tr, p_lidx, mask, y
    
    def __len__(self):
        return len(self.sample_list)

class NetDataModule(object):
    def __init__(self, args):
        super().__init__()
        self.logger = logging.getLogger(args.mode)
        self.train_path = args.data_dir + '/' + args.net + '/process/train'
        self.eval_path = args.data_dir + '/' + args.net + '/process/eval'
        if not args.label in ['delay', 'jitter']:
            self.logger.info('input wrong label_str. label_str should be one of [delay / jitter]')
            exit(0)
        self.args = args
    
    def process_data(self, data_path):
        if os.path.isdir(data_path):
            _, _, data_file = next(os.walk(data_path))
            # data_type = data_path.split('/')[-1]
        elif os.path.isfile(data_path):
            data_file = [data_path.split('/')[-1]]
            data_path = self.eval_path
        sample_list = []
        for file in data_file:
            with open(os.path.join(data_path, file), 'rb') as f:
                data_list = pickle.load(f)
            sample_list.extend(data_list)
        # with tqdm(data_file) as t:
        #     for file in t:
        #         t.set_description(f"{data_type} data loading")
        #         with open(os.path.join(data_path, file), 'rb') as f:
        #             data_list = pickle.load(f)
        #         sample_list.extend(data_list)
        return sample_list

    def setup(self):
        if self.args.mode == 'train':
            train_list = self.process_data(self.train_path)
            eval_list = self.process_data(self.eval_path)
            self.train, self.eval = NetDataset(train_list, self.args.label), NetDataset(eval_list, self.args.label)
        
        if self.args.mode == 'eval':
            eval_list = self.process_data(self.eval_path)
            self.eval = NetDataset(eval_list, self.args.label)

        if self.args.mode == 'eval_one':
            data_path = self.args.data_dir + '/' + self.args.net + '/process/eval/' + self.args.eval_file
            eval_list = self.process_data(data_path)
            self.eval = NetDataset(eval_list, self.args.label)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.args.batch, shuffle=True, collate_fn=collate_fn, num_workers=8)
    
    def val_dataloader(self):
        return DataLoader(self.eval, batch_size=self.args.batch, shuffle=True, collate_fn=collate_fn, num_workers=8)
    
    def test_dataloader(self):
        return DataLoader(self.eval, batch_size=self.args.batch, collate_fn=collate_fn, num_workers=8)

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='RouteNet')
    args.add_argument('-t', '--t', default=8, type=int, help='RNN repeat number')
    args.add_argument('-dl', '--d_link', default=32, type=int, help='link embedding dimensions')
    args.add_argument('-dp', '--d_path', default=32, type=int, help='path embedding dimensions')
    args.add_argument('-df', '--d_fc', default=256, type=int, help='hidden layer dimensions')

    args.add_argument('-d', '--device', default='cuda', type=str, help='running device (cpu / cuda)')
    args.add_argument('-s', '--max_steps', default=1000, type=int, help='max running steps')
    args.add_argument('-b', '--batch', default=32, type=int, help='batch size')
    args.add_argument('-op', '--optim', default='adam', type=str, help='optimization function')
    args.add_argument('-m', '--mode', default='train', type=str, help='running mode (train / eval)')
    args.add_argument('-l', '--label', default='delay', type=str, help='predict indicators (delay / jitter)')

    args.add_argument('-lr', '--lr', default=0.001, type=float, help='learning rate of optimizer')
    args.add_argument('-ls', '--log_step', default=100, type=int, help='learning rate of optimizer')
    args.add_argument('-st', '--step_per_test', default=5000, type=float, help='learning rate of optimizer')

    args.add_argument('-n', '--net', default='nsfnetbw', type=str, help='select dataset net')
    args.add_argument('-p', '--process', action='store_true', help='process dataset')
    args.add_argument('-dir', '--data_dir', default='./dataset', type=str, help='dataset path')
    args.add_argument('-tr', '--test_rate', default=0.2, type=float, help='split rate of test_data')
    args.add_argument('-nw', '--num_worker', default=0, type=int, help='num_worker of dataloader')
    args = args.parse_args()

    dm = NetDataModule(args)
    dm.setup()
    for bw, tr, p_lidx, mask, y in dm.train_dataloader():
        print(bw.size())
        print(tr.size())
        print(p_lidx.size())
        print(mask.size())
        print(y.size())
        exit(0)
    # for x, y in dm.test_dataloader():
    #     print(y)