import os
import argparse
import logging
from logger import config_logger
from dataloader import NetDataModule
from trainer import BaseModel
from process import DataProcess
from eval import Eval

args = argparse.ArgumentParser(description='')
args.add_argument('-model', '--model', default='NewModel', type=str, help='choose model')
args.add_argument('-v', '--version', default=0, type=int, help='Model Version')
args.add_argument('-de', '--depth', default=8, type=int, help='MHSA repeat number')
args.add_argument('-dim', '--dim', default=64, type=int, help='bw and tr embedding dimensions')
args.add_argument('-dm', '--mlp_dim', default=256, type=int, help='hidden layer dimensions')

args.add_argument('-d', '--device', default='cuda', type=str, help='running device (cpu / cuda)')
args.add_argument('-s', '--max_steps', default=1000, type=int, help='max running steps')
args.add_argument('-b', '--batch', default=32, type=int, help='batch size')
args.add_argument('-op', '--optim', default='adam', type=str, help='optimization function')
args.add_argument('-m', '--mode', default='train', type=str, help='running mode (train / eval / eval_one)')
args.add_argument('-f', '--eval_file', default=None, type=str, help='eval file name')
args.add_argument('-l', '--label', default='delay', type=str, help='predict indicators (delay / jitter)')

args.add_argument('-lr', '--lr', default=0.001, type=float, help='learning rate of optimizer')
args.add_argument('-dr', '--decay_rate', default=0.5, type=float, help='decay rate of optimizer learning rate')
# args.add_argument('-su', '--step_update', default=5000, type=int, help='step numbers of update learning rate')
args.add_argument('-ls', '--log_step', default=100, type=int, help='learning rate of optimizer')
args.add_argument('-st', '--step_per_test', default=5000, type=float, help='learning rate of optimizer')
args.add_argument('-es', '--earlystopping', default=10, type=int, help='stop training after best result existing n test_steps')

args.add_argument('-n', '--net', default='nsfnetbw', type=str, help='select dataset net')
args.add_argument('-p', '--process', action='store_true', help='process dataset')
args.add_argument('-dir', '--data_dir', default='./dataset', type=str, help='dataset path')
args.add_argument('-tr', '--test_rate', default=0.2, type=float, help='split rate of test_data')
args.add_argument('-md', '--model_dir', default='./models', type=str, help='model weight file')
args = args.parse_args()

if not os.path.exists(args.model_dir):
    os.mkdir(args.model_dir)
if not os.path.exists('./logs'):
    os.mkdir('./logs')

config_logger(mode=args.mode, filename='./logs/' + args.net + '_' + args.model + '_v' + str(args.version) +'.log')

args.model_dir = f"{args.model_dir}/{args.net}_{args.model}.pt"

if args.net == 'nsfnetbw':
    max_len = 4
    n_links = 42
elif args.net == 'geant2bw':
    max_len = 7
    n_links = 74
else:
    max_len = -1

logger = logging.getLogger(args.mode)
out_str = "dim:{} depth:{} mlp_dim:{}, batch:{}, lr:{}, decay_rate:{}, max_steps:{}".format(
            args.dim, args.depth, args.mlp_dim, args.batch, args.lr, args.decay_rate, args.max_steps)
logger.info(out_str)

if args.process:
    pes = DataProcess(args.data_dir, args.net, args.test_rate)
    max_len, n_links = pes.process()

dm = NetDataModule(args)
dm.setup()

if max_len == -1:
    print('use -p to process dataset first')
    exit(0)


if args.mode == 'train':
    trainer = BaseModel(
        data_loader=dm.train_dataloader(),
        args=args,
        max_len=max_len,
        n_links=n_links,
        valid_dataloader=dm.val_dataloader()
    )
    trainer.train()

if args.mode == 'eval':
    if os.path.exists(args.model_dir):
        eval = Eval(
            data_loader=dm.val_dataloader(),
            args=args,
            max_len=max_len,
            n_links=n_links
        )
        eval.eval()
    else:
        print('Error: model training weight does not exist!')
        exit(0)

if args.mode == 'eval_one':
    logger.info("# test network file: " + args.eval_file)
    eval = Eval(
            data_loader=dm.test_dataloader(),
            args=args,
            max_len=max_len,
            n_links=n_links
        )
    eval.eval()