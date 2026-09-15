import argparse
import math
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

from utils.experiment import experiment_setting
from utils.perturbation import PERTURBATION_TYPES

parser = argparse.ArgumentParser(description='Model family for Time Series Forecasting')

# random seed
parser.add_argument('--random_seed', type=int, default=2026, help='random seed')

# test-time robustness evaluation
parser.add_argument(
    '--perturb_type',
    type=str,
    default='none',
    choices=PERTURBATION_TYPES,
    help=(
        "test-time input perturbation; 'last' reproduces TameR's recent "
        "single anomalous point, 'point' targets --perturb_offset, and "
        "'none' keeps the original test behavior"
    ),
)
parser.add_argument(
    '--perturb_ratio',
    type=float,
    default=3.0,
    help='noise scale as a multiple of each input window/channel standard deviation',
)
parser.add_argument(
    '--perturb_seed',
    type=int,
    default=2024,
    help='independent seed used only to generate test-time perturbations',
)
parser.add_argument(
    '--perturb_offset',
    type=int,
    default=1,
    help='point position counted backward from the forecast origin (1=latest)',
)
parser.add_argument(
    '--boundary_fix',
    type=int,
    default=0,
    choices=(0, 1),
    help=(
        'test-time boundary comparison; 1 also evaluates a view where the '
        'latest input point is replaced by the preceding point'
    ),
)
parser.add_argument(
    '--boundary_reconstruct',
    type=int,
    default=0,
    choices=(0, 1),
    help=(
        'train/load a prefix-only boundary reconstructor and replace the '
        'latest channel value only when its increment exceeds the MAD filter'
    ),
)
parser.add_argument(
    '--boundary_threshold',
    type=float,
    default=3.0,
    help='MAD threshold for filtered context boundary replacement',
)
parser.add_argument(
    '--boundary_hidden_dim',
    type=int,
    default=64,
    help='hidden size of the context boundary reconstructor',
)
parser.add_argument(
    '--boundary_epochs',
    type=int,
    default=20,
    help='maximum training epochs for the context boundary reconstructor',
)
parser.add_argument(
    '--boundary_patience',
    type=int,
    default=5,
    help='validation patience for the context boundary reconstructor',
)
parser.add_argument(
    '--boundary_learning_rate',
    type=float,
    default=0.001,
    help='learning rate for the context boundary reconstructor',
)

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='GTR',
                    help='model name, options: [Informer, Autoformer, ...]')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument(
    '--dataset_name',
    type=str,
    default=None,
    help='stable dataset label written to perturbation result files',
)
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')  #fixed
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

# TQNet & CycleNet
parser.add_argument('--cycle', type=int, default=24, help='cycle length')
parser.add_argument('--model_type', type=str, default='mlp', help='model type, options: [linear, mlp]')
parser.add_argument('--use_revin', type=int, default=1, help='1: use revin or 0: no revin')

# parameter-free NTE wrapper
parser.add_argument(
    '--nte_cutoff_ratio',
    type=float,
    default=0.1,
    help='fraction of Seq_Len retained as low-frequency FFT bins',
)
parser.add_argument(
    '--nte_alpha',
    type=float,
    default=1.0,
    help='strength of the inverse-SNR trend damping used by GTRNTE',
)
parser.add_argument(
    '--nte_gamma_max',
    type=float,
    default=20.0,
    help='maximum damping coefficient used by GTRNTE',
)
parser.add_argument(
    '--nte_guard_sigma',
    type=float,
    default=3.0,
    help='prefix-only robust threshold for guarding the most recent point',
)

# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=0, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

# SegRNN
parser.add_argument('--rnn_type', default='gru', help='rnn_type')
parser.add_argument('--dec_way', default='pmf', help='decode way')
parser.add_argument('--seg_len', type=int, default=48, help='segment length')
parser.add_argument('--channel_id', type=int, default=1, help='Whether to enable channel position encoding')

# Formers
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args()

if args.perturb_ratio < 0:
    parser.error('--perturb_ratio must be non-negative')
if not 1 <= args.perturb_offset <= args.seq_len:
    parser.error('--perturb_offset must be in [1, seq_len]')
if args.boundary_fix and args.perturb_type != 'none':
    parser.error('--boundary_fix cannot be combined with input perturbation')
if args.boundary_fix and args.boundary_reconstruct:
    parser.error('--boundary_fix and --boundary_reconstruct are mutually exclusive')
if not math.isfinite(args.boundary_threshold) or args.boundary_threshold <= 0:
    parser.error('--boundary_threshold must be finite and positive')
if args.boundary_hidden_dim <= 0:
    parser.error('--boundary_hidden_dim must be positive')
if args.boundary_epochs <= 0:
    parser.error('--boundary_epochs must be positive')
if args.boundary_patience <= 0:
    parser.error('--boundary_patience must be positive')
if not math.isfinite(args.boundary_learning_rate) or args.boundary_learning_rate <= 0:
    parser.error('--boundary_learning_rate must be finite and positive')
if not math.isfinite(args.nte_cutoff_ratio) or not (
    0.0 < args.nte_cutoff_ratio <= 1.0
):
    parser.error('--nte_cutoff_ratio must be in (0, 1]')
if not math.isfinite(args.nte_alpha) or args.nte_alpha < 0.0:
    parser.error('--nte_alpha must be non-negative')
if not math.isfinite(args.nte_gamma_max) or args.nte_gamma_max <= 0.0:
    parser.error('--nte_gamma_max must be positive')
if not math.isfinite(args.nte_guard_sigma) or args.nte_guard_sigma <= 0.0:
    parser.error('--nte_guard_sigma must be positive')

# random seed
fix_seed = args.random_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main


if args.is_training:
    for ii in range(args.itr):

        # setting record of experiments
        setting = experiment_setting(args, fix_seed)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        if args.boundary_reconstruct:
            print(
                '>>>>>>>training boundary reconstructor : '
                '{}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting)
            )
            exp.train_boundary_reconstructor(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        torch.cuda.empty_cache()
else:
    ii = 0
    setting = experiment_setting(args, fix_seed)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
