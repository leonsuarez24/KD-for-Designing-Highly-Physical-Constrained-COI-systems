import torch
from utils import *
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import argparse
import os


def main(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(args.device)

    SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).to(args.device)
    PSNR = PeakSignalNoiseRatio(data_range=1.0).to(args.device)
    if args.binary:
        args.baseline_model_path = args.baseline_model_path+'_'+'binary'+str(args.k_small)+'_'+str(args.n_stages)
    else:
        args.baseline_model_path = args.baseline_model_path+'_real_valued_'+str(args.k_small)+'_'+str(args.n_stages)

    if args.is_noise:
        args.baseline_model_path = args.baseline_model_path + '_' + str(args.snr) + '_dB'
    else:
        args.baseline_model_path = args.baseline_model_path + '_NO_NOISE'
    train_base_model_unfolding(args.k_small, args.epoch_baseline, args.baseline_model_path, SSIM, PSNR, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='FashionMNIST')
    parser.add_argument('--epoch_baseline', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--k_small', type=int, default=int(32*32*0.5))
    parser.add_argument('--lr_baseline', type=float, default=1e-4)
    parser.add_argument('--baseline_model_path', type=str, default='./baseline_non_trainable')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_stages', type=int, default=7)
    parser.add_argument('--binary', type=bool, default=True)
    parser.add_argument('--is_noise', type=bool, default=False)
    parser.add_argument('--trainable_ca', type=bool, default=False)
    parser.add_argument('--snr', type=int, default=20000)
    args = parser.parse_args()
    print(args)
    main(args)
