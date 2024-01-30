from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from utils import *
import argparse
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
from torchsummary import summary


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    results = dict()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(args.device)

    SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).to(args.device)
    PSNR = PeakSignalNoiseRatio(data_range=1.0).to(args.device)

    channel, im_size, _, _, _, _, testloader, _ = get_dataset(
        args.dataset, args.data_path, batch_size=args.batch_size)


    student_distilled = E2E_Unfolding_Distill(args.k_student, im_size[0], im_size[1], channel, n_stages=args.stages, binary=args.binary_student,
                                    is_noise=args.is_noise, snr=args.snr).to(args.device)

    student_distilled.load_state_dict(torch.load(args.student_weights_path))
    student_distilled.to(args.device)

    for param in list(student_distilled.parameters()):
        param.requires_grad = False

    student_distilled.eval()

    torch.autograd.set_detect_anomaly(True)

    mse_loss = nn.MSELoss()
    args.save_path = args.save_path + '_' + str(args.k_student) + '_stages_' + str(
        args.stages) + '_k_student_' + str(args.k_student) + '_snr_' + str(args.snr) + '_dB_'
    
    writer, _, _, metrics_path = save_metrics(f'{args.save_path}')

    test_psnr = AverageMeter()
    test_ssim = AverageMeter()

    data_loop_test = tqdm(enumerate(testloader), total=len(testloader), colour='green')
    for _, test_data in data_loop_test:
        test_img, _ = test_data
        test_img = test_img.to(args.device)

        output_student, _ = student_distilled(test_img)

        loss = mse_loss(output_student[-1], test_img)

        batch_psnr = PSNR(test_img, output_student[-1])
        batch_ssim = SSIM(test_img, output_student[-1])

        test_psnr.update(batch_psnr.item(), test_img.size(0))
        test_ssim.update(batch_ssim.item(), test_img.size(0))

        dict_metrics = dict(test_psnr=test_psnr.avg, test_ssim=test_ssim.avg, mse=loss.item())

        data_loop_test.set_description(f'Test')
        data_loop_test.set_postfix(**dict_metrics)

        for key, value in dict_metrics.items():
            writer.add_scalar(f'test_{key}', value)

    results['psnr_test'] = test_psnr.avg
    results['ssim_test'] = test_ssim.avg
    results['mse_test'] = loss.item()
    print(results)

    # SAVE METRICS
    save_npy_metric(results, f'{metrics_path}/results')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='FashionMNIST')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--k_student', type=int, default=int(32 * 32 * 0.5))
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--stages', type=int, default=7)
    parser.add_argument('--save_path', type=str, default='./noise_test_BASE')
    parser.add_argument('--sparse', type=bool, default=True)
    parser.add_argument('--binary_student', type=bool, default=True)
    parser.add_argument('--is_noise', type=bool, default=True)
    parser.add_argument('--snr', type=int, default=5)
    parser.add_argument('--student_weights_path', type=str, default='2_baseline_binary_512_7_20_dB/model/43_lr_0.001_k_512_36.58_38.29.pth')

    args = parser.parse_args()
    print(args)
    main(args)