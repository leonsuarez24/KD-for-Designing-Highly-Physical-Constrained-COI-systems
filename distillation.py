from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from utils import *
import argparse
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
from torchsummary import summary


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    student_loss = np.zeros(args.epochs_distill)

    ssim_test_student = np.zeros(args.epochs_distill)
    psnr_test_student = np.zeros(args.epochs_distill)
    ssim_train_student = np.zeros(args.epochs_distill)
    psnr_train_student = np.zeros(args.epochs_distill)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).to(args.device)
    PSNR = PeakSignalNoiseRatio(data_range=1.0).to(args.device)

    channel, im_size, _, _, _, _, testloader, trainloader = get_dataset(
        args.dataset, args.data_path, batch_size=args.batch_size)

    student = E2E_Unfolding_Distill(args.k_student, im_size[0], im_size[1], channel, n_stages=args.stages, binary=args.binary_student).to(
        args.device)
    teacher = E2E_Unfolding_Distill(args.k_teacher, im_size[0], im_size[1], channel, n_stages=args.stages, binary=args.binary_teacher).to(
        args.device)

    teacher.load_state_dict(torch.load('teacher_real_valued_819_7/model/50_lr_0.0001_k_819_50.73_50.87.pth'))
    teacher.to(args.device)

    for param in list(teacher.parameters()):
        param.requires_grad = False

    teacher.eval()

    optimizer_student = torch.optim.Adam(student.parameters(), lr=args.lr_stu)
    torch.autograd.set_detect_anomaly(True)

    mse_loss = nn.MSELoss()
    args.save_path = args.save_path + '_' + str(args.k_student) + '_gamma_' + str(args.gamma) + '_stages_' + str(
        args.stages) + '_k_teacher_' + str(args.k_teacher)
    writer, images_path, model_path, metrics_path = save_metrics(f'{args.save_path}')

    for epoch in range(args.epochs_distill):
        student.train(True)

        train_stu_loss = AverageMeter()
        train_corr_loss = AverageMeter()
        train_psnr = AverageMeter()
        train_ssim = AverageMeter()
        test_psnr = AverageMeter()
        test_ssim = AverageMeter()

        data_loop_train = tqdm(enumerate(trainloader), total=len(trainloader), colour='red')
        for _, train_data in data_loop_train:

            train_img, _ = train_data
            train_img = train_img.to(args.device)

            output_teacher, sparse_teacher = teacher(train_img)
            output_student, sparse_student = student(train_img)

            # FEATURE LOSS
            if args.sparse:
                student_features = torch.permute(torch.stack(sparse_student), [1, 0, 2, 3, 4])
                teacher_features = torch.permute(torch.stack(sparse_teacher), [1, 0, 2, 3, 4])
            else:
                student_features = torch.permute(torch.stack(output_student), [1, 0, 2, 3, 4])
                teacher_features = torch.permute(torch.stack(output_teacher), [1, 0, 2, 3, 4])

            cc_s = matrix_combinatorial(student_features, gamma=args.gamma)
            cc_t = matrix_combinatorial(teacher_features, gamma=args.gamma)

            loss_corr = torch.linalg.norm(cc_s - cc_t, 'fro')
            optimizer_student.zero_grad()

            batch_psnr = PSNR(train_img, output_student[-1])
            batch_ssim = SSIM(train_img, output_student[-1])

            s_loss = mse_loss(output_student[-1], output_teacher[-1])
            losst_total = s_loss + loss_corr
            losst_total.backward()
            optimizer_student.step()

            # METRICS
            train_psnr.update(batch_psnr.item(), train_img.size(0))
            train_ssim.update(batch_ssim.item(), train_img.size(0))
            train_stu_loss.update(s_loss.item(), train_img.size(0))
            train_corr_loss.update(loss_corr.item(), train_img.size(0))

            dict_metrics = dict(stu_loss=train_stu_loss.avg, corr_loss=train_corr_loss.avg, train_psnr=train_psnr.avg,
                                train_ssim=train_ssim.avg)

            data_loop_train.set_description(f'Train  Epoch [{epoch + 1} / {args.epochs_distill}]')
            data_loop_train.set_postfix(**dict_metrics)

            for key, value in dict_metrics.items():
                writer.add_scalar(f'train_{key}', value, epoch)

        student_loss[epoch] = train_stu_loss.avg
        ssim_train_student[epoch] = train_ssim.avg
        psnr_train_student[epoch] = train_psnr.avg

        # TESTING
        data_loop_test = tqdm(enumerate(testloader), total=len(testloader), colour='green')
        student.eval()
        for _, test_data in data_loop_test:
            test_img, _ = test_data
            test_img = test_img.to(args.device)

            output_student, _ = student(test_img)

            loss = mse_loss(output_student[-1], test_img)

            batch_psnr = PSNR(test_img, output_student[-1])
            batch_ssim = SSIM(test_img, output_student[-1])

            test_psnr.update(batch_psnr.item(), test_img.size(0))
            test_ssim.update(batch_ssim.item(), test_img.size(0))

            dict_metrics = dict(test_psnr=test_psnr.avg, test_ssim=test_ssim.avg, test_loss=loss.item())

            data_loop_test.set_description(f'Test  Epoch [{epoch + 1} / {args.epochs_distill}]')
            data_loop_test.set_postfix(**dict_metrics)

            for key, value in dict_metrics.items():
                writer.add_scalar(f'test_{key}', value, epoch)

        psnr_test_student[epoch] = test_psnr.avg
        ssim_test_student[epoch] = test_ssim.avg

        # SAVE APERTURE CODES AS PNG
        save_aperture_codes(student.optical_layer, 32, 2, images_path, f'student_{test_psnr.avg}')

        # SAVE STUDENT WEIGHTS
        torch.save(student.state_dict(), f'{model_path}/student_{test_psnr.avg}.pth')

    # SAVE METRICS
    save_npy_metric(psnr_train_student, f'{metrics_path}/psnr_train_student')
    save_npy_metric(psnr_test_student, f'{metrics_path}/psnr_test_student')
    save_npy_metric(ssim_test_student, f'{metrics_path}/ssim_test_student')
    save_npy_metric(ssim_train_student, f'{metrics_path}/ssim_train_student')
    save_npy_metric(student_loss, f'{metrics_path}/student_loss')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--lr_stu', type=float, default='1e-3')
    parser.add_argument('--epochs_distill', type=int, default='50')
    parser.add_argument('--dataset', type=str, default='FashionMNIST')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--k_student', type=int, default=int(32 * 32 * 0.1))
    parser.add_argument('--k_teacher', type=int, default=int(32 * 32 * 0.8))
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--stages', type=int, default=7)
    parser.add_argument('--gamma', type=float, default='0.000001')
    parser.add_argument('--save_path', type=str, default='./Unrolling_distillation')
    parser.add_argument('--sparse', type=bool, default=True)
    parser.add_argument('--binary_teacher', type=bool, default=False)
    parser.add_argument('--binary_student', type=bool, default=True)
    args = parser.parse_args()
    print(args)
    main(args)


