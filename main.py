import argparse
import datetime
import os
import sys
from pathlib import Path
from tempfile import mkdtemp
import warnings
import numpy as np
from sklearn.metrics import cohen_kappa_score
from agent import Agent
from utils import Logger, param_count, save_toFile, visualize_ls


def args_define():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent-dim', type=int, default=20, metavar='L', help='latent dimensionality (default: 21)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='batch size of model [default: 64]')
    parser.add_argument('--iteration', type=int, default=10, metavar='N', help='No of iterations (mvae+mh) [default: 10]')
    parser.add_argument('--mvae-epochs', type=int, default=100, metavar='N', help='No of epochs of mvae [default: 10]')
    parser.add_argument('--mh-epochs', type=int, default=100, metavar='N', help='No of epochs of MH naming game [default: 10]')
    parser.add_argument('--expert', type=str, default='MoE', choices=['MoE', 'PoE', 'MoPoE'], help='Type of expert')
    parser.add_argument('--K', type=int, default=10, metavar='N', help='number of categories')
    parser.add_argument('--D', type=int, default=10000, metavar='N', help='number of data points')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR', help='learning rate [default: 1e-3]')
    parser.add_argument('--run-path', type=str, default=None, help='directory for saving models')
    parser.add_argument('--device', type=str, default='cpu', help='device for training [mps, cuda, cpu]')
    parser.add_argument('--variational-beta', type=float, default=1.0, metavar='N', help='beta of vae [default: 1.0]')
    parser.add_argument('--lambda-vision', type=float, default=1.0, help='scaling of vision [default: 1.0]')
    parser.add_argument('--lambda-audio', type=float, default=1.0, help='scaling of audio [default: 1.0]')
    parser.add_argument('--lambda-tactile', type=float, default=1.0, help='scaling of tactile [default: 1.0]')
    parser.add_argument('-f', '--file', help='path for input file')
    return parser.parse_args()


def initialize():
    # os.makedirs('./trained_models', exist_ok=True)
    runId = datetime.datetime.now().isoformat()
    experiment_dir = Path('experiments/')
    experiment_dir.mkdir(parents=True, exist_ok=True)
    runPath = mkdtemp(prefix=runId, dir=str(experiment_dir))
    sys.stdout = Logger('{}/run.log'.format(runPath))
    print('Expt:', runPath)
    print('RunID:', runId)
    os.mkdir(runPath + '/A')
    os.mkdir(runPath + '/B')
    return runPath


if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    args = args_define()
    global_w = np.random.multinomial(1, [1 / args.K] * args.K, size=args.D)
    args.run_path = initialize() + '/'
    a_path = args.run_path + 'A/'
    b_path = args.run_path + 'B/'
    A = Agent(name='A', args=args, channel=global_w)
    B = Agent(name='B', args=args, channel=global_w)

    print(args)
    print('--------------------')
    print(A.model)
    print('Expert:', args.expert)
    print('Number of parameters of Agent A: %d' % param_count(A.model))
    cohen_kappa = []
    mode = -1  # 0 = no communication; 1 = all communication; -1 = MH algorithm
    if mode == -1:
        print('Communication type: Metropolis Hastings Naming Game')
    elif mode == 1:
        print('Communication type: All Communication, All Accepted')
    else:
        print('Communication type: No Communication, All Rejected')

    for i0 in range(args.iteration):
        print('--------------------iteration', i0, '--------------------')
        print('Training Agent A...')
        A.train()
        print('Training Agent B...')
        B.train()

        A.initial_hyperparameter()
        B.initial_hyperparameter()
        if i0 == 0:
            A.initial_mu_lamb()
            B.initial_mu_lamb()

        print('-----MH Learning-----')
        if mode == -1 or mode == 1:
            for i1 in range(args.mh_epochs):
                A.speakTo(agent=B, mode=mode)
                B.evaluate()
                B.update()
                B.speakTo(agent=A, mode=mode)
                A.evaluate()
                A.update()
                if i1 % 20 == 0:
                    visualize_ls(A.z_means, A.pred_label, a_path, 'a_' + str(i1))
                    visualize_ls(B.z_means, B.pred_label, b_path, 'b_' + str(i1))
                kappa = cohen_kappa_score(A.pred_label, B.pred_label)
                cohen_kappa.append(kappa)
                print('MH Epoch {} - Kappa: {:.4f} - ARI(A): {:.4f} - ARI(B): {:.4f} - DBS(A): {:.4f} - DBS(B): {:.4f}'.
                      format(i1, kappa, A.ARI[-1], B.ARI[-1], A.davies_bouldin[-1], B.davies_bouldin[-1]))
            A.prior_update()
            B.prior_update()

        if mode == 0:
            for i1 in range(args.mh_epochs):
                A.speakTo(agent=B, mode=mode)
                A.update()
                B.speakTo(agent=A, mode=mode)
                B.update()
                kappa = cohen_kappa_score(A.pred_label, B.pred_label)
                cohen_kappa.append(kappa)
                print('MH Epoch {} - Kappa: {:.4f}'.format(i1, kappa))

        save_toFile(path=a_path, file_name='_means_' + str(i0), data_saved=A.z_means, rows=1)
        save_toFile(path=b_path, file_name='_means_' + str(i0), data_saved=B.z_means, rows=1)
        save_toFile(path=a_path, file_name='_pred_label_' + str(i0), data_saved=A.pred_label, rows=0)
        save_toFile(path=b_path, file_name='_pred_label_' + str(i0), data_saved=B.pred_label, rows=0)

    print('Kappa Coincidence:', cohen_kappa)
    print('-----Agent A Summary:-----')
    print('ARI:', A.ARI)
    print('Davies_Bouldin:', A.davies_bouldin)
    print('-----Agent B Summary:-----')
    print('ARI:', B.ARI)
    print('Davies_Bouldin:', B.davies_bouldin)

    save_toFile(path=args.run_path, file_name='kappa', data_saved=cohen_kappa, rows=0)
    save_toFile(path=a_path, file_name='loss', data_saved=A.lossList, rows=0)
    save_toFile(path=a_path, file_name='ari', data_saved=A.ARI, rows=0)
    save_toFile(path=a_path, file_name='dbs', data_saved=A.davies_bouldin, rows=0)
    save_toFile(path=b_path, file_name='loss', data_saved=B.lossList, rows=0)
    save_toFile(path=b_path, file_name='ari', data_saved=B.ARI, rows=0)
    save_toFile(path=b_path, file_name='dbs', data_saved=B.davies_bouldin, rows=0)

