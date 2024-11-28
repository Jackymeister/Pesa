#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import argparse
import pickle
import time
from utils import build_graph, Data, split_validation
from model import *
from SoftmaxEQL import get_eql_class_weights


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# torch.cuda.set_device(1)
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--batchSize', type=int, default=50, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
parser.add_argument('--norm', default=True, help='adapt NISER, l2 norm over item and session embedding')
parser.add_argument('--TA', default=False, help='use target-aware or not')
parser.add_argument('--scale', default=True, help='scaling factor sigma')

parser.add_argument('--use_SEQL', default="false", type=str2bool, help='use SEQL loss')
parser.add_argument('--include_common', default="false", type=str2bool, help='use common category')
parser.add_argument('--lambda_low', type=float, default=5, help='occurrence count threshold for SEQL')
parser.add_argument('--lambda_high', type=float, default=20, help='occurrence count threshold for SEQL')
parser.add_argument('--lambda_', type=float, default=5, help='occurrence count threshold for SEQL')
parser.add_argument('--ignore_prob', type=float, default=0.5, help='probability threshold to ignore weight for SEQL')
opt = parser.parse_args()
print(opt)


def main():
    train_data = pickle.load(open('../datasets/' + opt.dataset + '/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))
    # all_train_seq = pickle.load(open('../datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))
    # g = build_graph(all_train_seq)
    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)
    # del all_train_seq, g
    if opt.dataset == 'diginetica':
        n_node = 43098
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
    else:
        n_node = 310

    model = trans_to_cuda(SessionGraph(opt, n_node))

    # get class weights
    if opt.use_SEQL:
        class_weights = model.loss_function.class_weight.cpu().detach().numpy()
    else:
        class_weights = get_eql_class_weights(opt)

    # percentage of frequent items
    # print(class_weights)
    print(f"Percentage of frequent items : {round(np.mean(class_weights) * 100, 2)}")

    start = time.time()
    best_result = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    best_epoch = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit, mrr, cov, cov_tail, tail, hit_f, mrr_f, hit_r, mrr_r = train_test(model, train_data, test_data, class_weights)
        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        if cov >= best_result[2]:
            best_result[2] = cov
            best_epoch[2] = epoch
            flag = 1
        if cov_tail >= best_result[3]:
            best_result[3] = cov_tail
            best_epoch[3] = epoch
            flag = 1
        if tail >= best_result[4]:
            best_result[4] = tail
            best_epoch[4] = epoch
            flag = 1

        if hit_f >= best_result[5]:
            best_result[5] = hit_f
        if mrr_f >= best_result[6]:
            best_result[6] = mrr_f
        if hit_r >= best_result[7]:
            best_result[7] = hit_r
        if mrr_r >= best_result[8]:
            best_result[8] = mrr_r

        print('Best Result:')
        print('\tRecall@20: %.4f\tMMR@20: %.4f\tCoverage@20: %.4f\tTail_Coverage@20: %.4f\tTail@20: %.4f\tEpoch:\t%d,\t%d,\t%d,\t%d,\t%d'
            % (best_result[0], best_result[1], best_result[2], best_result[3], best_result[4],
               best_epoch[0], best_epoch[1], best_epoch[2], best_epoch[3], best_epoch[4]))
        print('Result (frequent):')
        print('\tRecall@20: %.4f\tMMR@20: %.4f' % (best_result[5], best_result[6]))
        print('Result (rare):')
        print('\tRecall@20: %.4f\tMMR@20: %.4f' % (best_result[7], best_result[8]))

        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
