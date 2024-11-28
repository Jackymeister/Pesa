import time
import argparse
import pickle
from model import *
from utils import *
from SoftmaxEQL import get_eql_class_weights


def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='diginetica/Nowplaying/Tmall/sample')
parser.add_argument('--hiddenSize', type=int, default=100)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--activate', type=str, default='relu')
parser.add_argument('--n_sample_all', type=int, default=12)
parser.add_argument('--n_sample', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay.')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay.')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty ')
parser.add_argument('--n_iter', type=int, default=1)                                    # [1, 2]
parser.add_argument('--dropout_gcn', type=float, default=0, help='Dropout rate.')       # [0, 0.2, 0.4, 0.6, 0.8]
parser.add_argument('--dropout_local', type=float, default=0, help='Dropout rate.')     # [0, 0.5]
parser.add_argument('--dropout_global', type=float, default=0.5, help='Dropout rate.')
parser.add_argument('--validation', default="false", type=str2bool, help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=3)

parser.add_argument('--use_SEQL', default="false", type=str2bool, help='use SEQL loss')
parser.add_argument('--include_common', default="false", type=str2bool, help='use common category')
parser.add_argument('--lambda_low', type=float, default=5, help='occurrence count threshold for SEQL')
parser.add_argument('--lambda_high', type=float, default=20, help='occurrence count threshold for SEQL')
parser.add_argument('--lambda_', type=float, default=5, help='occurrence count threshold for SEQL')
parser.add_argument('--ignore_prob', type=float, default=0.5, help='probability threshold to ignore weight for SEQL')

opt = parser.parse_args()


def main():
    init_seed(2020)

    if opt.dataset == 'diginetica':
        num_node = 43098
        opt.n_iter = 2
        opt.dropout_gcn = 0.2
        opt.dropout_local = 0.0
    elif opt.dataset == 'Nowplaying':
        num_node = 60417
        opt.n_iter = 1
        opt.dropout_gcn = 0.0
        opt.dropout_local = 0.0
    elif opt.dataset == 'Tmall':
        num_node = 40728
        opt.n_iter = 1
        opt.dropout_gcn = 0.6
        opt.dropout_local = 0.5
    elif opt.dataset == 'sample':
        num_node = 310
        opt.n_iter = 1
        opt.dropout_gcn = 0.6
        opt.dropout_local = 0.5
    else:
        num_node = 310

    train_data = pickle.load(open('datasets/' + opt.dataset + '/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('datasets/' + opt.dataset + '/test.txt', 'rb'))

    adj = pickle.load(open('datasets/' + opt.dataset + '/adj_' + str(opt.n_sample_all) + '.pkl', 'rb'))
    num = pickle.load(open('datasets/' + opt.dataset + '/num_' + str(opt.n_sample_all) + '.pkl', 'rb'))
    train_data = Data(train_data)
    test_data = Data(test_data)

    adj, num = handle_adj(adj, num_node, opt.n_sample_all, num)
    model = trans_to_cuda(CombineGraph(opt, num_node, adj, num))

    # get class weights
    if opt.use_SEQL:
        class_weights = model.loss_function.class_weight.cpu().detach().numpy()
    else:
        class_weights = get_eql_class_weights(opt)

    # percentage of frequent items
    # print(class_weights)
    print(f"Percentage of frequent items : {round(np.mean(class_weights) * 100, 2)}")

    print(opt)
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

        print('Current Result:')
        print('\tRecall@20: %.4f\tMMR@20: %.4f' % (hit, mrr))
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
