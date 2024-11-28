import torch
import pickle
import numpy as np
import torch.nn.functional as F
from collections import Counter


def get_eql_class_weights(opt):
    # read all labels
    labels = pickle.load(open('../datasets/' + opt.dataset + '/train.txt', 'rb'))[1]
    labels = [l - 1 for l in labels]  # minus 1 since 0 value is used for padding and it is not used during training
    # count the occurrence of each label
    label_count = Counter(labels)
    num_classes = max(set(labels)) + 1
    # initialize class weights array
    class_weights = np.zeros(num_classes)
    if opt.include_common:
        print("Include common item")
    else:
        print("Not include common item")
    for idx, (label, count) in enumerate(sorted(label_count.items(), key=lambda x: -x[1])):
        # if number of count is more than threshold, then set the weight to 1
        if opt.include_common:
            if count < opt.lambda_low:
                class_weights[label] = 0
            elif count >= opt.lambda_low and count <= opt.lambda_high:
                class_weights[label] = 0.5
            else:
                class_weights[label] = 1
        else:
            # class_weights[label] = 1 if count > opt.lambda_ else 0
            class_weights[label] = 0 if count < opt.lambda_ else 1
        # print('idx: {}, cls: {} count: {}, weight: {}'.format(idx, label, count, class_weights[label]))
    return class_weights


def replace_masked_values(tensor, mask, replace_with):
    assert tensor.dim() == mask.dim(), '{} vs {}'.format(tensor.shape, mask.shape)
    one_minus_mask = 1 - mask
    values_to_add = replace_with * one_minus_mask
    return tensor * mask + values_to_add


class SoftmaxEQL(object):
    def __init__(self, opt):
        self.opt = opt
        self.class_weight = torch.Tensor(get_eql_class_weights(self.opt)).cuda()

    def __call__(self, input, target):
        N, C = input.shape  # batch__size x number_node
        # reshape the class_weight array
        not_ignored = self.class_weight.view(1, C).repeat(N, 1)
        # generate list of random numbers and compare with a threshold
        over_prob = (torch.rand(input.shape).cuda() > self.opt.ignore_prob).float()
        is_gt = target.new_zeros((N, C)).float()
        # set the value at target index position to 1
        is_gt[torch.arange(N), target] = 1

        # if less than or equal to zero, then weight = 0
        weights = ((not_ignored + over_prob + is_gt) > 0).float()
        # replace with zero
        new_input = replace_masked_values(input, weights, -1e7)
        loss = F.cross_entropy(new_input, target)
        return loss
