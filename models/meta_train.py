"""
Build a simple meta-learning algorithm

Using explanations as inner loop optimizer

We need:
But instead of running a softmax matrix
We sample a new softmax matrix from pre-existing explanations?


"""
import os
import random
from collections import Counter, OrderedDict
import torch.optim as optim
import numpy as np
import torch
import math
from torch.autograd import Variable
from torchtext import data
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from sklearn import metrics
import logging

from util import ReversibleField, MultiLabelField

from os.path import join as pjoin

label_names = ['experimental-studies', 'allele-data', 'segregation-data', 'specificity-of-phenotype', 'case-control']

logging.basicConfig(filename="", format='[%(asctime)s] %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

class BaseConfig:
    emb_dim = 100
    hidden_size = 256
    depth = 1
    label_size = 5
    bidir = True
    dropout = 0.
    emb_update = True
    clip_grad = 5.
    seed = 1234
    meta_batch_size=15
    rand_unk = True
    run_name = "default"
    emb_corpus = "gigaword"

# TODO: Maybe we need to tune dtype but right now it's fine...
def np_to_var(np_obj, gpu_id=-1, requires_grad=False):
    if gpu_id == -1:
        return Variable(torch.from_numpy(np_obj), requires_grad=requires_grad)
    else:
        return Variable(torch.from_numpy(np_obj), requires_grad=requires_grad).cuda(gpu_id)

def to_cuda(obj, gpu_id):
    if gpu_id == -1:
        return obj
    else:
        return obj.cuda(gpu_id)

class MetaDataset(object):
    """this is simply an aggregate of 5 fields"""
    def __init__(self, data_path, batch_size=5, num_meta_labels=5, fix_length=None):
        """
        :param data_path: "./models/data/"
        :param batch_size: number of explanations to draw, let's say 5
        :param data_path: data should be in tsv format, and last label should be the grouping factor
        :param num_meta_labels:
        """
        self.num_meta_labels = num_meta_labels
        self.fix_length = fix_length
        self.batch_size = batch_size
        self.data_path = data_path

        self.TEXT_FIELD = ReversibleField(sequential=True, include_lengths=True, lower=False, fix_length=self.fix_length)
        # the vocab will be shared with the main text field in the main dataset

        self.datasets = []
        self.data_iters = []


    def load_meta_data(self, path_prefix='explanations_panel'):
        """
        :param path_prefix: "explanations_panel_{}"
        :return:
        """
        for i in range(self.num_meta_labels):
            dataset = data.TabularDataset(path=pjoin(self.data_path, path_prefix + "_{}.tsv".format(i)), format='tsv', fields=[('Explanation', self.TEXT_FIELD)])
            self.datasets.append(dataset)

    def load_data_iterators(self, device):
        def sort_w_error(x):
            if 'Explanation' in vars(x):
                return len(x.Explanation)
            else:
                setattr(x, 'Explanation', '<unk>')
                return 0

        for i in range(self.num_meta_labels):
            data_iter = data.Iterator(self.datasets[i], batch_size=self.batch_size, sort_key=sort_w_error,
                                      device=device, sort_within_batch=True, repeat=True)  # allow repeat
            self.data_iters.append(data_iter)

    def get_meta_data(self, i):
        return next(iter(self.data_iters[i]))

    def check_complete(self, i):
        # it only says epoch 1 after it finished epoch 1
        return self.data_iters[i].epoch >= 1

    def check_all_complete(self):
        all_complete = True
        for i in range(self.num_meta_labels):
            all_complete *= self.check_complete(i)
        return all_complete


class Dataset(object):
    def __init__(self, path='./data/',
                 dataset_prefix='vci_1543_abs_tit_key_apr_1_2019_',
                 test_data_name='',
                 full_meta_data_name='explanations_5panels.csv',
                 label_size=5, fix_length=None, meta_data=None):
        """
        :param meta_data: MetaData class instance. Will be used for vocab building.
        """
        # we will add metalabel here and make iterators
        self.TEXT = ReversibleField(sequential=True, include_lengths=True, lower=False, fix_length=fix_length)
        self.LABEL = MultiLabelField(sequential=True, use_vocab=False, label_size=label_size,
                                     tensor_type=torch.FloatTensor, fix_length=fix_length)

        # it's actually this step that will take 5 minutes
        self.train, self.val, self.test = data.TabularDataset.splits(
            path=path, train=dataset_prefix + 'train.csv',
            validation=dataset_prefix + 'valid.csv',
            test=dataset_prefix + 'test.csv', format='tsv',
            fields=[('Text', self.TEXT), ('Description', self.LABEL)])

        self.full_meta_data = data.TabularDataset(
            path=pjoin(path, full_meta_data_name),
            format='tsv', fields=[('Text', self.TEXT), ('Description', self.LABEL)]
        )

        self.meta_data = meta_data

        self.is_vocab_bulit = False
        self.iterators = []

        if test_data_name != '':
            self.external_test = data.TabularDataset(path=path + test_data_name,
                                                     format='tsv',
                                                     fields=[('Text', self.TEXT), ('Description', self.LABEL)])
        else:
            self.external_test = None

    def get_iterators(self, device, val_batch_size=128):
        if not self.is_vocab_bulit:
            raise Exception("Vocabulary is not built yet..needs to call build_vocab()")

        if len(self.iterators) > 0:
            return self.iterators  # return stored iterator

        # only get them after knowing the device (inside trainer or evaluator)
        train_iter, val_iter, test_iter = data.Iterator.splits(
            (self.train, self.val, self.test), sort_key=lambda x: len(x.Text),  # no global sort, but within-batch-sort
            batch_sizes=(32, val_batch_size, val_batch_size), device=device,
            sort_within_batch=True, repeat=False)

        return train_iter, val_iter, test_iter

    def xavier_uniform(self, tensor, fan_in, fan_out, gain=1):
        # fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        with torch.no_grad():
            return tensor.uniform_(-a, a)

    def init_emb(self, vocab, init="glorot", num_special_toks=2, silent=False):
        # we can try randn or glorot
        # mode="unk"|"all", all means initialize everything
        emb_vectors = vocab.vectors
        sweep_range = len(vocab)
        running_norm = 0.
        num_non_zero = 0
        total_words = 0

        fan_in, fan_out = emb_vectors.size()

        for i in range(num_special_toks, sweep_range):
            if len(emb_vectors[i, :].nonzero()) == 0:
                # std = 0.5 is based on the norm of average GloVE word vectors
                self.xavier_uniform(emb_vectors[i], fan_in, fan_out)
            else:
                num_non_zero += 1
                running_norm += torch.norm(emb_vectors[i])
            total_words += 1
        if not silent:
            print("average GloVE norm is {}, number of known words are {}, total number of words are {}".format(
                running_norm / num_non_zero, num_non_zero, total_words))  # directly printing into Jupyter Notebook

    def build_vocab(self, config, silent=False):
        if config.emb_corpus == 'common_crawl':
            self.TEXT.build_vocab(self.train, self.full_meta_data,
                                  vectors="glove.840B.300d")
            config.emb_dim = 300  # change the config emb dimension
        else:
            # add all datasets
            self.TEXT.build_vocab(self.train, self.full_meta_data,
                                  vectors="glove.6B.{}d".format(config.emb_dim))
        self.is_vocab_bulit = True
        self.vocab = self.TEXT.vocab
        if config.rand_unk:
            if not silent:
                print("initializing random vocabulary")
            self.init_emb(self.vocab, silent=silent)

        # synchronize vocab by making them the same object
        self.meta_data.TEXT_FIELD.vocab = self.TEXT.vocab

# Methods:
# 1. Shared encoder for both explanation and document
# 2. Different encoder
class MetaClassifier(nn.Module):
    def __init__(self, vocab, config):
        super(MetaClassifier, self).__init__()
        self.config = config
        self.drop = nn.Dropout(config.dropout)  # embedding dropout

        self.encoder = nn.LSTM(
                config.emb_dim,
                config.hidden_size,
                config.depth,
                dropout=config.dropout,
                bidirectional=config.bidir)  # ha...not even bidirectional
        d_out = config.hidden_size if not config.bidir else config.hidden_size * 2

        self.out_bias = nn.Parameter(torch.Tensor(config.label_size))
        # self.out = nn.Linear(d_out, config.label_size)  # include bias, to prevent bias assignment

        # self.feature_proj = nn.Sequential(
        #     nn.Linear(d_out, d_out),
        #     nn.SELU(),
        #     nn.Linear(d_out, d_out)
        # )

        self.embed = nn.Embedding(len(vocab), config.emb_dim)
        self.embed.weight.data.copy_(vocab.vectors)
        self.embed.weight.requires_grad = True if config.emb_update else False

    def forward(self, input, lengths=None, meta_input_tups=None):
        """
        :param meta_input_tups: (input, lengths), 5 of them; need to be arranged exactly by label order
        :return:
        """

        output_vecs = self.get_vectors(input, lengths)

        assert len(meta_input_tups) == 5

        meta_vecs = []
        for tup in meta_input_tups:
            vecs = self.get_vectors(tup[0], tup[1])
            # vecs: (time, batch, hid)
            # then max-pool over the temporal dim
            # vecs = torch.max(vecs, 0)[0].squeeze(0)
            vecs = torch.mean(vecs, 0)
            # then mean across batch dim
            vec = torch.mean(vecs, 0)
            meta_vecs.append(vec)

        return self.get_logits(output_vecs, meta_vecs)

    def get_vectors(self, input, lengths=None):
        embed_input = self.embed(input)

        packed_emb = embed_input
        if lengths is not None:
            lengths = lengths.view(-1).tolist()
            packed_emb = nn.utils.rnn.pack_padded_sequence(embed_input, lengths)

        output, hidden = self.encoder(packed_emb)  # embed_input

        if lengths is not None:
            output = unpack(output)[0]

        # we ignored negative masking
        return output

    def l2_vec(self, x, p):
        # x: (batch_size, hid_dim)
        # p: (hid_dim, class_size)
        return -(torch.norm(x, 2, dim=1).view(-1, 1).expand(-1, self.c) ** 2 + \
                 torch.norm(p, 2, dim=0).view(1, -1).expand(self.b, -1) ** 2 \
                 - 2 * torch.matmul(x, p))

    def l2_norm_vec(self, x, p):
        # unitize two vectors first
        x_norm = x / x.norm(2, dim=1)
        p_norm = p / p.norm(2, dim=0)
        return self.l2_vec(x_norm, p_norm)

    def cos_vec(self, x, p):
        return torch.matmul(x, p) / torch.ger(x.norm(2, dim=1), p.norm(2, dim=0))

    def dot_vec(self, x, p):
        return torch.matmul(x, p)

    def get_logits(self, output_vec, meta_vecs=None):
        output = torch.max(output_vec, 0)[0].squeeze(0)

        # output = self.feature_proj(output)

        proto_vecs = torch.stack(meta_vecs).t()  # (5, hid) -> (hid, 5)

        # output = torch.matmul(output, proto_vecs) + self.out_bias
        output = self.cos_vec(output, proto_vecs)

        return output


class Trainer(object):
    def __init__(self, classifier, dataset, config, save_path, device, load=False, run_order=0,
                 **kwargs):
        # save_path: where to save log and model
        self.classifier = classifier
        if load:
            # or we can add a new keyword...
            if os.path.exists(pjoin(save_path, 'model-{}.pickle'.format(run_order))):
                self.classifier = torch.load(pjoin(save_path, 'model-{}.pickle'.format(run_order))).cuda(device)
            else:
                self.classifier = torch.load(pjoin(save_path, 'model.pickle')).cuda(device)

        if device != -1:
            self.classifier = classifier.cuda(device)

        # replace old cached config with new config
        self.classifier.config = config

        self.dataset = dataset
        self.device = device
        self.config = config
        self.save_path = save_path

        self.train_iter, self.val_iter, self.test_iter = self.dataset.get_iterators(device)

        if self.dataset.external_test is not None:
            self.external_test_iter = self.dataset.get_test_iterator(device)

        self.bce_logit_loss = nn.BCEWithLogitsLoss()

        need_grad = lambda x: x.requires_grad
        self.optimizer = optim.Adam(
            filter(need_grad, classifier.parameters()),
            lr=0.001)  # obviously we could use config to control this

        # setting up logging
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
        file_handler = logging.FileHandler("{0}/log.txt".format(save_path))
        self.logger = logging.getLogger(save_path.split('/')[-1])  # so that no model is sharing logger
        self.logger.addHandler(file_handler)

        self.logger.info(config)

    def load(self, run_order):
        self.classifier = torch.load(pjoin(self.save_path, 'model-{}.pickle').format(run_order)).cuda(self.device)

    def train(self, run_order=0, epochs=5):
        # train loop
        exp_cost = None
        for e in range(epochs):
            self.classifier.train()
            for iter, data in enumerate(self.train_iter):
                self.classifier.zero_grad()
                (x, x_lengths), y = data.Text, data.Description

                # we sample explanations
                meta_data_tups = []
                for i in range(self.dataset.meta_data.num_meta_labels):
                    meta_x, meta_length = self.dataset.meta_data.get_meta_data(i).Explanation
                    if self.device != -1:
                        meta_x = meta_x.cuda(self.device)
                        meta_length = meta_length.cuda(self.device)
                    meta_data_tups.append((meta_x, meta_length))

                if self.device != -1:
                    x = x.cuda(self.device)
                    x_lengths = x_lengths.cuda(self.device)
                    y = y.cuda(self.device)

                # output_vec = self.classifier.get_vectors(x, x_lengths)  # this is just logit (before calling sigmoid)
                # final_rep = torch.max(output_vec, 0)[0].squeeze(0)
                # logits = self.classifier.get_logits(output_vec)

                logits = self.classifier(x, x_lengths, meta_data_tups)

                # batch_size = x.size(0)

                loss = self.bce_logit_loss(logits, y)
                loss.backward()

                torch.nn.utils.clip_grad_norm(self.classifier.parameters(), self.config.clip_grad)
                self.optimizer.step()

                if not exp_cost:
                    exp_cost = loss.data[0]
                else:
                    exp_cost = 0.99 * exp_cost + 0.01 * loss.data[0]

                if iter % 1 == 0:
                    self.logger.info(
                        "iter {} lr={} train_loss={} exp_cost={} \n".format(iter, self.optimizer.param_groups[0]['lr'],
                                                                            loss.data[0], exp_cost))
            self.logger.info("enter validation...")
            valid_em, micro_tup, macro_tup = self.evaluate(is_test=False)
            self.logger.info("epoch {} lr={:.6f} train_loss={:.6f} valid_acc={:.6f}\n".format(
                e + 1, self.optimizer.param_groups[0]['lr'], loss.data[0], valid_em
            ))

        # save model
        torch.save(self.classifier, pjoin(self.save_path, 'model-{}.pickle'.format(run_order)))

    def test(self, silent=False, return_by_label_stats=False, return_instances=False):
        self.logger.info("compute test set performance...")
        return self.evaluate(is_test=True, silent=silent, return_by_label_stats=return_by_label_stats,
                             return_instances=return_instances)

    def evaluate(self, is_test=False, is_external=False, silent=False, return_by_label_stats=False,
                 return_instances=False, return_roc_auc=False):
        self.classifier.eval()
        data_iter = self.test_iter if is_test else self.val_iter  # evaluate on CSU
        data_iter = self.external_test_iter if is_external else data_iter  # evaluate on adobe

        all_preds, all_y_labels, all_confs = [], [], []

        label_index = 5

        for iter, data in enumerate(data_iter):
            (x, x_lengths), y = data.Text, data.Description

            # TODO: added meta explanations
            meta_data_tups = []
            for i in range(self.dataset.meta_data.num_meta_labels):
                meta_x, meta_length = self.dataset.meta_data.get_meta_data(i).Explanation
                if self.device != -1:
                    meta_x = meta_x.cuda(self.device)
                    meta_length = meta_length.cuda(self.device)
                meta_data_tups.append((meta_x, meta_length))

            logits = self.classifier(x, x_lengths, meta_data_tups)

            preds = (torch.sigmoid(logits) > 0.5).data.cpu().numpy().astype(float)
            all_preds.append(preds)
            all_y_labels.append(y.data.cpu().numpy())
            all_confs.append(torch.sigmoid(logits).data.cpu().numpy().astype(float))

        preds = np.vstack(all_preds)[:, :label_index]
        ys = np.vstack(all_y_labels)[:, :label_index]
        confs = np.vstack(all_confs)[:, :label_index]

        # this code works :)
        roc_auc = np.zeros(ys.shape[1])
        roc_auc[:] = 0.5  # base value for ROC AUC
        non_zero_label_idices = ys.sum(0).nonzero()

        non_zero_ys = np.squeeze(ys[:, non_zero_label_idices])
        non_zero_preds = np.squeeze(preds[:, non_zero_label_idices])
        non_zero_roc_auc = metrics.roc_auc_score(non_zero_ys, non_zero_preds, average=None)

        roc_auc[non_zero_label_idices] = non_zero_roc_auc

        if not silent:
            self.logger.info("\n" + metrics.classification_report(ys, preds, digits=3))  # write to file
            self.logger.info("\n ROC-AUC: {}".format(roc_auc))

        # this is actually the accurate exact match
        em = metrics.accuracy_score(ys, preds)
        accu = np.array([metrics.accuracy_score(ys[:, i], preds[:, i]) for i in range(self.config.label_size)],
                        dtype='float32')
        p, r, f1, s = metrics.precision_recall_fscore_support(ys, preds, average=None)

        # because some labels are NOT present in the test set, we need to message this function
        # filter out labels that have no examples

        if return_by_label_stats and return_roc_auc:
            return p, r, f1, s, accu, roc_auc
        elif return_by_label_stats:
            return p, r, f1, s, accu
        elif return_instances:
            return ys, preds, confs

        micro_p, micro_r, micro_f1 = np.average(p, weights=s), np.average(r, weights=s), np.average(f1, weights=s)

        # compute Macro-F1 here
        # if is_external:
        #     # include clinical finding
        #     macro_p, macro_r, macro_f1 = np.average(p[14:]), np.average(r[14:]), np.average(f1[14:])
        # else:
        #     # anything > 10
        #     macro_p, macro_r, macro_f1 = np.average(np.take(p, [12] + range(21, 42))), \
        #                                  np.average(np.take(r, [12] + range(21, 42))), \
        #                                  np.average(np.take(f1, [12] + range(21, 42)))

        # we switch to non-zero macro computing, this can figure out boost from rarest labels
        if is_external:
            # include clinical finding
            macro_p, macro_r, macro_f1 = np.average(p[p.nonzero()]), np.average(r[r.nonzero()]), \
                                         np.average(f1[f1.nonzero()])
        else:
            # anything > 10
            macro_p, macro_r, macro_f1 = np.average(p[p.nonzero()]), \
                                         np.average(r[r.nonzero()]), \
                                         np.average(f1[f1.nonzero()])

        return em, (micro_p, micro_r, micro_f1), (macro_p, macro_r, macro_f1)

import sys
if __name__ == '__main__':
    random.seed(1234)
    gpu_device = 0

    config = BaseConfig()
    meta_dataset = MetaDataset("./data",
                               batch_size=config.meta_batch_size)
    dataset = Dataset(meta_data=meta_dataset)
    dataset.build_vocab(config, False)

    meta_dataset.load_meta_data()
    meta_dataset.load_data_iterators(gpu_device)

    classifier = MetaClassifier(dataset.vocab, config)
    logging.info(classifier)

    trainer = Trainer(classifier, dataset, config,
                      save_path='./saved/meta_exp_apr7_2019/',
                      device=gpu_device, load=False, run_order=0)

    trainer.train(0, 5)

    em, micro_tup, macro_tup = trainer.test()
    logging.info("test acc: {}".format(em))
