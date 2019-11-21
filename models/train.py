import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torchtext import data
from os.path import join as pjoin
import os
import logging
import numpy as np
from sklearn import metrics
import random
import math

from pathlib import Path

from argparse import ArgumentParser

from IPython import embed

from util import ReversibleField, MultiLabelField

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

class Dataset(object):
    def __init__(self, path='./data/',
                 weak_train_dataset="",
                 acmg_weak_data_path="",
                 dataset_prefix='vci_1543_abs_tit_key_apr_1_2019_',
                 test_data_name='vci_358_abs_tit_key_may_7_2019_true_test.csv',
                 multi_task_train_dataset="",
                 label_size=5, fix_length=None):
        self.TEXT = ReversibleField(sequential=True, include_lengths=True, lower=False, fix_length=fix_length)
        self.LABEL = MultiLabelField(sequential=True, use_vocab=False, label_size=label_size,
                                     tensor_type=torch.FloatTensor, fix_length=fix_length)

        if weak_train_dataset != "":
            self.weak_train = data.TabularDataset(weak_train_dataset, format='tsv',
                                                  fields=[('Text', self.TEXT), ('Description', self.LABEL)])
            if acmg_weak_data_path != "":
                acmg_weak_data = data.TabularDataset(acmg_weak_data_path, format='tsv',
                                                  fields=[('Text', self.TEXT), ('Description', self.LABEL)])
                # this should be enough!
                self.weak_train.examples.extend(acmg_weak_data.examples)
        else:
            self.weak_train = None

        if multi_task_train_dataset != "":
            self.multi_task_train = data.TabularDataset(multi_task_train_dataset, format='tsv',
                                                        fields=[('Text', self.TEXT), ('Description', self.LABEL)])
        else:
            self.multi_task_train = None

        # it's actually this step that will take 5 minutes
        self.train, self.val, self.test = data.TabularDataset.splits(
            path=path, train=dataset_prefix + 'train.csv',
            validation=dataset_prefix + 'valid.csv',
            test=dataset_prefix + 'test.csv', format='tsv',
            fields=[('Text', self.TEXT), ('Description', self.LABEL)])

        if test_data_name != '':
            self.external_test = data.TabularDataset(path=path + test_data_name,
                                                     format='tsv',
                                                     fields=[('Text', self.TEXT), ('Description', self.LABEL)])
        else:
            self.external_test = None

        self.is_vocab_bulit = False
        self.iterators = []
        self.test_iterator = None
        self.weak_train_iterator = None
        self.multi_task_train_iterator = None

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

        fan_in, fan_out = emb_vectors.size()  # 16870, 300 # std = 0.01 # a = 1.73 * 0.01

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
        datasets = [self.train]
        if self.weak_train is not None and args.weak_vocab:
            datasets.append(self.weak_train)

        if self.multi_task_train is not None:
            datasets.append(self.multi_task_train)  # we always build vocab for multitask

        if config.emb_corpus == 'common_crawl':
            # self.TEXT.build_vocab(self.train, vectors="glove.840B.300d")
            self.TEXT.build_vocab(*datasets, vectors="glove.840B.300d")
            config.emb_dim = 300  # change the config emb dimension
        else:
            self.TEXT.build_vocab(*datasets, vectors="glove.6B.{}d".format(config.emb_dim))

        self.is_vocab_bulit = True
        self.vocab = self.TEXT.vocab
        if config.rand_unk:
            if not silent:
                print("initializing random vocabulary")
            self.init_emb(self.vocab, silent=silent)

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

    def get_test_iterator(self, device):
        if not self.is_vocab_bulit:
            raise Exception("Vocabulary is not built yet..needs to call build_vocab()")

        if self.test_iterator is not None:
            return self.test_iterator

        external_test_iter = data.Iterator(self.external_test, 128, sort_key=lambda x: len(x.Text),
                                           device=device, train=False, repeat=False, sort_within_batch=True)
        return external_test_iter

    def get_weak_train_iterator(self, device):
        if not self.is_vocab_bulit:
            raise Exception("Vocabulary is not built yet..needs to call build_vocab()")

        if self.weak_train_iterator is not None:
            return self.weak_train_iterator

        weak_train_iterator = data.Iterator(self.weak_train, 128, sort_key=lambda x: len(x.Text),
                                           device=device, train=True, repeat=False, sort_within_batch=True)

        return weak_train_iterator

    def get_multi_task_train_iterator(self, device):
        if not self.is_vocab_bulit:
            raise Exception("Vocabulary is not built yet..needs to call build_vocab()")

        if self.multi_task_train_iterator is not None:
            return self.multi_task_train_iterator

        self.multi_task_train_iterator = data.Iterator(self.multi_task_train, 128, sort_key=lambda x: len(x.Text),
                                        device=device, train=True, repeat=False, sort_within_batch=True)

        return self.multi_task_train_iterator

class BaseConfig:
    emb_dim = 100
    hidden_size = 256
    depth = 1
    label_size = 5
    bidir = True
    dropout = 0.
    emb_update = True
    clip_grad = 5.
    seed = 1234,
    rand_unk = True
    run_name = "default"
    emb_corpus = "gigaword"
    share_decoder = False


class Classifier(nn.Module):
    def __init__(self, vocab, config):
        super(Classifier, self).__init__()
        self.config = config
        self.drop = nn.Dropout(config.dropout)  # embedding dropout

        self.encoder = nn.LSTM(
                config.emb_dim,
                config.hidden_size,
                config.depth,
                dropout=config.dropout,
                bidirectional=config.bidir)  # ha...not even bidirectional
        d_out = config.hidden_size if not config.bidir else config.hidden_size * 2

        self.out = nn.Linear(d_out, config.label_size)  # include bias, to prevent bias assignment
        self.embed = nn.Embedding(len(vocab), config.emb_dim)
        self.embed.weight.data.copy_(vocab.vectors)
        self.embed.weight.requires_grad = True if config.emb_update else False

        if not config.share_decoder:
            self.multi_task_out = nn.Linear(d_out, config.label_size)

    def forward(self, input, lengths=None):
        output_vecs = self.get_vectors(input, lengths)
        return self.get_logits(output_vecs)

    def multi_task_forward(self, input, lengths=None):
        output_vecs = self.get_vectors(input, lengths)
        output = torch.max(output_vecs, 0)[0].squeeze(0)
        return self.multi_task_out(output)

    def get_vectors(self, input, lengths=None):

        pad_embed = self.embed(input)  # input should be padded already...
        # embed_input = self.embed(input)

        # if self.config.conv_enc:
        #     output = self.encoder((embed_input, lengths.view(-1).tolist()))
        #     return output

        pad_embed_pack = pad_embed
        if lengths is not None:
            # lens = list(map(len, input))
            pad_embed_pack = pack_padded_sequence(pad_embed, lengths)

        output, hidden = self.encoder(pad_embed_pack)  # embed_input

        if lengths is not None:
            # output = unpack(output)[0]
            output = pad_packed_sequence(output)[0]

        # we ignored negative masking
        return output

    # def get_vectors(self, input, lengths=None):
    #     embed_input = self.embed(input)
    #
    #     # if self.config.conv_enc:
    #     #     output = self.encoder((embed_input, lengths.view(-1).tolist()))
    #     #     return output
    #
    #     packed_emb = embed_input
    #     if lengths is not None:
    #         lengths = lengths.view(-1).tolist()
    #         packed_emb = nn.utils.rnn.pack_padded_sequence(embed_input, lengths)
    #
    #     output, hidden = self.encoder(packed_emb)  # embed_input
    #
    #     if lengths is not None:
    #         output = unpack(output)[0]
    #
    #     # we ignored negative masking
    #     return output

    def get_logits(self, output_vec):
        output = torch.max(output_vec, 0)[0].squeeze(0)
        return self.out(output)

    def get_softmax_weight(self):
        return self.out.weight


class Trainer(object):
    def __init__(self, classifier, dataset, config, save_path, device, load=False, run_order=0,
                 **kwargs):
        # save_path: where to save log and model
        if load:
            # or we can add a new keyword...
            if os.path.exists(pjoin(save_path, 'model-{}.pickle'.format(run_order))):
                self.classifier = torch.load(pjoin(save_path, 'model-{}.pickle'.format(run_order))).cuda(device)
            else:
                self.classifier = torch.load(pjoin(save_path, 'model.pickle')).cuda(device)
        else:
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
        else:
            self.external_test_iter = None

        if self.dataset.weak_train is not None:
            self.weak_train_iter = self.dataset.get_weak_train_iterator(device)
        else:
            self.weak_train_iter = None

        if self.dataset.multi_task_train is not None:
            self.multi_task_train_iter = self.dataset.get_multi_task_train_iterator(device)
        else:
            self.multi_task_train_iter = None

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

    def weak_train(self):
        # currently this is only for one epoch
        exp_cost = None
        for iter, data in enumerate(self.weak_train_iter):
            self.classifier.zero_grad()
            (x, x_lengths), y = data.Text, data.Description
            if self.device != -1:
                x = x.cuda(self.device)
                x_lengths = x_lengths.cuda(self.device)
                y = y.cuda(self.device)

            if args.share_decoder:
                logits = self.classifier(x, x_lengths)
            else:
                logits = self.classifier.multi_task_forward(x, x_lengths)  # this can be a "weak" data head as well

            if args.weak_label_smoothing_esp == 0.0:
                loss = self.bce_logit_loss(logits, torch.squeeze(y)) * args.weak_loss_scale
            else:
                # weak_label_smoothing_esp (may or may not be correct...)
                loss = self.bce_logit_loss(logits, torch.squeeze(y) - args.weak_label_smoothing_esp) * args.weak_loss_scale

            loss.backward()

            torch.nn.utils.clip_grad_norm(self.classifier.parameters(), self.config.clip_grad)
            self.optimizer.step()

            if not exp_cost:
                exp_cost = loss.data.item()
            else:
                exp_cost = 0.99 * exp_cost + 0.01 * loss.data.item()

            if iter % 1 == 0:
                self.logger.info(
                    "weak iter {} lr={} train_loss={} exp_cost={} \n".format(iter, self.optimizer.param_groups[0]['lr'],
                                                                        loss.data.item(), exp_cost))

    def multi_task_train(self):
        # currently this is only for one epoch
        exp_cost = None
        for iter, data in enumerate(self.multi_task_train_iter):
            self.classifier.zero_grad()
            (x, x_lengths), y = data.Text, data.Description
            if self.device != -1:
                x = x.cuda(self.device)
                x_lengths = x_lengths.cuda(self.device)
                y = y.cuda(self.device)

            if classifier.config.share_decoder:
                logits = self.classifier(x, x_lengths)
            else:
                logits = self.classifier.multi_task_forward(x, x_lengths)

            if args.mtl_label_smoothing_esp == 0:
                loss = self.bce_logit_loss(logits, y) * args.mtl_loss_scale
            else:
                loss = self.bce_logit_loss(logits, y - args.mtl_label_smoothing_esp) * args.mtl_loss_scale
            loss.backward()

            torch.nn.utils.clip_grad_norm(self.classifier.parameters(), self.config.clip_grad)
            self.optimizer.step()

            if not exp_cost:
                exp_cost = loss.data.item()
            else:
                exp_cost = 0.99 * exp_cost + 0.01 * loss.data.item()

            if iter % 1 == 0:
                self.logger.info(
                    "multitask iter {} lr={} train_loss={} exp_cost={} \n".format(iter, self.optimizer.param_groups[0]['lr'],
                                                                             loss.data.item(), exp_cost))

    def train(self, run_order=0, epochs=5):
        # train loop
        exp_cost = None
        for e in range(epochs):
            self.classifier.train()

            if self.multi_task_train_iter is not None:
                self.multi_task_train()

            if self.weak_train_iter is not None:
                self.weak_train()

            for iter, data in enumerate(self.train_iter):
                self.classifier.zero_grad()
                (x, x_lengths), y = data.Text, data.Description
                if self.device != -1:
                    x = x.cuda(self.device)
                    x_lengths = x_lengths.cuda(self.device)
                    y = y.cuda(self.device)

                # output_vec = self.classifier.get_vectors(x, x_lengths)  # this is just logit (before calling sigmoid)
                # final_rep = torch.max(output_vec, 0)[0].squeeze(0)
                # logits = self.classifier.get_logits(output_vec)

                logits = self.classifier(x, x_lengths)

                # batch_size = x.size(0)

                loss = self.bce_logit_loss(logits, y)
                loss.backward()

                torch.nn.utils.clip_grad_norm(self.classifier.parameters(), self.config.clip_grad)
                self.optimizer.step()

                if not exp_cost:
                    exp_cost = loss.data.item()
                else:
                    exp_cost = 0.99 * exp_cost + 0.01 * loss.data.item()

                if iter % 1 == 0:
                    self.logger.info(
                        "iter {} lr={} train_loss={} exp_cost={} \n".format(iter, self.optimizer.param_groups[0]['lr'],
                                                                            loss.data.item(), exp_cost))

            # if self.multi_task_train_iter is not None:
            #     self.multi_task_train()
            #
            # if self.weak_train_iter is not None:
            #     self.weak_train()

            self.logger.info("enter validation...")
            valid_em, _, micro_tup, macro_tup = self.evaluate(is_test=False)
            self.logger.info("epoch {} lr={:.6f} train_loss={:.6f} valid_acc={:.6f}\n".format(
                e + 1, self.optimizer.param_groups[0]['lr'], loss.data.item(), valid_em
            ))

        # save model
        # TODO: upgrade this to avoid Python dependency...
        torch.save(self.classifier, pjoin(self.save_path, 'model-{}.pickle'.format(run_order)))

    def test(self, silent=False, return_by_label_stats=False, return_instances=False, is_external=False):
        self.logger.info("compute test set performance...")
        return self.evaluate(is_test=True, silent=silent, return_by_label_stats=return_by_label_stats,
                             return_instances=return_instances, is_external=is_external)

    def evaluate(self, is_test=False, is_external=False, silent=False, return_by_label_stats=False,
                 return_instances=False, return_roc_auc=False):
        self.classifier.eval()
        data_iter = self.test_iter if is_test else self.val_iter  # evaluate on CSU
        data_iter = self.external_test_iter if is_external else data_iter  # evaluate on adobe

        all_preds, all_y_labels, all_confs = [], [], []

        label_index = 5

        for iter, data in enumerate(data_iter):
            (x, x_lengths), y = data.Text, data.Description
            logits = self.classifier(x, x_lengths)

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

        return em, accu, (micro_p, micro_r, micro_f1), (macro_p, macro_r, macro_f1)

import csv

# Move Experiment class from DeepTag
# Ehhh...I forgot how the folders are arranged :(
# Bad code design!!!
class Experiment(object):
    def __init__(self, dataset, exp_save_path):
        """
        :param dataset: Dataset class
        :param exp_save_path: the overall saving folder
        """
        if not os.path.exists(exp_save_path):
            os.makedirs(exp_save_path)

        self.dataset = dataset
        self.exp_save_path = exp_save_path
        self.saved_random_states = [49537527, 50069528, 44150907, 25982144, 12302344,
                                    49537527, 50069528, 44150907, 25982144, 12302344]
        # Cuda will add additional randomness into this...same set of seeds won't matter

        header = ['model',
                 'Top-1 EM', 'Top-1 micro-P', 'Top-1 micro-R', 'Top-1 micro-F1',
                 'Top-1 macro-P', 'Top-1 macro-R', 'Top-1 macro-F1']

        # we never want to overwrite this file
        if not os.path.exists(pjoin(exp_save_path, "all_runs_stats.csv")):
            with open(pjoin(self.exp_save_path, "all_runs_stats.csv"), 'w') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(['model',
                                     'Top-1 EM', 'Top-1 micro-P', 'Top-1 micro-R', 'Top-1 micro-F1',
                                     'Top-1 macro-P', 'Top-1 macro-R', 'Top-1 macro-F1'])


    def set_random_seed(self, config):
        seed = config.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(config.seed)  # need to seed cuda too

    def get_trainer(self, config, device, run_order=0, build_vocab=False, load=False, silent=True, **kwargs):
        # build each trainer and classifier by config; or reload classifier
        # **kwargs: additional commands for the two losses

        if build_vocab:
            self.dataset.build_vocab(config, silent)  # because we might try different word embedding size

        self.set_random_seed(config)

        classifier = Classifier(self.dataset.vocab, config)
        logging.info(classifier)
        trainer_folder = args.exp_path
        # trainer = Trainer(classifier, self.dataset, config,
        #                   save_path=pjoin(self.exp_save_path, trainer_folder),
        #                   device=device, load=load, run_order=run_order, **kwargs)

        trainer = Trainer(classifier, dataset, config,
                          save_path=trainer_folder,  # ./saved/init_exp_apr1_2019/
                          device=device, load=load, run_order=run_order, **kwargs)

        return trainer

    def record_meta_result(self, meta_results, append, config, file_name='all_runs_stats.csv'):
        # this records result one line at a time!
        mode = 'a' if append else 'w'
        model_str = self.config_to_string(config)

        # TODO: this is wrong...change it!
        csu_em, csu_micro_tup, csu_macro_tup, \
        pp_em, pp_micro_tup, pp_macro_tup = meta_results

        with open(pjoin(self.exp_save_path, file_name), mode=mode) as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([model_str, csu_em, csu_micro_tup[0],
                                 csu_micro_tup[1], csu_micro_tup[2],
                                 csu_macro_tup[0], csu_macro_tup[1], csu_macro_tup[2],
                                 pp_em, pp_micro_tup[0], pp_micro_tup[1], pp_micro_tup[2],
                                 pp_macro_tup[0], pp_macro_tup[1], pp_macro_tup[2]])


parser = ArgumentParser()
parser.add_argument('--exp_path', type=str, default="")
parser.add_argument('--dataset_prefix', type=str, default="vci_1543_abs_tit_key_apr_1_2019_")
parser.add_argument('--hid_dim', type=int, default=256)
parser.add_argument('--emb_dim', type=int, default=100)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--emb_corpus', type=str, default="gigaword")
parser.add_argument('--weak_data_path', type=str, default="")
parser.add_argument('--acmg_weak_data_path', type=str, default="")
parser.add_argument('--weak_vocab', action='store_true', help="Build joint vocab on both training and weak training set")
parser.add_argument('--weak_loss_scale', type=float, default=0.3, help="Build joint vocab on both training and weak training set")
parser.add_argument('--mtl_loss_scale', type=float, default=0.3, help="Build joint vocab on both training and weak training set")
parser.add_argument('--weak_label_smoothing_esp', type=float, default=0., help="adjust uncertainty in source")
parser.add_argument('--mtl_label_smoothing_esp', type=float, default=0., help="adjust uncertainty in source")
parser.add_argument('--epochs', type=int, default=7, help="Number of epoch. Weak supervision requires longer epoch.")
parser.add_argument('--weak_epochs', type=int, default=0, help="Only works when --mtl_first is flagged as true")
parser.add_argument('--multi_task_data_path', type=str, default="",
                    help="Add a training file for multi-task learning")
parser.add_argument("--share_decoder", action="store_true", help="share decoder between multi-task and main task")
parser.add_argument('--mtl_first', action='store_true', help="Joint train model with MTL, then train on weak")

args = parser.parse_args()

if __name__ == '__main__':
    random.seed(1234)
    torch.random.manual_seed(1234)

    # TODO 1: add label smoothing andmulti_task_data_path run it
    # TODO 2: train MTL (VCI + Exp) (no shared head) + unsupervised learning (and make this one work by tuning a lot!)
    # DONE! Just need to run it to get a good result...

    config = BaseConfig()
    config.hidden_size = args.hid_dim
    config.share_decoder = args.share_decoder
    config.emb_dim = args.emb_dim
    config.emb_corpus = args.emb_corpus
    config.dropout = args.dropout

    dataset = Dataset(dataset_prefix=args.dataset_prefix,
                      weak_train_dataset=args.weak_data_path,
                      acmg_weak_data_path=args.acmg_weak_data_path,
                      multi_task_train_dataset=args.multi_task_data_path)
    dataset.build_vocab(config, False)

    classifier = Classifier(dataset.vocab, config)
    logging.info(classifier)

    trainer = Trainer(classifier, dataset, config,
                      save_path=args.exp_path,  # ./saved/init_exp_apr1_2019/
                      device=0, load=False, run_order=0)

    if args.mtl_first:
        trainer.weak_train_iter = None

    trainer.train(0, args.epochs)  # 7 is the best epoch for highest validation

    em, accus, micro_tup, macro_tup = trainer.test()
    logging.info("internal test EM: {}".format(em))
    logging.info("internal by-label accuracy: {}".format(accus))
    logging.info("internal average accuracy: {}".format(np.mean(accus)))

    em, accus, micro_tup, macro_tup = trainer.test(is_external=True)
    logging.info("external test EM: {}".format(em))
    logging.info("external by-label accuracy: {}".format(accus))
    logging.info("external average accuracy: {}".format(np.mean(accus)))

    # MTL training done
    if args.mtl_first:
        print("---------- MTL training done ---------")

        # This might override the exp folder....
        trainer = Trainer(classifier, dataset, config,
                          save_path=args.exp_path,  # ./saved/init_exp_apr1_2019/
                          device=0, load=False, run_order=0)

        trainer.multi_task_train_iter = None  # reset this part
        args.mtl_first = False
        if args.weak_epochs == 0:
            args.weak_epochs = args.epochs

        trainer.train(0, args.weak_epochs)  # train the model jointly with Unsup data

        em, accus, micro_tup, macro_tup = trainer.test()
        logging.info("internal test EM: {}".format(em))
        logging.info("internal by-label accuracy: {}".format(accus))
        logging.info("internal average accuracy: {}".format(np.mean(accus)))

        em, accus, micro_tup, macro_tup = trainer.test(is_external=True)
        logging.info("external test EM: {}".format(em))
        logging.info("external by-label accuracy: {}".format(accus))
        logging.info("external average accuracy: {}".format(np.mean(accus)))


