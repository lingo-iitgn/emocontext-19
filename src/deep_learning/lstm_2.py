import csv
import io

import numpy as np
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.tokenize import TweetTokenizer
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from tensorboardX import SummaryWriter
from torch import optim
from torch.nn import Hardtanh
from torchtext import data

# from keras.utils import to_categorical
print(torch.__version__)


def custom_tokenizer(line):
    line = "<sol> " + line + " <eol>"
    repeatedChars = ['.', '?', '!', ',']
    for c in repeatedChars:
        lineSplit = line.split(c)
        while True:
            try:
                lineSplit.remove('')
            except:
                break
        cSpace = ' ' + c + ' '
        line = cSpace.join(lineSplit)
    line = line.strip().split(' ')[1:-1]
    while True:
        try:
            line.remove('')
        except:
            break
    return line


class encoder(nn.Module):
    '''Base LSTM model for 2nd LSTM architecture. It contains
        * Embedding Layer initilised by Glove
        * Mutiple LSTM layers  
    '''

    def __init__(self, VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, ENC_OUTPUT_DIM, ENC_N_LAYERS, BIDIRECTIONAL):
        super().__init__()
        self.hidden_size = HIDDEN_DIM
        self.init_hid = 2 * ENC_N_LAYERS if BIDIRECTIONAL else ENC_N_LAYERS
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.embedding.weight.data.copy_(src.vocab.vectors).detach()
        self.rnn = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, num_layers=ENC_N_LAYERS, bidirectional=BIDIRECTIONAL)

    def forward(self, x, hidden=None):
        if hidden == None:
            hidden = self.initHidden(x.shape[1])
            cell = self.initHidden(x.shape[1])
            hidden = (hidden, cell)
        # x.shape = [sent len, batch size]

        embedded = self.embedding(x)
        # embedded.shape = [sent len, batch size, emb dim]

        output, hidden = self.rnn(embedded, hidden)
        # output = [sent len, batch size, hid dim * num directions]
        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]
        return output, hidden

    def initHidden(self, BATCH_SIZE):
        return torch.randn(self.init_hid, BATCH_SIZE, HIDDEN_DIM, device=device)


class classifier(nn.Module):
    '''Creates the architecture described in the paper, using the `encoder` class.
    '''

    def __init__(self, VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, ENC_OUTPUT_DIM, ENC_N_LAYERS, BIDIRECTIONAL,
                 OUTPUT_CLASSES):
        super().__init__()
        self.init_hid = 2 * HIDDEN_DIM if BIDIRECTIONAL else HIDDEN_DIM
        self.enc = encoder(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, ENC_OUTPUT_DIM, ENC_N_LAYERS, BIDIRECTIONAL)
        self.reducer = nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM)
        self.out = nn.Linear(HIDDEN_DIM * 2, OUTPUT_CLASSES)

    def forward(self, turn1, turn2, turn3):
        out1, hidden1 = self.enc(turn1)
        out2, hidden2 = self.enc(turn2, hidden1)
        # out1 = (max_length, batch size, hidden dim)
        # hidden1 = (num layers, batch size, hidden dim)

        out_1_2 = torch.cat((hidden1[0][1:], hidden2[0][1:]), 2)
        # out_1_2 = (max_len, batch_size, hidden dim*2)

        reduce_1_2 = F.relu(self.reducer(out_1_2))
        # reduce_1_2 = (max len, batch size, hidden dim)

        out3, hidden3 = self.enc(turn3, hidden2)

        out_1_2_3 = torch.cat((reduce_1_2, hidden3[0][1:]), 2)
        # out_1_2_3 = (max len, batch size, hid dim*2)

        result = self.out(out_1_2_3[0, :, :])

        result = F.log_softmax(result, dim=1)
        return result


def val_score(iterator, name='valid'):
    '''Calculates the metric necessary for the evaluation including
        * Precision (test set) for all emotions
        * Recall (test set) for all emotions
        * micro F1 for the emotions for validation and test set
    writes the calculated metric to tensorboard
    '''
    pred_tens = torch.zeros((1, 4), dtype=torch.float).to(device)
    gold_tens = torch.zeros((1,), dtype=torch.long).to(device)
    eval_loss = 0

    for eval_batch in iterator:
        emo_class.eval()
        eval_pred = emo_class(eval_batch.turn1, eval_batch.turn2, eval_batch.turn3)
        # Collecting the predictions
        pred_tens = torch.cat([pred_tens, eval_pred])
        gold_tens = torch.cat([gold_tens, eval_batch.label])

        eval_loss += criterion(eval_pred, eval_batch.label).item()

    writer.add_scalar('{} Loss'.format(name), eval_loss / len(iter_valid), iteration)

    log_prob, indx_eval = pred_tens.topk(1)
    microf1_eval = f1_score(gold_tens.detach().cpu(), indx_eval.cpu().detach(), average='micro', labels=[1, 2, 3])
    writer.add_scalar('{} microF1'.format(name), microf1_eval, iteration)
    print('Dev set Micro F1:', microf1_eval)
    if name == 'test':
        print('Precision on Test:\n {0}'.format(precision_score(gold_tens.detach().cpu(),
                                                                indx_eval.cpu().detach(),
                                                                average=None)))
        print('Recall on Test:\n {0}'.format(recall_score(gold_tens.detach().cpu(),
                                                          indx_eval.cpu().detach(),
                                                          average=None)))


if __name__ == "__main__":

    ####### Initilizers #######
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 25
    VOCAB_SIZE = 10000
    EMBEDDING_DIM = 100
    ENC_N_LAYERS = 2
    BIDIRECTIONAL = False
    HIDDEN_DIM = 256
    MAX_LENGTH = 20
    ENC_OUTPUT_DIM = 256
    KAPPA = 100
    norm_eps = 0.1
    eps = norm_eps
    norm_max = 14.0
    DROPOUT = 0
    learning_rate = 0.00035
    MAX_ITER = 5500

    ####### DATA READING #######
    src = data.Field(tokenize=custom_tokenizer,
                     init_token='<sos>',
                     eos_token='<eos>',
                     lower=True,
                     stop_words=['.', '!', '?', '"'],
                     fix_length=MAX_LENGTH)
    tgt = data.Field(sequential=False, is_target=True, unk_token=None)
    ids = data.Field(sequential=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dir_emo = '../../data/'

    train_emo, val_emo, test_emo = data.TabularDataset.splits(
        path=dir_emo,
        train='train.txt',
        validation='dev.txt',
        test='test.txt',
        skip_header=True,
        format='tsv',
        fields=[('id', ids), ('turn1', src), ('turn2', src), ('turn3', src), ('label', tgt)],
        csv_reader_params={'quoting': csv.QUOTE_NONE})

    ####### CREATING ITERATORS FOR TRAIN, VAL, TEST #######
    iter_train = data.BucketIterator(train_emo, batch_size=TRAIN_BATCH_SIZE, train=True, repeat=True, shuffle=True,
                                     device=device)
    iter_valid = data.BucketIterator(val_emo, batch_size=VALID_BATCH_SIZE, train=True, shuffle=True, device=device)
    iter_test = data.BucketIterator(test_emo, batch_size=VALID_BATCH_SIZE, train=True, shuffle=True, device=device)
    print('Dataset loaded!')

    ####### CREATING VOCAB #######
    if EMBEDDING_DIM not in [50, 100, 200, 300]:
        print('choose `EMBEDDING_DIM` to be one of [50, 100, 200, 300]')
        print('using the default 100d')
        pretrained_embedding = 'glove.6B.{}d'.format(100)
    else:
        pretrained_embedding = 'glove.6B.{}d'.format(EMBEDDING_DIM)

    src.build_vocab(train_emo, max_size=VOCAB_SIZE, vectors=pretrained_embedding)
    tgt.build_vocab(train_emo)
    ids.build_vocab(train_emo)
    print('Training Dataset Distrbution of Emotions:\n', tgt.vocab.freqs.most_common(5))

    ####### ARCITECTURE INIT #######
    emo_class = classifier(len(src.vocab), EMBEDDING_DIM, HIDDEN_DIM, ENC_OUTPUT_DIM, ENC_N_LAYERS, BIDIRECTIONAL,
                           len(tgt.vocab)).to(device)
    criterion = nn.NLLLoss().to(device)
    optimiser = optim.Adam(emo_class.parameters(), lr=learning_rate)
    print('New model parameters initilised!')

    ####### MODEL TRAINING #######
    writer = SummaryWriter('./LSTM_2/')
    print('Tensorboard logging at `./LSTM_2/` for loss and micro F1 score on train and validation set.')
    eval_iteration_total = len(iter_valid)

    for iteration, batch in enumerate(iter_train):
        emo_class.train()
        optimiser.zero_grad()
        pred = emo_class(batch.turn1, batch.turn2, batch.turn3)
        loss = criterion(pred, batch.label)

        loss.backward()
        torch.nn.utils.clip_grad_value_(emo_class.parameters(), 3)
        optimiser.step()

        if iteration % 100 == 0:
            writer.add_scalar('Loss', loss.item(), iteration)
            log_prob, indx = pred.topk(1)
            microf1 = f1_score(batch.label.detach().cpu(), indx.cpu().detach(), average='micro', labels=[1, 2, 3])
            writer.add_scalar('MicroF1', microf1, iteration)
            eval_loss = 0

            val_score(iter_valid)

        if iteration == MAX_ITER:
            val_score(iter_test, name='test')
            break

    torch.save({'model': emo_class.state_dict()}, './LSTM_2/final_model.ckt')
