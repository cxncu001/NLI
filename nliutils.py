'''
Created on Mar 22, 2018

@author: wucx
'''

import time
import pickle
import numpy as np
import tensorflow as tf
import tensorflow.contrib.keras as kr

from datetime import timedelta
from collections import Counter

UNKNOWN = '<<UNK>>'
PADDING = '<<PAD>>'
CATEGORIE_ID = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

def load_embeddings(embdding_path, vocab):
    """
    Load pre-trained word embeddings for words in the vocabulary
    :param embeddings_path: path of pre-trained embeddings file
    :param vocab: the vocabulary (word->id mapping)
    """
    with open(embdding_path, 'rb') as fin:
        _embeddings, _vocab = pickle.load(fin)
    embedding_size = _embeddings.shape[1]
    
    embeddings = init_embeddings(vocab, embedding_size)
    for word, id in vocab.items():
        if word in _vocab:
            embeddings[id] = _embeddings[_vocab[word]]
    
    return embeddings.astype(np.float32)


def init_embeddings(vocab, embedding_size): 
    """
    Initialize word embeddings randomly (normal distribution)
    :param vocab: the vocabulary (word->id mapping)
    :param embedding_size: word embedding size
    """
    rng = np.random.RandomState()
    embeddings = rng.normal(loc = 0.0, scale = 1.0, size=(len(vocab), embedding_size))
    return embeddings.astype(np.float32)


def normalize_embeddings(embeddings):
    """
    Normalize the word embeddings to have norm 1.
    :param embeddings: 2-d numpy array
    :return: normalized embeddings
    """
    # normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1).reshape((-1, 1))
    return embeddings / norms


def prt_log(*args, **keyargs):
    """
    Print on SCREEN and LOG file simultaneously, used just as 'print()'
    :Eg prt_log('train begin ...', file = log_f)
        prt_log('train begin ...')
    """
    print(*args)
    if len(keyargs) > 0:
        print(*args, **keyargs)
    return None


def prt_args(args, log_f):
    """
    Print all used parameters on both SCREEN and LOG file 
    :Param args: all parameters
    :Param log_f: the log life
    """
    argsDict = vars(args)
    argsList = sorted(argsDict.items())
    prt_log("------------- HYPER PARAMETERS -------------", file = log_f)
    for a in argsList:
        prt_log("%s: %s" % (a[0], str(a[1])), file = log_f)
    print("-----------------------------------------", file = log_f)
    return None


def count_parameters():
    """
    Count the number of trainable parameters in the current graph.
    """
    total_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_params = 1
        for dim in shape:
            variable_params *= dim.value
        total_params += variable_params
    return total_params


def get_time_dif(start_time):
    """Calculate the time used"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def open_f(filename, mode='r'):
    """
    mode: 'r' or 'w' for read or write
    """
    return open(filename, mode, encoding='utf-8', errors='ignore')


def build_vocab(data_path, vocab_path, nfreq = 0, lowercase=True):
    """Build vocabulary according the training data
    :param data_path: training data file
    :param vocab_path: save vocabulary file, with format 'word ||| frequence'
    :param nfreq: just keep words occurred more than or equal nfreq times in the training data
    """
    counter = Counter()
    # read train data file
    with open_f(data_path) as f:
        for line in f:
            try:
                if lowercase:
                    line = line.lower()
                words = line.strip().split()
                for word in list(words):
                    counter[word] += 1
            except:
                pass

    count_pairs = [item for item in counter.items() if item[1] >= nfreq]
    count_pairs = sorted(count_pairs, key=lambda k: k[1], reverse=True)
    word_freqs = [' ||| '.join([w, str(f)]) for w, f in count_pairs]
    open_f(vocab_path, mode='w').write('\n'.join(word_freqs) + '\n')
    print('Vocabulary is stored in: {0}'.format(vocab_path))


def load_vocab(vocab_path, cut_off=0, adding=True):
    """Load vocabulary from file
    :param vocab_path: path to text file with vocabulary
    :param cut_off: discard word occurred < cut_off 
    :param adding: whether to add '<<UNK>>' and '<<PAD>>'
    """
    vocab = {}
    idx = 0
    if adding:
        vocab[PADDING] = 0
        vocab[UNKNOWN] = 1
        idx = 2
    with open(vocab_path, encoding='utf-8') as fp:
        for ln in fp:
            items = ln.split('|||')
            if len(items) != 2:
                print('Wrong format: ', ln)
                continue
            word, freq = ln.split('|||')
            word, freq = word.strip(), int(freq.strip())
            if freq >= cut_off:
                vocab[word] = idx
                idx += 1
    return vocab


def process_file(data_fn, word_id, cat_id=CATEGORIE_ID, max_len1=50, max_len2=50, lowercase=True):
    """Prepare data for model:
    convert words into indexes according the vocabulary,
    padding sentences into fix length, and create the corresponding mask arrays
    loading the classification labels of data into a 1-D array
    """
    sen1_id, sen2_id, label_id = [], [], []
    with open_f(data_fn) as f:
        for line in f:
            try:
                label, sen1, sen2 = [x.strip() for x in line.strip().split('|||')]
                if lowercase:
                    sen1 = sen1.lower()
                    sen2 = sen2.lower()
                sen1 = [x.strip() for x in sen1.split()]
                sen2 = [x.strip() for x in sen2.split()]
                if label in cat_id:
                    sen1_id.append([word_id[x] if x in word_id else word_id[UNKNOWN] for x in sen1])
                    sen2_id.append([word_id[x] if x in word_id else word_id[UNKNOWN] for x in sen2])
                    label_id.append(cat_id[label])
            except:
                ValueError('Value error!')
                
    # padding sequence to a fix length
    sen1_pad = kr.preprocessing.sequence.pad_sequences(sen1_id, max_len1, padding='post')
    sen2_pad = kr.preprocessing.sequence.pad_sequences(sen2_id, max_len2, padding='post')
    # mask
    sen1_mask = (sen1_pad > 0).astype(np.int32)
    sen2_mask = (sen2_pad > 0).astype(np.int32)
    # label
    y_pad = np.asarray(label_id, np.int32)
    return sen1_pad, sen1_mask, sen2_pad, sen2_mask, y_pad


def batch_iter(s1, s1_mask, s2, s2_mask, y_pad, batch_size=64, shuffle=True):
    """Generate a batch of data"""
    data_len = len(s1)
    num_batch = int((data_len - 1) / batch_size) + 1
    
    if shuffle:
        indices = np.random.permutation(np.arange(data_len))
        s1 = s1[indices]
        s1_mask = s1_mask[indices]
        s2 = s2[indices]
        s2_mask = s2_mask[indices]
        y_pad = y_pad[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield (s1[start_id:end_id], s1_mask[start_id:end_id],
               s2[start_id:end_id], s2_mask[start_id:end_id],
               y_pad[start_id:end_id])
        

def convert_data(json_path, txt_path):
    """
    Convert NLI data from 'JSON' format to the following 'TXT' format:
    gold_label ||| sentence1 ||| sentence2
    :param json_path:
    :param txt_path:
    """
    import json
    dout = open(txt_path, 'w')
    with open(json_path) as fn:
        i = 0
        entailment, neutral, contradiction, _label = 0, 0, 0, 0
        for line in fn:
            text = json.loads(line.strip())
            if text['gold_label'] == 'entailment':
                entailment += 1
            elif text['gold_label'] == 'neutral':
                neutral += 1
            elif text['gold_label'] == 'contradiction':
                contradiction += 1
            elif text['gold_label'] == '-':
                _label += 1
                
            print(' ||| '.join([text['gold_label'], text['sentence1'], text['sentence2']]), 
                  file = dout)
            
            i += 1
            if i % 10000 == 0:
                print(i)
                
    msg = 'entailment: {0}, neutral: {1}, contradiction: {2}, _label: {3}'
    print(msg.format(entailment, neutral, contradiction, _label))
    print('Finished. From {0} to {1}'.format(json_path, txt_path))
    dout.close()


def convert_embeddings(embedding_path, nwords, size):
    """
    Convert embeddings from TXT (one word embedding per line) 
            to a easy-to-use format in Python, 
            which consists of a 2-d numpy arrays for embeddings and a dictionary for vocabulary,
            and is store in file 'embedding_path + .we'
    :param embedding_path
    :param nwords, number of words
    :param size, embedding size
    """
    vocab = {}
    wid = 0
    wrong = 0
    embeddings = np.zeros((nwords, size), dtype=np.float32)
    with open(embedding_path, 'r', encoding = 'utf-8', errors = 'ignore') as fin:
        for ln in fin:
            items = ln.strip().split()
            if len(items) != size + 1:
                wrong += 1
                print(ln)
                continue
            
            if items[0] in vocab:
                wrong += 1
                print(ln)
                continue
                
            vocab[items[0]] = wid
            embeddings[wid] = [float(it) for it in items[1:]]
            wid += 1
            
    # dump
    dump_path = embedding_path.rsplit('.', 1)[0] + '.we'
    embeddings = embeddings[0:wid,]
    with open(dump_path, 'wb') as fout:
        pickle.dump([embeddings, vocab], fout)
       
    print(len(vocab), embeddings.shape, 'wrong words: ', wrong, 'total words: ', nwords)
    print("Save in: ", dump_path)


# for data preparation
if __name__ == "__main__":
    # convert embeddings
    # convert_embeddings('/home/wucx/wworkspace/res/embeddings/glove.840B.300d.txt', 2196017, 300)
    
    # obtain data needed
    #convert_data('cdata/snli_raw/snli_1.0_train.jsonl', 'cdata/snli/train.txt')
    #convert_data('cdata/snli_raw/snli_1.0_dev.jsonl', 'cdata/snli/dev.txt')
    #convert_data('cdata/snli_raw/snli_1.0_test.jsonl', 'cdata/snli/test.txt')
    
    # build vocabulary
    # build_vocab('cdata/snli/train.tok.forvocab', 'cdata/snli/vocab.txt', nfreq = 0, lowercase=True)
    pass

