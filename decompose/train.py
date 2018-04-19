'''
Created on Apr 2, 2018

@author: wucx
'''

import os
import sys
import time
import argparse
import tensorflow as tf
import nliutils
from datetime import datetime
from decompose.model import DcomposableNLIModel
from nliutils import prt_log, prt_args

"""
Script to train a decompose NLI model
"""

def feed_data(s1_batch, s1_batch_mask, s2_batch, s2_batch_mask, y_batch, 
              learning_rate, dropout_keep, l2, clip_value):
    feed_dict = {
        model.s1: s1_batch,
        model.s1_m: s1_batch_mask,
        model.s2: s2_batch,
        model.s2_m: s2_batch_mask,
        model.y: y_batch,
        model.lr: learning_rate,
        model.dropout_keep: dropout_keep,
        model.l2: l2,
        model.clip_value: clip_value
    }
    return feed_dict

def evaluate(sess, s1, s1_mask, s2, s2_mask, y):
    """Evaluate the performance on a data set
    return loss and accuracy
    """
    data_len = len(s1)
    batch_eval = nliutils.batch_iter(s1, s1_mask, s2, s2_mask, y, args.batch_size, shuffle=False)
    total_loss = 0.0
    total_acc = 0.0
    for batch in batch_eval:
        #s1_batch, s1_batch_mask, s2_batch, s2_batch_mask, y_batch = batch
        s1_batch = batch[0]
        batch_len = len(s1_batch)
        feed_dict = feed_data(*batch, args.lr, 1.0, args.l2, args.clip_value)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len
    return total_loss / data_len, total_acc / data_len

def train():
    # data preparation
    prt_log("Loading training and validation data...", file=log)
    start_time = time.time()
    s1_train, s1_train_mask, s2_train, s2_train_mask, y_train = \
           nliutils.process_file(args.train, vocab, max_len1=args.mlen1, max_len2=args.mlen2, lowercase=True)
    s1_val, s1_val_mask, s2_val, s2_val_mask, y_val = \
           nliutils.process_file(args.validation, vocab, max_len1=args.mlen1, max_len2=args.mlen2, lowercase=True)
    data_len = len(s1_train)
    time_dif = nliutils.get_time_dif(start_time)
    prt_log("Time usage:", time_dif, file=log)
    
    # for tensorBoard
    prt_log("Configuring TensorBoard and Saver...", file=log)
    if not os.path.exists(args.tfboard):
        os.makedirs(args.tfboard)
    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(args.tfboard)
    # for model saving
    saver = tf.train.Saver()
    save_dir, _ = os.path.split(args.save)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # create session, initialization
    sess = tf.Session()
    sess.run(tf.global_variables_initializer(), {model.embeddings_ph: embeddings})
    writer.add_graph(sess.graph)
    # count parameters in model
    total_params = nliutils.count_parameters()
    prt_log('Total parameters: {0}'.format(total_params), file=log)
    
    # training start
    prt_log('Training and evaluating...', file=log)
    start_time = time.time()
    total_batch = 0     # total batches
    best_acc_val = 0.0  # the best accuracy on validation set
    last_improved = 0   # the last batch with improved accuracy
    flag = False        # stop training
    for epoch in range(args.num_epochs):
        prt_log('Epoch:', epoch + 1, file=log)
        batch_train = nliutils.batch_iter(s1_train, s1_train_mask, 
                                        s2_train, s2_train_mask, 
                                        y_train, 
                                        args.batch_size,
                                        shuffle=True)
        total_loss, total_acc = 0.0, 0.0
        for batch in batch_train:
            batch_len = len(batch[0])
            #s1_batch, s1_batch_mask, s2_batch, s2_batch_mask, y_batch = batch
            feed_dict = feed_data(*batch, args.lr, args.dropout_keep, args.l2, args.clip_value)
            # optimize, obtain the loss and accuracy on the current training batch
            _, batch_loss, batch_acc = sess.run([model.train_op, model.loss, model.acc], feed_dict=feed_dict) 
            total_loss += batch_loss * batch_len
            total_acc += batch_acc * batch_len
            
            if total_batch % args.save_per_batch == 0:
                # write tensorboard scalar
                s = sess.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)
            
            if total_batch % args.report_per_batch == 0:
                feed_dict[model.dropout_keep] = 1.0
                # the loss and accuracy on the whole validation data
                loss_val, acc_val = evaluate(sess, s1_val, s1_val_mask, s2_val, s2_val_mask, y_val)
                if acc_val > best_acc_val:
                    # save the best results and model
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=sess, save_path=args.save)
                    improved_str = '*'
                else:
                    improved_str = ''
                # report training information
                time_dif = nliutils.get_time_dif(start_time)    
                msg = 'Epoch: {0:>2}, Batch: {1:>7}, Train Batch Loss: {2:>6.2}, Train Batch Acc: {3:>7.2%},' \
                          + ' Val Loss: {4:>6.2}, Val Acc: {5:>7.2%}, Time: {6} {7}'
                prt_log(msg.format(epoch + 1, total_batch, batch_loss, batch_acc, loss_val, 
                                   acc_val, time_dif, improved_str), file=log)
                
            total_batch += 1
            if total_batch - last_improved > args.require_improvement:
                # stop training when no performance improvement for a long time
                prt_log("No optimization for a long time, auto-stopping...", file=log)
                flag = True
                break
        if flag:
            break
        # report train loss and accuracy on the train data set at the Epoch
        time_dif = nliutils.get_time_dif(start_time) 
        total_loss, total_acc = total_loss / data_len, total_acc / data_len
        msg = '*** Epoch: {0:>2}, Train Loss: {1:>6.2}, Train Acc: {2:7.2%}, Time: {3}'    
        prt_log(msg.format(epoch + 1, total_loss, total_acc, time_dif), file=log)
    
    sess.close()
    # END train() #

if __name__ == '__main__':
    # Part 1, hyper-parameters preparation
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-ep', dest='num_epochs', default=200, type=int, help='Number of epochs')
    parser.add_argument('-bs', dest='batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('-dr', dest='dropout_keep', default=0.8, type=float, help='Dropout keep probability')
    parser.add_argument('-cl', dest='clip_value', default=5.0, type=float, help='Norm to clip training gradients')
    parser.add_argument('-lr', dest='lr', default=0.05, type=float, help='Learning rate') 
    parser.add_argument('-l2', dest='l2', default=0.001, type=float, help='L2 normalization constant')
    parser.add_argument('-m1', dest='mlen1', default=100, type=int, help='Max length of sentence 1')
    parser.add_argument('-m2', dest='mlen2', default=100, type=int, help='Max length of sentence 2')
    parser.add_argument('-op', dest='optimizer_algorithm', default='adagrad', 
                        choices=['adagrad', 'adadelta', 'adam'], help='Optimizer algorithm')
    # embddings
    parser.add_argument('--vocab', dest='vocab', default='../cdata/snli/vocab.txt', 
                        help='Vocabulary file')
    parser.add_argument('--cut_off', dest='cut_off', default=0, type = int,  
                        help='Cut off freq(word)<N in vocabulary')
    parser.add_argument('--embedding_size', dest='embedding_size', default=50, type=int,
                        help='Word embedding size')
    parser.add_argument('--embeddings', dest='embeddings', 
                        help='Pre-trained word embeddings file')
    parser.add_argument('--normalize', dest='normalize', default=1, type=int, 
                        help='normalize word embeddings')
    parser.add_argument('--train_em', dest='train_em', default=0, type=int, 
                        help='fine-tuning word embeddings')
    # layer sizes
    parser.add_argument('--attend_sizes', dest = 'attend_layer_sizes', nargs = 2, 
                        type = int, default=[200, 200], help = 'attend_layer_sizes')
    parser.add_argument('--compare_sizes', dest = 'compare_layer_sizes', nargs = 2, 
                        type = int, default=[200, 200], help = 'compare_layer_sizes')
    parser.add_argument('--aggregate_sizes', dest = 'aggregate_layer_sizes', nargs = 2, 
                        type = int, default=[200, 200], help = 'aggregate_layer_sizes')
    # projecting word embeddings or not
    parser.add_argument('--proj_emb', dest = 'proj_emb', type = int, default=1, 
                        help = 'project word embeddings or not')
    parser.add_argument('--proj_size', dest = 'proj_emb_size', type = int, default=200, 
                        help = 'project word embeddings size')
    # train, validation and test data
    parser.add_argument('--train', dest='train', default='../cdata/snli/train.tok', 
                        help='Training corpus')
    parser.add_argument('--validation', dest='validation', default='../cdata/snli/dev.tok', 
                        help='Validation corpus')
    # save and report settings
    parser.add_argument('--save', dest='save', default='data/model/bestval',
                        help='Directory to save the model files')
    parser.add_argument('--report', dest='report_per_batch', default=500, type=int, 
                        help='Number of batches between performance reports')
    parser.add_argument('--save_per_batch', dest='save_per_batch', default=500, type=int, 
                        help='Number of batches between saving to tensorboard scalar')
    parser.add_argument('--require_improvement', dest='require_improvement', default=100000, type=int, 
                        help='Max number of batches between two improvements on validation set,' + \
                         'otherwise STOP training')
    parser.add_argument('--tfboard', dest='tfboard', default='data/tensorboard',
                        help='Directory to save the TensorBoard files')
    args = parser.parse_args()
    # load vocabulary and embeddings
    vocab = nliutils.load_vocab(args.vocab, cut_off=args.cut_off, adding=True)
    args.vocab_size = len(vocab)
    if args.embeddings:
        # load pre-trained word ebmeddings
        embeddings = nliutils.load_embeddings(args.embeddings, vocab)
        args.embedding_size = embeddings.shape[1]
    else:
        # initialize word embeddings randomly
        embeddings = nliutils.init_embeddings(vocab, args.embedding_size)
    if args.normalize:
        embeddings = nliutils.normalize_embeddings(embeddings)
    # other parameters
    # the number of classes
    args.n_classes = 3
    args.train_em = args.train_em != 0                     
    args.proj_emb = args.proj_emb != 0
    # report CMD line, hyper-parameters
    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    args.log = "data/log/log.{0}".format(dt)
    log = open(args.log, 'w')
    prt_log('CMD: python3 {0}'.format(' '.join(sys.argv)), file=log)
    prt_log('Training with following options:', file=log)
    prt_args(args, log)
    # Part 2, model training
    model = DcomposableNLIModel(args.n_classes, args.vocab_size, args.embedding_size, args.mlen1, args.mlen2, 
                     args.vocab, args.attend_layer_sizes, args.compare_layer_sizes, args.aggregate_layer_sizes, 
                     args.proj_emb_size, args.optimizer_algorithm, train_em=args.train_em, proj_emb=args.proj_emb)
    train()
    log.close()
    