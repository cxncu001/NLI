'''
Created on Apr 2, 2018

@author: wucx
'''
import tensorflow as tf

class DcomposableNLIModel(object):
    '''
    The identical model with that in the paper 
    'A decomposable attention model for nature language inference'.
    '''
    def __init__(self, n_classes, vocab_size, embedding_size, mlen1, mlen2, vocab,
                 attend_layer_sizes, compare_layer_sizes, aggregate_layer_sizes, proj_emb_size, 
                 optimizer_algorithm='adagrad', train_em=True, proj_emb=False):
        """
        """
        # hyper-parameters
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.attend_layer_sizes = attend_layer_sizes
        self.compare_layer_sizes = compare_layer_sizes
        self.aggregate_layer_sizes = aggregate_layer_sizes
        self.proj_emb_size = proj_emb_size
        self.optimizer_algorithm = optimizer_algorithm
        self.train_em = train_em
        self.proj_emb = proj_emb
        self.mlen1 = mlen1
        self.mlen2 = mlen2
        
        # just needed when loading model for test
        _, _, _ = tf.constant(mlen1, name='mlen1'), tf.constant(mlen2, name='mlen2'), \
                  tf.constant(vocab, name='vocab')
        
        # sentence 1 and its mask
        self.s1 = tf.placeholder(tf.int32, (None, self.mlen1), name = 'sen1')
        self.s1_m = tf.placeholder(tf.float32, (None, self.mlen1), name = 'sen1_mask')
        # sentence 2 and its mask
        self.s2 = tf.placeholder(tf.int32, (None, self.mlen2), name = 'sen2')
        self.s2_m = tf.placeholder(tf.float32, (None, self.mlen2), name = 'sen2_mask')
        # labels
        self.y = tf.placeholder(tf.int32, (None), name = 'n_classes')
        
        # learning rate, l2 loss coefficient, dropout keeping, and clip value
        self.lr = tf.placeholder(tf.float32, (), name = 'learning_rate')
        self.l2 = tf.placeholder(tf.float32, (), name = 'l2_contant')
        self.dropout_keep = tf.placeholder(tf.float32, (), name = 'dropout_keep')
        self.clip_value = tf.placeholder(tf.float32, (), name = 'clip_value')
        
        # initialize the embedding from a placeholder
        self.embeddings_ph = tf.placeholder(tf.float32, 
                                            (self.vocab_size, self.embedding_size) )
        self.embeddings = tf.Variable(self.embeddings_ph, trainable=self.train_em,
                                      validate_shape=True, name = 'embeddings')
        # building global graph
        self.build_graph()
        
    
    def build_graph(self):
        
        def _project_embeddings(emb, num_unit, reuse_weights=False):
            """
            Project word embeddings into another dimensionality
            :param emb: embedded sentences, with shape (batch, mlen, embedding_size)
            :param num_unit: the size of output
            :param reuse_weights: reuse weights in internal layers or not
            :return: projected embeddings with shape (batch, mlen, num_unit)
            """
            with tf.variable_scope('proj_emb', reuse=reuse_weights) as self.proj_scope:
                initializer = tf.random_normal_initializer(0.0, 0.1)
                projected = tf.layers.dense(emb, num_unit, kernel_initializer=initializer)
            return projected
            

        def _apply_feedforward(inputs, scope, num_units,
                               reuse_weights=False, 
                               initializer=None):
            """
            Apply two feed forward layers with num_units on the inputs.
            :param inputs: tensor in shape (batch, mlen, num_unit_input)
            :param reuse_weights: reuse the weights inside the same variable scope
            :param initializer: by default a normal distribution
            :param num_units: list of length 2, containing the number of units in each layer
            :return: a tensor with shape (batch, mlen, num_units[-1])
            """
            scope = scope or 'feedforward'
            with tf.variable_scope(scope, reuse=reuse_weights):
                if initializer is None:
                    initializer = tf.random_normal_initializer(0.0, 0.1)
                    
                with tf.variable_scope('layer1'):
                    inputs = tf.nn.dropout(inputs, self.dropout_keep)
                    relus1 = tf.layers.dense(inputs, num_units[0], tf.nn.relu, kernel_initializer=initializer)
                with tf.variable_scope('layer2'):
                    inputs = tf.nn.dropout(relus1, self.dropout_keep)
                    relus2 = tf.layers.dense(inputs, num_units[1], tf.nn.relu, kernel_initializer=initializer)
            return relus2
        
        
        def _attend(sent1, sent2):
            """
            Compute inter-sentence attention, the step 1 (attend) in the paper
            :param sent1: tensor in shape (batch, mlen1, num_unit_input),
            :param sent2: tensor in shape (batch, mlen2, num_unit_input),
            :return: a tuple of 3-d tensors, alpha and beta.
                     alpha with shape (batch, mlen1, num_units[-1]),
                     beta with shape (batch, mlen2, num_units[-1]),
             """
            with tf.variable_scope('attend_scope') as self.attend_scope:
                num_units = self.attend_layer_sizes
                
                # repr1 has shape (batch, mlen1, num_units[-1])
                # repr2 has shape (batch, mlen2, num_units[-1]), shared parameters
                repr1 = _apply_feedforward(sent1, self.attend_scope, num_units)
                repr2 = _apply_feedforward(sent2, self.attend_scope, num_units, reuse_weights=True)
                # cross mask
                m1_m2 = tf.multiply(tf.expand_dims(self.s1_m, 2), tf.expand_dims(self.s2_m, 1))
                
                # compute the unnormalized attention for all word pairs
                # raw_attentions has shape (batch, mlen1, mlen2)
                repr2 = tf.transpose(repr2, [0, 2, 1])
                raw_atten = tf.matmul(repr1, repr2)
                raw_atten = tf.multiply(raw_atten, m1_m2)
                # weighted attention, 
                # using Softmax at two directions axis=-1 and axis=-2, for alpha and beta respectively
                atten1 = tf.exp(raw_atten - tf.reduce_max(raw_atten, axis=2, keep_dims=True))
                atten2 = tf.exp(raw_atten - tf.reduce_max(raw_atten, axis=1, keep_dims=True))
                # mask
                atten1 = tf.multiply(atten1, tf.expand_dims(self.s2_m, 1))
                atten2 = tf.multiply(atten2, tf.expand_dims(self.s1_m, 2))
                # get softmax value
                atten1 = tf.divide(atten1, tf.reduce_sum(atten1, axis=2, keep_dims=True))
                atten2 = tf.divide(atten2, tf.reduce_sum(atten2, axis=1, keep_dims=True))
                # mask
                atten1 = tf.multiply(atten1, m1_m2)
                atten2 = tf.multiply(atten2, m1_m2)
                
                # here (alpha, beta) = (beta, alpha) in the paper
                # represents the soft alignment in the other sentence
                alpha = tf.matmul(atten1, sent2, name='alpha')
                beta = tf.matmul(tf.transpose(atten2,[0,2,1]), sent1, name='beta')
                
            return alpha, beta
        
        
        def _compare(sen, soft_align, reuse_weights=False):
            '''
            Apply a feed forward network to compare one sentence to its 
            soft alignment with the other.
            :param sentence: embedded and projected sentence,
                   shape (batch, mlen, embedding_size)
            :param soft_alignment: tensor with shape (batch, mlen, num_unit_input)
            :param reuse_weights: whether to reuse weights in the internal layers
            :return: a tensor (batch, mlen, num_units[-1])
            '''
            with tf.variable_scope('compare_score', reuse=reuse_weights) as self.comapre_score:
                inputs = [sen, soft_align]
                inputs = tf.concat(inputs, axis=2)
                # two-layer NN
                num_units = self.compare_layer_sizes
                output = _apply_feedforward(inputs, self.comapre_score, 
                                            num_units, reuse_weights)
            
            return output
        
        
        def _aggregate(v1, v2):
            """
            Aggregate the representations induced from both sentences and their
            representations
            Note that: No masks are used.
            :param v1: tensor with shape (batch, mlen1, num_unit_input1)
            :param v2: tensor with shape (batch, mlen2, num_unit_input2)
            :return: logits over classes, shape (batch, n_classes)
            """
            # calculate sum
            v1_sum = tf.reduce_sum(v1, 1)
            v2_sum = tf.reduce_sum(v2, 1)
            
            inputs = tf.concat(axis=1, values=[v1_sum, v2_sum])
            with tf.variable_scope('aggregate_scope') as self.aggregate_scope:
                num_units = self.aggregate_layer_sizes
                logits = _apply_feedforward(inputs, self.aggregate_scope, num_units)
                # the last layer
                logits = tf.layers.dense(logits, self.n_classes, name='last_layer')
                
            return logits
        
        
        def _create_training_op(optimizer_algorithm):
            """
            Create the operation used for training
            """
            with tf.name_scope('training'):
                if optimizer_algorithm == 'adagrad':
                    optimizer = tf.train.AdagradOptimizer(self.lr)
                elif optimizer_algorithm == 'adam':
                    optimizer = tf.train.AdamOptimizer(self.lr)
                elif optimizer_algorithm == 'adadelta':
                    optimizer = tf.train.AdadeltaOptimizer(self.lr)
                else:
                    ValueError('Unkown optimizer: {0}'.format(optimizer_algorithm))
            
            # clip gradients
            gradients, v = zip(*optimizer.compute_gradients(self.loss))
            
            if self.clip_value is not None:
                gradients, _ = tf.clip_by_global_norm(gradients, self.clip_value)
            train_op = optimizer.apply_gradients(zip(gradients, v))
            return train_op
        
        # build graph
        with tf.device('/cpu:0'):
            # emb1 has shape (batch, mlen1, embedding_size)
            # emb2 has shape (batch, mlen2, embedding_size)
            emb1 = tf.nn.embedding_lookup(self.embeddings, self.s1)
            emb2 = tf.nn.embedding_lookup(self.embeddings, self.s2)
        
        # the architecture has 3 main steps: soft align, compare and aggregate
        with tf.name_scope('align_compare_aggregate'):
            if self.proj_emb:
                repr1 = _project_embeddings(emb1, self.proj_emb_size)
                repr2 = _project_embeddings(emb2, self.proj_emb_size, reuse_weights=True)
            else:
                repr1, repr2 = emb1, emb2
            # soft align, Eq.2 in the paper
            alpha, beta = _attend(repr1, repr2)
            # compare, Eq.3 in the paper
            repr1 = tf.multiply(repr1, tf.expand_dims(self.s1_m, -1))
            repr2 = tf.multiply(repr2, tf.expand_dims(self.s2_m, -1))
            v1, v2 = _compare(repr1, alpha), _compare(repr2, beta, reuse_weights=True)
            # aggregate, Eq.4 and Eq.5 in the paper (difference)
            self.logits = _aggregate(v1, v2)
            
        # for training
        with tf.name_scope('optimize'):
            # for classification loss
            cross_entropy = \
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
            labeled_loss = tf.reduce_mean(cross_entropy)
            # for L2 loss
            weights = [v for v in tf.trainable_variables() if 'kernel' in v.name]
            # the same as tf.reduce_sum([tf.nn.l2_loss(weight) for weight in weights]) 
            l2_partial_sum = sum([tf.nn.l2_loss(weight) for weight in weights])
            l2_loss = tf.multiply(self.l2, l2_partial_sum)
            # total loss = classification loss + L2 loss
            self.loss = tf.add(labeled_loss, l2_loss)
            self.train_op = _create_training_op(self.optimizer_algorithm)
            
        with tf.name_scope('predict'):
            # predict y
            self.y_pred = tf.cast(tf.argmax(tf.nn.softmax(self.logits), axis = 1), tf.int32, name='y')
            # accuracy
            correct_pred = tf.equal(self.y, self.y_pred)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        ## end build_graph
            