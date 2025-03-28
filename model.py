# Graph Neural Networks for Drug Repositioning
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from keras.regularizers import l2
from tensorflow.python.keras.layers import Dense
from utils import random_uniform_init
from clr import cyclic_learning_rate
from model_utils import inter_sage, intra_sage



class Model(object):

    def __init__(self, config, dr_feat, di_feat):
        """
        :param config:
        """
        self.config = config
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.disease_dim = config['disease_dim']
        self.drug_dim = config['drug_dim']
        self.disease_size = config['disease_size']
        self.drug_size = config['drug_size']
        self.latent_dim = config['latent_dim']
        self.l2 = config['l2']  # init=0

        self.global_step = tf.Variable(0, trainable=False)
        # self.best_dev_auroc = tf.Variable(0.0, trainable=False)
        # self.best_test_auroc = tf.Variable(0.0, trainable=False)
        # self.best_dev_aupr = tf.Variable(0.0, trainable=False)
        # self.best_test_aupr = tf.Variable(0.0, trainable=False)

        # input
        self.disease_drug_Adj = tf.placeholder(dtype=tf.float32,
                                               shape=[self.disease_size, self.drug_size])
        self.disease_disease_Adj = tf.placeholder(dtype=tf.float32,
                                                  shape=[self.disease_size, self.disease_size])
        self.drug_drug_Adj = tf.placeholder(dtype=tf.float32,
                                            shape=[self.drug_size, self.drug_size])

        self.input_disease = tf.placeholder(dtype=tf.int32, shape=[None])
        self.input_drug = tf.placeholder(dtype=tf.int32, shape=[None])
        self.label = tf.placeholder(dtype=tf.float32, shape=[None])

        if config['use_LLM']:
            self.disease_embedding_LLM = tf.convert_to_tensor(di_feat.numpy(), dtype=tf.float32)
            self.drug_embedding_LLM = tf.convert_to_tensor(dr_feat.numpy(), dtype=tf.float32)
            self.disease_embedding = tf.nn.l2_normalize(self.disease_embedding_LLM, axis=1, name="l2_normalize_1")
            self.drug_embedding = tf.nn.l2_normalize(self.drug_embedding_LLM, axis=1, name="l2_normalize_2")
            self.disease_dim = self.disease_embedding.shape[1]
            self.drug_dim = self.drug_embedding.shape[1]
        else:
            self.disease_embedding = random_uniform_init(name="disease_embedding_matrix",
                                                    shape=[self.disease_size, self.disease_dim])
            self.drug_embedding = random_uniform_init(name="drug_embedding_matrix",
                                                      shape=[self.drug_size, self.drug_dim])


        with tf.variable_scope("model_disease", reuse=tf.AUTO_REUSE):

            disease_inter_output = inter_sage(self.disease_drug_Adj, self.drug_embedding, self.drug_dim,
                                              self.disease_embedding, self.latent_dim)
            disease_intra_output = intra_sage(self.disease_disease_Adj, self.disease_embedding, self.disease_dim,
                                              self.latent_dim)

            self.disease_aggregation = tf.nn.selu(tf.add(disease_inter_output, disease_intra_output))

        with tf.variable_scope("model_drug", reuse=tf.AUTO_REUSE):
            drug_inter_ouput = inter_sage(tf.transpose(self.disease_drug_Adj), self.disease_embedding, self.disease_dim,
                                          self.drug_embedding, self.latent_dim)
            drug_intra_output = intra_sage(self.drug_drug_Adj, self.drug_embedding, self.drug_dim, self.latent_dim)

            self.drug_aggregation = tf.nn.selu(tf.add(drug_inter_ouput, drug_intra_output))

        with tf.variable_scope("drug_rec", reuse=tf.AUTO_REUSE):
            disease_aggregation_batch = tf.nn.embedding_lookup(self.disease_aggregation, self.input_disease)  # batch_size * disease_latent_dim
            drug_aggregation_batch = tf.nn.embedding_lookup(self.drug_aggregation, self.input_drug)  # batch_size * drug_latent_dim
            input_temp = tf.multiply(disease_aggregation_batch, drug_aggregation_batch)
            for l_num in range(config['mlp_layer_num']):
                input_temp = Dense(self.disease_dim, activation='selu', kernel_initializer='lecun_uniform')(input_temp)  # MLP hidden layer
            z = Dense(1, kernel_initializer='lecun_uniform', name='prediction')(input_temp)
            z = tf.squeeze(z)

        self.label = tf.squeeze(self.label)
        self.loss = tf.losses.sigmoid_cross_entropy(self.label, z)
        self.z = tf.sigmoid(z)

        # train
        with tf.variable_scope("optimizer"):
            self.opt = tf.train.AdamOptimizer(learning_rate=cyclic_learning_rate(global_step=self.global_step,
                                                                                 learning_rate=self.lr*0.1,
                                                                                 max_lr=self.lr,
                                                                                 mode='exp_range',
                                                                                 gamma=.999))

            # apply grad clip to avoid gradient explosion
            self.grads_vars = self.opt.compute_gradients(self.loss)
            capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
                                 for g, v in self.grads_vars]

            self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)

        # saver of the model
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)


    def run_step(self, sess, is_train, batch):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        disease_drug_Adj, disease_disease_Adj, drug_drug_Adj, input_disease, input_drug, label = batch
        feed_dict = {
            self.disease_drug_Adj: np.asarray(disease_drug_Adj),
            self.disease_disease_Adj: np.asarray(disease_disease_Adj),
            self.drug_drug_Adj: np.asarray(drug_drug_Adj),
            self.input_disease: np.asarray(input_disease),
            self.input_drug: np.asarray(input_drug),
            self.label: np.asarray(label)
        }

        if is_train:
            global_step, loss, z, grads_vars, _ = sess.run(
                [self.global_step, self.loss, self.z, self.grads_vars, self.train_op], feed_dict)
            return global_step, loss, z, grads_vars
        else:
            z, labels = sess.run([self.z, self.label], feed_dict)
            return z, labels
