import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from keras.regularizers import l2
from tensorflow.python.keras.layers import Dense
from utils import random_uniform_init
from clr import cyclic_learning_rate


def inter_sage(adj, ner_inputs, ner_dim, target_inputs, hidden_dim):
    """
    Aggregate information from neighbor nodes
    :param adj: Adjacency matrix
    :param ner_inputs: disease or drug embedding
    :param ner_dim: ner_inputs dimension
    :param target_inputs: target embedding
    :param hidden_dim: output dimension
    :return:
    """
    edge_index = tf.where(tf.not_equal(adj, 0))
    num_nodes = tf.shape(adj)[0]

    h_i = tf.gather(ner_inputs, edge_index[:, 1])
    mean_feature = tf.math.unsorted_segment_mean(h_i, edge_index[:, 0], num_segments=num_nodes)


    weight = tf.get_variable('inter_sage_weight', shape=[ner_dim, hidden_dim],
                             initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
    bias = tf.get_variable('inter_sage_bias', shape=[hidden_dim],
                           initializer=tf.truncated_normal_initializer(mean=0, stddev=1))

    e_r = tf.nn.xw_plus_b(mean_feature, weight, bias)

    # add target_inputs
    weight_target = tf.get_variable('inter_sage_weight_target', shape=[ner_dim, hidden_dim],
                                    initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
    e_r_target = tf.matmul(target_inputs, weight_target)
    e_r_final = tf.add(e_r, e_r_target)

    return e_r_final



def intra_sage(adj, ner_inputs, feature_dim, hidden_dim):
    """
    Aggregate information from neighbor nodes
    :param adj: Adjacency matrix
    :param ner_inputs: ner embedding
    :param hidden_dim: output dimension
    :return:
    """
    edge_index = tf.where(tf.not_equal(adj, 0))
    num_nodes = tf.shape(adj)[0]

    h_i = tf.gather(ner_inputs, edge_index[:, 1])
    mean_feature = tf.math.unsorted_segment_mean(h_i, edge_index[:, 0], num_segments=num_nodes)

    weight = tf.get_variable('intra_sage_weight', shape=[feature_dim, hidden_dim],
                             initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))
    bias = tf.get_variable('intra_sage_bias', shape=[hidden_dim],
                           initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))

    embedding = tf.nn.xw_plus_b(mean_feature, weight, bias)

    return embedding

