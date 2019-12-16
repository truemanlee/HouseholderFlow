import numpy as np
import tensorflow as tf 

# HouseHolder tranformation
# z' = (I - 2vv^T/||vt||^2) z
# z' = H z
class HouseHolderFlow():
    def __init__(self, num_flow):
        self.num_flow = num_flow

    def hf_transform(self, z, mu, layer_index):
        with tf.variable_scope('hf_transform_{}'.format(layer_index)):
            # z:[B, D] -> [B, D]
            dims = z.shape().as_list()[-1]
            I = tf.eye(dims) # [D, D]
            v = tf.get_variable('v_{}'.format(i), [dims], tf.float32)
            v_norm = tf.reduce_sum(v ** 2) # may need clip to prevent the overflow
            v = tf.expand_dims(v, axis=-1) # [D, 1]
            v = tf.matmul(v, v, transpose_b=True) # [D, D]
            H = I - 2 * v / v_norm
            z_t = tf.transpose(tf.matmul(H, z, transpose_b=True), [1, 0])
            mu_t = tf.transpose(tf.matmul(H, mu, transpose_b=True), [1, 0])
            return z_t, mu_t 

    def NN_encoder(self, inputs):
        # mu and sigma encoder, it's just a template and one could design it as him/her/it likes.
        mean = inputs
        stddev = tf.nn.softplus(inputs)
        return mean, stddev # stddev may need clip to prevent the overflow

    def hf_flow(self, inputs):
        output = dict()
        mu, sigma = self.NN_encoder(inputs) # [B, D], [B, D]
        z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
        output['mu'] = mu
        output['sigma'] = sigma
        output['z'] = z
        for i in range(self.num_flow):
            z, mu_t = self.hf_transform(z, i)
        output['z_t'] = z
        output['mu_t'] = mu_t
        return output

    def NN_decoder(self, inputs):
        # decoder, it's just a template and one could design it as him/her/it likes.
        # the inputs should be z_t, if one needs other conditions, one could add it as him/her/it likes
        outputs = inputs
        return outputs

    def add_loss(self, hf_output, targets):
        # reconstruction error & KL loss
        rec = self.NN_decoder(hf_output['zt'])
        loss1 = tf.reduce_mean((ref - targets) ** 2)
        # KL divergence
        # householder flow does not change the sigma but changes the mu, and mu = H_t H_{t-1} ... H1
        loss2 = 0.5 * tf.reduce_mean(-tf.log(hf_output['sigma']) + hf_output['sigma'] + (hf_output['mu'] - hf_output['zmu']))
        loss2 = -loss2
        return loss1 + loss2
