import torch
import math
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from time import time


# LSTM = long short term memory recurrent neural nets
# there are many implementations of LSTMs, why?
# - people want to try different variations, for example multiplicative LSTM, or convolutional LSTM
# - the "default" LSTMs in pytorch or tensorflow is VERY SLOW, and other people can make it run faster
# now everyone uses nn.LSTM because:
# - people are not interested in LSTM anymore
# - the default LSTM has been implemented well (as fast as the previous fast versions)
# - it takes years for the developers to finalize an implementation of LSTM to satisfy the users


def scale_dot_product_attention(q, k, v, scale=1.0, mask=None):
    # the inputs of dot product attention is
    # matrix Q: size [B x Tq x H]: translates to T time steps, each timestep has a batch of B vectors of size H
    # or you can also see it as: B sequences, each sequence has T time steps, each time step is a vector size H
    # matrix K: size [B x Tk x H]
    # matrix V: size [B x Tk x H]

    # matmul Q and K: it means that we compute the "content interaction" between Q (queries) and K (keys)
    # for every batch B, Q is a sequence of T query vectors and K and is sequence of T key vectors
    # the matmul will return a matrix in which: M_ij will compute the relationship between Q_i and K_j

    # Q is transposed from [Tq x B x H] -> [B x Tk x H]
    # K is transposed from [Tq x B x H] -> [B x Tk x H] -> [B x H x Tk]
    matmul_qk = torch.bmm(q.transpose(0, 1), k.transpose(0, 1).transpose(1, 2))
    # the dimension of matmul_qk is [B x T x T] (no more H)
    # this also means that if we have an image (T = W * H) then this matrix will require ( W * H * W * H) storage

    attn_score = matmul_qk * scale

    # mask option: means that sometimes we don't want to look at some positions
    # those positions can be padding or we can dealing with local attention

    if mask is not None:
        attn_score = attn_score.masked_fill_(mask, -999999)

    # [B x Tq x Tk]
    attn_weights = F.softmax(attn_score, dim=2)

    # final matmul: we multiply the attn_weights with the values
    # (for the weighted sum of values)

    # [B x Tq x Tk] x [B x Tk x H] -> [B x Tq x H]
    output = torch.bmm(attn_weights, v.transpose(0, 1))
    # the output dimension suggest that for every query (in Tq), we have a vector sized H, which is the weighted sum
    # of the values -> this is what we need to for attention
    output = output

    return output


# implements MultiHeadAttention
class MultiHeadSelfAttention(nn.Module):

    def __init__(self, model_size, n_heads):
        super(MultiHeadSelfAttention, self).__init__()

        self.model_size = model_size
        self.n_heads = n_heads
        # we divide the model size into different heads
        # 512 / 8 = 64.0  = float
        # 512 // 8 = 64 = integer
        self.head_dim = model_size // n_heads

        self.q_linear = nn.Linear(model_size, model_size)
        self.k_linear = nn.Linear(model_size, model_size)
        self.v_linear = nn.Linear(model_size, model_size)

        self.linear_out = nn.Linear(model_size, model_size)

    # the good thing about images is that all images in the batch have exactly the same size
    # -> we don't need a mask
    # if we batch different sequences and we have to add pads so they have the same size
    # -> we need a mask so that we don't pay attention to the padded positions
    def forward(self, x, impl='fast'):
        # x size: [T x B x model_size]
        # T is the number of pixels (W x H) after the efficient net (for example 16 x 16 = 256)
        # B is the batch size
        # model size is our choice
        t, b = x.size(0), x.size(1)
        scale = 1 / math.sqrt(self.head_dim)
        # this is self attention so x is considered as both queries, keys and values
        q = self.q_linear(x)  # [T x B x H]
        k = self.k_linear(x)
        v = self.v_linear(x)

        q = q.view(t, b, self.n_heads, self.head_dim)
        k = k.view(t, b, self.n_heads, self.head_dim)
        v = v.view(t, b, self.n_heads, self.head_dim)

        # this is the slow way
        if impl == 'slow':
            outputs = list()
            for head in range(self.n_heads):
                q_head = q[:, :, head, :]
                k_head = k[:, :, head, :]
                v_head = v[:, :, head, :]

                output_head = scale_dot_product_attention(q_head, k_head, v_head, scale=scale, mask=None)
                outputs.append(output_head)

            output = torch.cat(outputs, dim=-1).view(b, t, self.model_size)  # [T x B*head x head_dim]
            output = output.transpose(0, 1).contiguous()
            output = self.linear_out(output)

            return output

        elif impl == 'fast':

            q = q.view(t, b * self.n_heads, self.head_dim)
            k = k.view(t, b * self.n_heads, self.head_dim)
            v = v.view(t, b * self.n_heads, self.head_dim)

            output = scale_dot_product_attention(q, k, v, scale=scale, mask=None)
            output = output.transpose(0, 1).contiguous().view(t, b, self.model_size)
            output = self.linear_out(output)

            return output


class PositionalEncoding(nn.Module):
    '''PE(pos,2i) =sin(pos/100002i/dmodel)
       PE(pos,2i+1) =cos(pos/100002i/dmodel)
    '''
    def __init__(self, model_size, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, model_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_size, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, model_size]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# implements Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):

    def __init__(self, model_size=512, n_heads=8, dropout=0.1):
        # have to call super init first to initialize the PyTorch module
        # (it means that PyTorch will understand that this is a network part and register the parmaters)
        super(TransformerEncoderLayer, self).__init__()

        # assign attributes to the class instannce (self)
        # it allows the instance (self) to re-access these numbers later if necessary
        self.model_size = model_size
        self.n_heads = n_heads

        # a transformer layer has 3 main components: multihead-self-attention, 2x layer norm and feed-forward neural net

        # the normalization before or after self-attention

        # layer normalization means that the final dimension of the input (x) will be normalized
        # ( the values are substracted by the mean and then divided by sqrt(var(x))
        # in order to make the layer a bit more robust, the normalized(x) is then multiplied with weights and plus bias
        # (the initialized values of weights is 1 and bias is 0) (it means that the layer norm tries
        # to center the input x around 0 and 1

        # batch norm is very similar but it takes the average over the batch dimension
        # (not the channel dimension as in layer norm)
        self.attn_norm = nn.LayerNorm(model_size, eps=1e-05, elementwise_affine=True)

        # layer normalization before or after the feed forward network
        # by default the layer is created in CPU, and then we will copy it to GPU later
        # (via model = model.cuda() or model = model.device("gpu0"))
        # but pytorch allows us to create a layer directly on GPU if we ever need
        self.ffn_norm = nn.LayerNorm(model_size, eps=1e-05, elementwise_affine=True)

        # each layer has two sets of parameters:
        self.fc1 = nn.Linear(self.model_size, self.model_size * 4)
        # the intermediate layer is larger the "model size" -> in some paper, this layer is called memory
        # so larger memory is better
        self.fc2 = nn.Linear(self.model_size * 4, self.model_size)

        # multihead attention
        self.self_attn = MultiHeadSelfAttention(self.model_size, self.n_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, impl='fast'):
        # x size should be [T x B x H]

        # first block
        residual = x
        x = self.self_attn(x, impl=impl)
        x = self.dropout(x)  # apply dropout
        x = x + residual  # residual connection
        x = self.attn_norm(x)  # layer norm

        # second block
        residual = x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout(x)  # apply dropout
        x = x + residual
        x = self.ffn_norm(x)

        return x

#Z
# implements Transformer Decoder Layer
class TransformerDecoderLayer(nn.Module):
    def __init__(self, model_size=512, n_heads=8, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()

        # assign attributes to the class instance (self)
        # it allows the instance (self) to re-access these numbers later if necessary
        self.model_size = model_size
        self.n_heads = n_heads

        # a transformer decoder layer has 4 main components:
        # Masked multihead-self-attentio, multihead-self-attention, 3x layer norm and feed-forward neural net

        # batch norm is very similar but it takes the average over the batch dimension
        # (not the channel dimension as in layer norm)
        self.batch_norm = nn.LayerNorm(model_size, eps=1e-05, elementwise_affine=True)

        # each layer has two sets of parameters:
        self.fc1 = nn.Linear(self.model_size, self.model_size * 4)
        # the intermediate layer is larger the "model size" -> in some paper, this layer is called memory
        # so larger memory is better
        self.fc2 = nn.Linear(self.model_size * 4, self.model_size)

        # multihead attention
        self.enc_self_attn = MultiHeadSelfAttention(self.model_size, self.n_heads)
        self.dec_self_attn = MultiHeadSelfAttention(self.model_size, self.n_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask, impl='fast'):
        '''
        dec_inputs: [batch_size, tgt_len, model_size]
        enc_outputs: [batch_size, src_len, model_size]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        '''

        residual = dec_inputs
        x = self.dec_self_attn(dec_inputs, impl=impl)
        x = self.get_attn_subsequence_mask(x)
        x = self.droput(x)
        x = x + residual
        x = self.batch_norm(x)

        residual = x
        x = self.dec_self_attn(enc_outputs+dec_inputs)
        x = self.dropout(x)
        x = x + residual
        x = self.batch_norm(x)

        residual = x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout(x)  # apply dropout
        x = x + residual
        x = self.batch_norm(x)

        return x



def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1) # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask # [batch_size, tgt_len, tgt_len]


if __name__ == '__main__':
    # multihead_attn = MultiHeadSelfAttention(512, 64)

    seq_len = 256
    batch_size = 64

    test_input = torch.randn(seq_len, batch_size, 512)

    # multihead_attn = multihead_attn.cuda()
    test_input = test_input.cuda()

    dropout = 0.1
    # with dropout > 0: the outputs of each layer are randomly set to 0
    # and this process is randomly different run by run
    # -> so the same network with dropout will have different  results each run
    transformer_encoder_layer = TransformerEncoderLayer(512, 64, dropout)

    # this command will copy all modules to cuda, including the layer norms (they were initialized in CPU)
    transformer_encoder_layer = transformer_encoder_layer.cuda()

    output = transformer_encoder_layer(test_input, impl='fast')
    output_slow = transformer_encoder_layer(test_input, impl='slow')

    print(output - output_slow)

    # output_fast = multihead_attn(test_input, impl='fast')
    # output_slow = multihead_attn(test_input, impl='slow')
    #
    # print(output_fast - output_slow)
    # (output_fast + output_slow).sum().backward()
    #
    # num_iters = 30
    # torch.cuda.profiler.start()
    # torch.cuda.synchronize()
    # start_time = time()
    # for _ in range(num_iters):
    #     output_fast = multihead_attn(test_input, impl='fast')
    #     output_fast.sum().backward()
    #     multihead_attn.zero_grad()
    #
    # torch.cuda.synchronize()
    # stop_time = time()
    # print(F"\nFast Self-Attn time {(stop_time - start_time) * 1000. / num_iters:.4f} ms")
    #
    # torch.cuda.profiler.start()
    # torch.cuda.synchronize()
    # start_time = time()
    # for _ in range(num_iters):
    #     output_slow = multihead_attn(test_input, impl='slow')
    #     output_slow.sum().backward()
    #     multihead_attn.zero_grad()
    #
    # torch.cuda.synchronize()
    # stop_time = time()
    # print(F"\nSlow Self-Attn time {(stop_time - start_time) * 1000. / num_iters:.4f} ms")
