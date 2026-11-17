import torch
import torch.nn as nn

class ImageReconstruction(nn.Module):
    def __init__(self, inputs):
        super(ImageReconstruction, self).__init__()
        dropout_rate = 0.4
        self.input_feature = inputs
        hidden_layers_1 = 1024
        self.fc1 = nn.Linear(inputs, hidden_layers_1)
        self.ln1 = nn.LayerNorm(hidden_layers_1)

        hidden_layers_2 = 512
        self.fc2 = nn.Linear(hidden_layers_1, hidden_layers_2)
        self.ln2 = nn.LayerNorm(hidden_layers_2)

        hidden_layers_3 = 256
        self.fc3 = nn.Linear(hidden_layers_2, hidden_layers_3)
        self.ln3 = nn.LayerNorm(hidden_layers_3)

        hidden_layers_4 = 128
        self.fc4 = nn.Linear(hidden_layers_3, hidden_layers_4)
        self.ln4 = nn.LayerNorm(hidden_layers_4)

        hidden_layers_5 = 64
        self.fc5 = nn.Linear(hidden_layers_4, hidden_layers_5)
        self.ln5 = nn.LayerNorm(hidden_layers_5)

        self.fc6 = nn.Linear(hidden_layers_5, 1)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.drop_out = nn.Dropout(dropout_rate)

        #######################################################
        # Adding Softplus activation layer for the final output
        self.softplus = nn.Softplus()
        #######################################################
        

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.gelu(x)

        x = self.fc2(x)
        x = self.ln2(x)
        x = self.gelu(x)
        x = self.drop_out(x)

        x = self.fc3(x)
        x = self.ln3(x)
        x = self.gelu(x)
        x = self.drop_out(x)

        x = self.fc4(x)
        x = self.ln4(x)
        x = self.gelu(x)
        x = self.drop_out(x)

        x = self.fc5(x)
        x = self.ln5(x)
        x = self.gelu(x)
        x = self.drop_out(x)

        x = self.fc6(x)

        #######################################################
        # Apply Softplus to the output of the last layer
        x = self.softplus(x)
        #######################################################
        
        return x


def make_padding_mask(input_ids, padding_idx=0):
    padding_mask = input_ids.eq(padding_idx)
    if padding_mask.dim() == 3:
        padding_mask = padding_mask.squeeze(-1)
    padding_mask = padding_mask.to(torch.bool)
    return padding_mask if padding_mask.any() else None

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.source_emb = nn.Linear(in_features=1, out_features=256)
        self.target_emb = nn.Linear(in_features=1, out_features=256)
        self.class_token = nn.Parameter(torch.randn(1, 1, 256))

        self.decoderLayer = nn.TransformerDecoderLayer(d_model=256, nhead=4, dim_feedforward=1024, dropout=0.1, activation="gelu", batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoderLayer, num_layers=2)

        self.encoderLayer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=1024, dropout=0.1, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoderLayer, num_layers=2)

        self.r2s_mlp = ImageReconstruction(256)

    def forward(self, src, tgt):
        src_key_padding_mask = make_padding_mask(src, padding_idx=-999)
        tgt_key_padding_mask = make_padding_mask(tgt, padding_idx=-999)

        # Ensure the input tensors are in float32
        src = src.to(torch.float32)
        tgt = tgt.to(torch.float32)

        # Transform the source and target using the embeddings
        src = self.source_emb(src)
        tgt = self.target_emb(tgt)


        # ######################################### add class token ##################################################
        # batch_size = src.size(0)
        # # class_token = repeat(self.class_token, '1 1 d -> b 1 d', b=batch_size)
        # class_token = self.class_token.expand(batch_size, -1, -1)

        # # Adjust target by adding the class token
        # tgt = torch.cat((class_token, tgt), dim=1)

        # # Adjust the tgt_key_padding_mask to account for the added class token
        # # Assuming the class token is not masked (False means it will not be masked because it's a valid token)
        # tgt_key_padding_mask = torch.cat(
        #     (torch.zeros(batch_size, 1, dtype=torch.bool, device=tgt.device), tgt_key_padding_mask), dim=1
        # )
        # ######################################### finish class token ##################################################

        # Process through the encoder and decoder
        # src = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        target = self.decoder(tgt, src, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=src_key_padding_mask)
        # target = self.decoder(tgt, src, tgt_key_padding_mask=tgt_key_padding_mask)

        # Final output
        class_r2s = self.r2s_mlp(target[:, 0, :])
        return class_r2s


# ---------------------------------------------------------------------------------------------
# Faster inference, same result
# ---------------------------------------------------------------------------------------------
# Transformer.forward spends about 94 % of its time in the two decoder layers, and part of that
# work never reaches the output:
#
#   * the R2* head reads token 0 of the last decoder layer only (`target[:, 0, :]`), so the
#     last layer needs to be evaluated for token 0 alone. Its self-attention still looks at
#     all the tokens of the layer before; only the queries are restricted;
#   * the cross-attention keys and values are functions of the echo times, which are the same
#     for every voxel of a scan, so they can be projected once instead of once per voxel.
#
# forward_shared_te does exactly that with the module's own weights and the same arithmetic.
# It is 2.4x faster than forward() and equal to it to rounding (largest difference 0.01 1/s
# over a million in-vivo voxels). It assumes what holds at inference in this repository: one
# TE vector for all voxels, no padding (-999) tokens, dropout off (model.eval()).

def _split_in_proj(mha):
    w, b = mha.in_proj_weight, mha.in_proj_bias
    d = w.shape[1]
    return (w[:d], b[:d]), (w[d:2 * d], b[d:2 * d]), (w[2 * d:], b[2 * d:])


def _to_heads(t, num_heads):
    # [..., tokens, d] -> [..., heads, tokens, d / heads]
    return t.reshape(*t.shape[:-1], num_heads, -1).transpose(-3, -2)


def _keys_values(mha, t):
    _, (wk, bk), (wv, bv) = _split_in_proj(mha)
    return (_to_heads(nn.functional.linear(t, wk, bk), mha.num_heads),
            _to_heads(nn.functional.linear(t, wv, bv), mha.num_heads))


def _attend(mha, queries, keys, values):
    """Multi-head attention of `queries` [B, Tq, d] on already-projected keys/values
    [(B,) heads, Tk, d / heads]; returns [B, Tq, d] after the output projection."""
    (wq, bq), _, _ = _split_in_proj(mha)
    q = _to_heads(nn.functional.linear(queries, wq, bq), mha.num_heads)
    attn = torch.softmax(q @ keys.transpose(-1, -2) / (q.shape[-1] ** 0.5), dim=-1)
    return mha.out_proj((attn @ values).transpose(-3, -2).reshape(queries.shape))


def forward_shared_te(model, te, tgt):
    """model(te expanded to the batch, tgt) for a TE vector shared by every voxel.

    model : Transformer (not wrapped in DataParallel), in eval mode
    te    : [1, n_echo, 1] echo times
    tgt   : [B, n_echo, 1] median-normalised magnitudes
    Returns [B, 1], as Transformer.forward does."""
    memory = model.source_emb(te[0].to(torch.float32))          # [n_echo, d], the same for every voxel
    h = model.target_emb(tgt.to(torch.float32))                 # [B, n_echo, d]
    layers = list(model.decoder.layers)
    for i, layer in enumerate(layers):
        q = h[:, :1] if i == len(layers) - 1 else h             # last layer: token 0 only
        k, v = _keys_values(layer.self_attn, h)
        q = layer.norm1(q + _attend(layer.self_attn, q, k, v))
        k, v = _keys_values(layer.multihead_attn, memory)
        q = layer.norm2(q + _attend(layer.multihead_attn, q, k, v))
        h = layer.norm3(q + layer.linear2(layer.activation(layer.linear1(q))))
    return model.r2s_mlp(h[:, 0, :])

