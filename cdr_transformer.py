import torch
import torch.nn as nn

# Transformer module -------------
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        # Get number of training examples
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        queries = self.queries(query)  # (N, query_len, heads, heads_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

        # Mask padded indices so their weights become 0
        if mask is not None:
            # when src mask is added to decoderblock, here is where dims don't match
            energy = energy.masked_fill(mask == False, float("-1e20")) # mask == -1, mask is already boolean

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)

        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()

        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size) # one layer normalization
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask = None):

        attention = self.attention(value, key, query, mask) # encoder receives source mask
        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query)) # skip connection online transformer example
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class Encoder(nn.Module):
    def __init__(
            self,
            src_trajectory_len,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            coord_dim # 2 by default, longitude and latitude
    ):

        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.spatial_feature_extraction = nn.Sequential(
            nn.Linear(coord_dim, embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.temporal_embedding = nn.Embedding(num_embeddings = src_trajectory_len, embedding_dim=embed_size)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x,  mask = None):
        N = x.shape[0]
        seq_length = x.shape[1]
        temporal_input = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(
            (self.spatial_feature_extraction(x) + self.temporal_embedding(temporal_input))
        )
        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case.
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out


class CellEncoder(nn.Module):
    def __init__(
            self,
            src_trajectory_len,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            no_of_link
    ):

        super(CellEncoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.embedding =  nn.Embedding(no_of_link + 1, embed_size)
        self.temporal_embedding = nn.Embedding(num_embeddings = src_trajectory_len, embedding_dim=embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x,  mask = None):
        N = x.shape[0]
        seq_length = x.shape[1]
        temporal_input = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(
            (self.embedding(x) + self.temporal_embedding(temporal_input))
        )
        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case.
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(DecoderBlock, self).__init__()
        self.layernorm = nn.LayerNorm(embed_size) # one layer normalization
        self.attention = SelfAttention(embed_size, heads)

        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask = None, trg_mask = None): # x is output from decoder input embedding
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.layernorm(attention + x)) # skip connection
        out = self.transformer_block(value, key, query)  # src mask is optional here and should be considered for
        # computation purposes
        return out

class Decoder(nn.Module):
    def __init__(
            self,
            tgt_trajectory_len,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout, # NEEDS TO BE REMOVED FOR EVAL MODE  https://stackoverflow.com/questions/53879727/pytorch-how-to-deactivate-dropout-in-evaluation-mode
            no_of_link, # no of links - max number of road segments
            coord_dim # 2 by default, longitude and latitude
    ):

        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        # No spatial feature extraction in decoder, which receives the encoded source
        # and the target, which is a one-dimensional array
        self.spatial_feature_extraction = nn.Sequential(
            nn.Linear(coord_dim, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size)
        )

        self.embedding =  nn.Embedding(no_of_link + 1, embed_size)
        self.temporal_embedding = nn.Embedding(num_embeddings = tgt_trajectory_len, embedding_dim=embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, no_of_link + 1) # number of link +1 from paper, because of padding tokens in input trajectory
        self.dropout = nn.Dropout(dropout)
        self.sf_max = nn.Softmax(dim = 1)

    def forward(self, x, enc_out, src_mask, trg_mask = None):
        N = x.shape[0]
        seq_length = x.shape[1]
        temporal_input = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        out = self.dropout(
            (self.embedding(x) + self.temporal_embedding(temporal_input))
        )

        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case.
        for layer in self.layers:
            out = layer(out, enc_out, enc_out, src_mask = src_mask, trg_mask = trg_mask)
        out = self.fc_out(out)
        out = self.sf_max(out)

        return out

class CellDecoder(nn.Module):
    def __init__(
            self,
            tgt_trajectory_len,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            no_of_link,
    ):

        super(CellDecoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.embedding =  nn.Embedding(no_of_link + 1, embed_size)

        self.temporal_embedding = nn.Embedding(num_embeddings = tgt_trajectory_len, embedding_dim=embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, no_of_link + 1)
        self.dropout = nn.Dropout(dropout)
        self.sf_max = nn.Softmax(dim = 1)

    def forward(self, x, enc_out, src_mask, trg_mask = None):
        N = x.shape[0]
        seq_length = x.shape[1]
        temporal_input = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(
            (self.embedding(x) + self.temporal_embedding(temporal_input))
        )

        for layer in self.layers:
            out = layer(out, enc_out, enc_out, src_mask = src_mask, trg_mask = trg_mask)
        out = self.fc_out(out)
        out = self.sf_max(out)

        return out

class Transformer(nn.Module):
    def __init__(
            self,
            src_trajectory_len,
            tgt_trajectory_len,
            src_pad_idx,
            trg_pad_idx,
            embed_size=512,
            num_layers=6,
            forward_expansion=4,
            heads=8,
            dropout=0,
            device=None,
            no_of_link = 15,
            coord_dim = 2,
            batch_size = 1,
            no_masking = False,
            max_batch_size = 1,
            cell_transformer = False # if True, switches to different encoder and decoder architecture for cell id and road id mapmatching
    ):

        super(Transformer, self).__init__()
        if cell_transformer:
            self.encoder = CellEncoder(
                src_trajectory_len,
                embed_size,
                num_layers,
                heads,
                device,
                forward_expansion,
                dropout,
                coord_dim,
            )
        else:
            self.encoder = Encoder(
                src_trajectory_len,
                embed_size,
                num_layers,
                heads,
                device,
                forward_expansion,
                dropout,
                coord_dim,
            )
        if cell_transformer:
            self.decoder = CellDecoder(
                tgt_trajectory_len,
                embed_size,
                num_layers,
                heads,
                device,
                forward_expansion,
                dropout,
                no_of_link
            )
        else:
            self.decoder = Decoder(
                tgt_trajectory_len,
                embed_size,
                num_layers,
                heads,
                device,
                forward_expansion,
                dropout,
                no_of_link,
                coord_dim,
            )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        self.coord_dim = coord_dim
        self.batch_size = batch_size
        self.src_trajectory_len = src_trajectory_len
        self.no_masking = no_masking
        self.cell_transformer = cell_transformer
        self.max_batch_size = max_batch_size

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx)
        # we have 2 types of cases, 1 dimensional source (cell or road id), or 2 dimensional source ( gps trajectory):
        if not self.cell_transformer:
            src_mask = src_mask[:,:, 0] # presuming that source pad index is for x and y set up correctly,
            # that is, both X and Y are set to source pad index
        src_mask = src_mask.repeat(self.src_trajectory_len, 1).unsqueeze(0).unsqueeze(0) # row gets canceled out, not column

        # from pytorch multiheadattention: For a binary mask, a True value indicates that the corresponding key value will be ignored for the purpose of attention.
        # If we want to send in by batch, src mask must have the dimensions of batch_size, 1, trajectory_len
        src_mask = src_mask.reshape(src.shape[0], 1, self.src_trajectory_len, self.max_batch_size)

        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N = trg.shape[0]
        trg_len = trg.shape[1]

        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)


    def forward(self, src, trg):
        if self.no_masking:
            src_mask = None
            trg_mask = None
        else:
            src_mask = self.make_src_mask(src)
            trg_mask = self.make_trg_mask(trg) #None

        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask = src_mask, trg_mask = trg_mask)

        return out
