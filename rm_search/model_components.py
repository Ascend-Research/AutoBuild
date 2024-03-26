import torch
from search.rm_search.utils.model_utils import init_weights, to_sorted_tensor, to_original_tensor, device


class SimpleRNN(torch.nn.Module):

    def __init__(self, input_size, hidden_size,
                 n_layers=1, rnn_dir=2, dropout_prob=0.0, rnn_type="gru",
                 return_aggr_vector_only=False, return_output_vector_only=False,
                 flatten_params=False):
        super(SimpleRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.rnn_dir = rnn_dir
        self.rnn_type = rnn_type
        self.dropout_prob = dropout_prob
        self.return_aggr_vector_only = return_aggr_vector_only
        self.return_output_vector_only = return_output_vector_only
        self.flatten_params = flatten_params
        if self.rnn_type.lower() == "lstm":
            self.rnn = torch.nn.LSTM(input_size=self.input_size,
                                     hidden_size=self.hidden_size,
                                     num_layers=self.n_layers,
                                     dropout=self.dropout_prob if self.n_layers > 1 else 0,
                                     batch_first=True,
                                     bidirectional=self.rnn_dir == 2)
        else:
            self.rnn = torch.nn.GRU(input_size=self.input_size,
                                    hidden_size=self.hidden_size,
                                    num_layers=self.n_layers,
                                    dropout=self.dropout_prob if self.n_layers > 1 else 0,
                                    batch_first=True,
                                    bidirectional=self.rnn_dir == 2)
        init_weights(self, base_model_type="rnn")

    def forward(self, embedded, hidden=None, input_seq_length_tsr=None, input_mask_tsr=None):
        if input_mask_tsr is not None and input_seq_length_tsr is None:
            input_seq_length_tsr = input_mask_tsr.sum(dim=-1).squeeze()
            if len(input_seq_length_tsr.shape) == 0:
                input_seq_length_tsr = input_seq_length_tsr.unsqueeze(0)
        if self.flatten_params:
            self.rnn.flatten_parameters()
        if input_seq_length_tsr is None:
            outputs, hidden = self.rnn(embedded, hidden)
        else:
            embedded, sorted_lens, s_idx = to_sorted_tensor(embedded, input_seq_length_tsr, sort_dim=0)
            packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, sorted_lens, batch_first=True)
            outputs, hidden = self.rnn(packed, hidden)
            outputs, sorted_lens = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            outputs = to_original_tensor(outputs, s_idx, sort_dim=0)

        if self.return_aggr_vector_only:
            if input_seq_length_tsr is not None:
                input_seq_length_tsr = input_seq_length_tsr.to(device())
                input_seq_length_tsr = input_seq_length_tsr - 1
                rv = torch.gather(outputs, 1, input_seq_length_tsr.view(-1, 1).unsqueeze(2).repeat(1, 1, outputs.size(-1)))
            else:
                rv = outputs[:, -1, :].unsqueeze(1)
            return rv
        elif self.return_output_vector_only:
            return outputs
        else:
            return outputs, hidden


class RNNAggregator(torch.nn.Module):

    def __init__(self, aggr_method="last"):
        super(RNNAggregator, self).__init__()
        self.aggr_method = aggr_method

    def forward(self, node_embedding, index=None):
        assert len(node_embedding.shape) == 3
        if self.aggr_method == "sum":
            graph_embedding = node_embedding.sum(dim=1)
        elif self.aggr_method == "last":
            graph_embedding = node_embedding[:, -1, :]
        elif self.aggr_method == "mean":
            graph_embedding = node_embedding.mean(dim=1)
        elif self.aggr_method == "indexed":
            index = index.to(device())
            index = index.reshape(-1, 1, 1).repeat(1, 1, node_embedding.size(-1))
            graph_embedding = torch.gather(node_embedding, 1, index).squeeze(1)
        elif self.aggr_method == "none":
            graph_embedding = node_embedding
        elif self.aggr_method == "squeeze":
            assert len(node_embedding.shape) == 3 and node_embedding.shape[1] == 1, \
                "Invalid input shape: {}".format(node_embedding.shape)
            graph_embedding = node_embedding.squeeze(1)
        elif self.aggr_method == "flat":
            graph_embedding = node_embedding.reshape(node_embedding.shape[0], -1)
        elif self.aggr_method == "de-batch":
            assert len(node_embedding.shape) == 3, "Invalid input shape: {}".format(node_embedding.shape)
            inst_node_embeddings = []
            for bi in range(node_embedding.shape[0]):
                inst_node_embeddings.append(node_embedding[bi, :])
            graph_embedding = torch.cat(inst_node_embeddings, dim=0)
        else:
            raise ValueError("Unknown aggr_method: {}".format(self.aggr_method))
        return graph_embedding
