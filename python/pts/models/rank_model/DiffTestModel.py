import torch
import torch.nn as nn
from typing import List

from pts.models.rank_model.Encoder import Encoder
from pts.models.rank_model.FC_layer import FC_layer


class DiffTestModel(nn.Module):

    def __init__(self, config, embed_size, embedding_store, hidden_size=64, output_size=32, num_layers=1,
                 dropout=0.4, num_heads=16):
        super(DiffTestModel, self).__init__()
        self.config = config
        self.embedding_store = embedding_store
        self.diff_encoder = Encoder(embed_size, hidden_size, num_layers, dropout)
        self.code_encoder = Encoder(embed_size, hidden_size, num_layers, dropout)
        self.self_attention = nn.MultiheadAttention(2 * hidden_size, num_heads, dropout)
        self.diff_attention_transform_matrix = None
        self.fc = FC_layer(2 * hidden_size, hidden_size, output_size, dropout)

        if "additional_features" in self.config:
            self.additional_features: List = self.config["additional_features"]
            self.features_to_output = nn.Linear(self.config["cross_feature_size"] + len(self.additional_features),
                                                          self.config["cross_feature_size"], bias=False)
        else:
            self.additional_features = None

    def compute_attention_states(self, key_states, masks, query_states, transformation_matrix=None,
                                 multihead_attention=None):
        if multihead_attention is not None:
            if transformation_matrix is not None:
                key = torch.einsum('bsh,hd->sbd', key_states, transformation_matrix)  # S x B x D
            else:
                key = key_states.permute(1, 0, 2)  # S x B x D

            query = query_states.permute(1, 0, 2)  # T x B x D
            value = key
            attn_output, attn_output_weights = multihead_attention(query, key, value, key_padding_mask=masks.squeeze(1))
            return attn_output.permute(1, 0, 2)

    def forward(self, batch_data, device):
        # Extract embeddings
        diff_code_embed = self.embedding_store.get_code_embeddings(batch_data.code_seqs)
        pos_test_embed = self.embedding_store.get_code_embeddings(batch_data.pos_test_seqs)
        neg_test_embed = self.embedding_store.get_code_embeddings(batch_data.neg_test_seqs)
        # Encoding layer
        diff_code_rep, _ = self.diff_encoder.forward(diff_code_embed, batch_data.code_lengths, device)
        pos_test_code_rep, _ = self.code_encoder.forward(pos_test_embed, batch_data.pos_test_lengths, device)
        neg_test_code_rep, _ = self.code_encoder.forward(neg_test_embed, batch_data.neg_test_lengths, device)

        diff_code_masks = (torch.arange(
            diff_code_rep.shape[1], dtype=torch.int64, device=device).view(1, -1) >= batch_data.code_lengths.view(-1, 1)).unsqueeze(1)

        pos_code_masks = (torch.arange(
            pos_test_code_rep.shape[1], dtype=torch.int64, device=device).view(1, -1) >= batch_data.pos_test_lengths.view(-1, 1)).unsqueeze(1)
        neg_code_masks = (torch.arange(
            neg_test_code_rep.shape[1], dtype=torch.int64, device=device).view(1, -1) >= batch_data.neg_test_lengths.view(-1, 1)).unsqueeze(1)
        

        
        pos_attention_states = self.compute_attention_states(pos_test_code_rep, pos_code_masks,
                                                         diff_code_rep,
                                                         transformation_matrix=self.diff_attention_transform_matrix,
                                                         multihead_attention=self.self_attention)
        neg_attention_states = self.compute_attention_states(neg_test_code_rep, neg_code_masks,
                                                             diff_code_rep,
                                                             transformation_matrix=self.diff_attention_transform_matrix,
                                                             multihead_attention=self.self_attention)
        # ipdb.set_trace()
        # FC layer
        pos_final_state = torch.mean(pos_attention_states, 1)
        pos_output = self.fc.forward(pos_final_state)
        neg_final_state = torch.mean(neg_attention_states, 1)
        neg_output = self.fc.forward(neg_final_state)

        if self.additional_features:
            pos_output = self.features_to_output(torch.cat([pos_output, batch_data.pos_bm25.unsqueeze(1)], dim=-1))
            neg_output = self.features_to_output(torch.cat([neg_output, batch_data.neg_bm25.unsqueeze(1)], dim=-1))

        return pos_output, neg_output
