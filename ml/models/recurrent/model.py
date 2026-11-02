"""Padding-aware recurrent classifiers."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RecurrentClassifier(nn.Module):
    def __init__(self, architecture, vocab_size, embedding_dim=64, hidden_dim=64, classes=4, dropout=.2):
        super().__init__()
        if architecture not in ('rnn','lstm','gru','bilstm','bilstm_attention'): raise ValueError('Unknown architecture')
        self.architecture=architecture
        self.embedding=nn.Embedding(vocab_size,embedding_dim,padding_idx=0)
        bidirectional=architecture.startswith('bilstm')
        cls={'rnn':nn.RNN,'lstm':nn.LSTM,'gru':nn.GRU,'bilstm':nn.LSTM,'bilstm_attention':nn.LSTM}[architecture]
        self.recurrent=cls(embedding_dim,hidden_dim,batch_first=True,bidirectional=bidirectional)
        dimension=hidden_dim*(2 if bidirectional else 1)
        self.attention=nn.Linear(dimension,1) if architecture=='bilstm_attention' else None
        self.dropout=nn.Dropout(dropout)
        self.head=nn.Linear(dimension,classes)

    def forward(self, ids, return_attention=False):
        if ids.ndim!=2 or ids.shape[1]==0: raise ValueError('Expected a nonempty 2-D token tensor')
        mask=ids.ne(0); lengths=mask.sum(1); safe=lengths.clamp(min=1)
        if (mask[:,1:] & ~mask[:,:-1]).any(): raise ValueError('Padding must be on the right')
        packed=pack_padded_sequence(self.embedding(ids),safe.cpu(),batch_first=True,enforce_sorted=False)
        output,state=self.recurrent(packed)
        weights=None
        if self.attention is not None:
            padded,_=pad_packed_sequence(output,batch_first=True,total_length=ids.shape[1])
            scores=self.attention(padded).squeeze(-1).masked_fill(~mask,torch.finfo(padded.dtype).min)
            weights=torch.softmax(scores,dim=1)*mask
            weights=weights/weights.sum(1,keepdim=True).clamp_min(1e-12)
            representation=(padded*weights.unsqueeze(-1)).sum(1)
        else:
            hidden=state[0] if isinstance(state,tuple) else state
            representation=torch.cat((hidden[-2],hidden[-1]),dim=1) if self.recurrent.bidirectional else hidden[-1]
            representation=representation*lengths.ne(0).unsqueeze(1)
        logits=self.head(self.dropout(representation))
        return (logits,weights) if return_attention else logits
