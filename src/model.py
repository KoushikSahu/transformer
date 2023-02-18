import math
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
	def __init__(self, max_seq_len, embed_model_dim):
		"""
		Args:
			seq_len: length of input sequence
			embed_model_dim: demension of embedding
		"""
		super(PositionalEmbedding, self).__init__()
		self.embed_dim = embed_model_dim

		pe = torch.zeros(max_seq_len, self.embed_dim)
		for pos in range(max_seq_len):
			for i in range(0, self.embed_dim, 2):
				pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.embed_dim)))
				pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/self.embed_dim)))
		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)


	def forward(self, x):
		"""
		Args:
			x: input vector
		Returns:
			x: output
		"""
	  
		# make embeddings relatively larger
		x = x * math.sqrt(self.embed_dim)
		#add constant to embedding
		seq_len = x.size(1)
		x = x + torch.autograd.Variable(self.pe[:,:seq_len], requires_grad=False)
		return x


class Transformer(nn.Module):
	def __init__(self, max_seq_len, embed_model_dim, vocab_size):
		super(Transformer, self).__init__()
		self.max_seq_len = max_seq_len
		self.embed_model_dim = embed_model_dim
		self.vocab_size = vocab_size

		self.pe = PositionalEmbedding(max_seq_len, embed_model_dim)
		self.embd1 = nn.Embedding(vocab_size+1, embed_model_dim)
		self.embd2 = nn.Embedding(vocab_size+1, embed_model_dim)
		self.transformer = nn.Transformer(embed_model_dim)
		self.linear = nn.Linear(max_seq_len*embed_model_dim, vocab_size+1)

	def forward(self, inps, outs):
		inps = self.pe(self.embd1(inps))
		outs = self.pe(self.embd2(outs))

		transformer_out = self.transformer(inps, outs).reshape(-1, self.max_seq_len*self.embed_model_dim)
		lin_out = self.linear(transformer_out)
		return lin_out
