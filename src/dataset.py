from torch.utils.data import Dataset
import torch

class CharLvlEncodedDataset(Dataset):
	def __init__(self, block_size, text, vocab):
		self.block_size = block_size
		self.text = text
		self.vocab = vocab

	def __len__(self):
		return len(self.text) - self.block_size

	def __getitem__(self, idx):
		curr_text = self.text[idx:idx+self.block_size+1]
		tokenized_text = list(map(lambda x: self.vocab[x], curr_text))

		feats = list()
		tgts = list()
		for i in range(0, self.block_size+1):
			feat = tokenized_text[:i]
			feat += [0]*(self.block_size-i)
			feats.append(feat)

			tgt = tokenized_text[i]
			tgts.append(tgt)

		return {
			'input': torch.tensor(tokenized_text[:self.block_size]),
			'features': torch.tensor(feats),
			'targets': torch.tensor(tgts)
		}
