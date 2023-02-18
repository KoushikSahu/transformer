from torch.utils.data import DataLoader


def get_dl(dataset, batch_size, shuffle):
	return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
