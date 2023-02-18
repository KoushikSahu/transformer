from tqdm import tqdm
import constants
from model import Transformer


def train(epochs, model, optimizer, scheduler, loss_fn, dl):
	model = model.train()

	for i in range(epochs):
		print(f'***********EPOCH {i+1}**************')
		for x in (itr:=tqdm(dl)):
			inp = x['input'].reshape(-1, constants.BLOCK_SIZE)
			inp = inp.repeat(1, constants.BLOCK_SIZE+1).reshape(-1, constants.BLOCK_SIZE)
			feat = x['features'].reshape(-1, constants.BLOCK_SIZE)
			targ = x['targets'].reshape(-1)

			inp = inp.to(constants.DEVICE)
			feat = feat.to(constants.DEVICE)
			targ = targ.to(constants.DEVICE)

			optimizer.zero_grad()
			out = model(inp, feat)
			loss = loss_fn(out, targ)
			loss.backward()
			optimizer.step()
			scheduler.step()

			itr.set_description(f'Loss: {loss}')
