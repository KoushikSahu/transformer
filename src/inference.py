import constants
import torch


def inference(input_text, model, vocab, decode_vocab, max_gen_len=50):
	model = model.eval()

	inp = list()
	targ = list()
	for i in range(min(constants.BLOCK_SIZE, len(input_text))):
		print(input_text[i], end='')
		inp.append(vocab[input_text[i]])

	for i in range(min(constants.BLOCK_SIZE, len(input_text)-1)):
		targ.append(vocab[input_text[i+1]])

	for _ in range(max_gen_len):
		inp_tensor = inp + [0] * (constants.BLOCK_SIZE - len(inp))
		inp_tensor = torch.tensor(inp_tensor).long().to(constants.DEVICE)
		targ_tensor = targ + ([0] * (constants.BLOCK_SIZE - len(targ)))
		targ_tensor = torch.tensor(targ_tensor).long().to(constants.DEVICE)
		
		out = model(inp_tensor, targ_tensor)
		pred = torch.argmax(out, dim=1)
		pred = pred.cpu().detach().numpy()[0]
		pred_char = decode_vocab[pred]
		print(pred_char, end='')

		inp = targ
		targ.append(pred)
		if len(targ) > constants.BLOCK_SIZE:
			targ.pop(0)

	print()
