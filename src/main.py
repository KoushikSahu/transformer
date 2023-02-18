from dataloader import get_dl
from dataset import CharLvlEncodedDataset
from inference import inference
from model import Transformer
from preprocessing import *
from train import train
from utils import *
import constants
import torch.nn as nn
import torch.optim as optim


if __name__ == '__main__':
	text = load_file_asstring(f'{constants.TINYSHAKESPEARE_PATH}/train.csv')[:constants.TRAIN_STR_SIZE]
	vocab = create_vocab(f'{constants.TINYSHAKESPEARE_PATH}/train.csv', f'{constants.TINYSHAKESPEARE_PATH}/test.csv', f'{constants.TINYSHAKESPEARE_PATH}/validation.csv')
	decode_vocab = swapkeyvalues_dict(vocab)

	dataset = CharLvlEncodedDataset(constants.BLOCK_SIZE, text, vocab)
	dl = get_dl(dataset, constants.BATCH_SIZE, True)
	transformer_model = Transformer(constants.BLOCK_SIZE, constants.EMBEDDING_SIZE, len(vocab)).to(constants.DEVICE)
	loss_fn = nn.CrossEntropyLoss()
	optimizer = optim.AdamW(transformer_model.parameters(), lr=constants.MAX_LR) 
	scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=constants.BASE_LR, max_lr=constants.MAX_LR, cycle_momentum=False)

	train(constants.EPOCHS, transformer_model, optimizer, scheduler, loss_fn, dl)
	input_text = 'William Shakespeare'
	inference(input_text, transformer_model, vocab, decode_vocab)
