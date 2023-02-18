def load_file(filepath):
	with open(filepath, 'rb') as f:
		file_content = f.read()

	return file_content

def load_file_asstring(filepath):
	f = load_file(filepath)
	return f.decode()

def create_vocab(*filepaths):
	unique_chars = set()

	for fp in filepaths:
		filecontent = load_file_asstring(fp)

		for i in filecontent:
			unique_chars.add(i)

	vocab = dict()
	for idx, val in enumerate(sorted(unique_chars)):
		vocab[val] = idx+1
	return vocab
