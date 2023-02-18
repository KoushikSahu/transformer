def prettyprint_bytes(b):
	print(b.decode())


def swapkeyvalues_dict(d):
	new_dict = dict()

	for key, value in d.items():
		new_dict[value] = key

	return new_dict
