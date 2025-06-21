def read_words(path):
    words = open(path, 'r').read().splitlines()
    return words


def get_mapping(words):
    chars = sorted(list(set(''.join(words))))
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i:s for s,i in stoi.items()}
    return stoi, itos

def build_dataset(words, stoi, block_size=1):
    X = []
    Y = []

    for word in words:
        s = ('.' * block_size) + word + '.'
        for i in range(len(s) - block_size):
            X.append([stoi[c] for c in s[i:i+block_size]])
            Y.append(stoi[s[i+block_size]])
    return X, Y



