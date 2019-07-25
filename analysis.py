import os
import pickle


d = './cifar10_results'
f = 'cifar10_sorted.pkl'

with open(os.path.join(d, f), 'rb') as fin:
    loaded = pickle.load(fin)

print(loaded)
with open('indices.txt', 'w+') as f:
    for x in loaded['indices']:
        f.write(str(x) + '\n')

with open('counts.txt', 'w+') as f:
    for x in loaded['forgetting counts']:
        f.write(str(x) + '\n')
