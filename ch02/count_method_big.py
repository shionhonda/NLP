import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from common.util import most_similar, create_co_matrix, ppmi
from dataset import ptb
from sklearn.utils.extmath import randomized_svd

window_size = 2
wordvec_size = 100


corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
print('counting co-ocurrence ...')
C = create_co_matrix(corpus, vocab_size, window_size)
print('calculating PPMI ...')
W = ppmi(C, verbose=True)

print('calculating SVD ...')
# Truncated SVD
U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5,
                        random_state=None)
print(wordvec_size, len(S), U.shape)
word_vecs = U[:, :wordvec_size]

querys = ['you', 'year', 'beer', 'wine', 'dog', 'cat', 'kick', 'need', 'we',
        'education', 'italy', 'walk', 'quickly', 'big', 'large', 'nine',]
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)

for query in querys:
    query_id = word_to_id[query]
    plt.annotate(query, (U[query_id,0], U[query_id, 1]))
plt.scatter(U[:,0], U[:,1], marker=".", alpha=0.5)
plt.show()
