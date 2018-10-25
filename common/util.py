import numpy as np

def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            id = len(word_to_id)
            word_to_id[word] = id
            id_to_word[id] = word

    corpus = np.array([word_to_id[word] for word in words])

    return corpus, word_to_id, id_to_word

def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size,vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size+1):
            left_idx = idx-i
            right_idx = idx+i

            if left_idx>=0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1
            if right_idx<corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix

def cos_similarity(x, y, eps=1e-8):
    nx = x/(np.sqrt(np.sum(x**2))+eps)
    ny = y/(np.sqrt(np.sum(y**2))+eps)
    return np.dot(nx, ny)

def most_similar(query, word_to_id, id_to_word, co_matrix, top=5):
    if query not in word_to_id:
        print("%s is not found" % query)
        return

    print("\n[query] " + query)
    query_id = word_to_id[query]
    query_vec = co_matrix[query_id]

    vocab_size = len(word_to_id)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(query_vec, co_matrix[i])

    cnt = 0
    for i in np.argsort(similarity)[::-1]:
        if id_to_word[i]==query:
            continue
        print(" %s: %s" % (id_to_word[i], similarity[i]))
        cnt += 1
        if cnt>=top:
            return
