import numpy as np

def build_word_embeddings(vocab, pretrained_embedding, weights_output_file):
    # Load 预训练的embedding
    lines = open(pretrained_embedding, "r", encoding="utf8").readlines()
    emb_dict = dict()
    error_line = 0
    embed_size = 0
    for line in lines:
        row = line.strip().split()
        try:
            embedding = [float(w) for w in row[1:]]
            emb_dict[row[0]] = np.array(embedding)
            if embed_size == 0:
                embed_size = len(embedding)
        except:
            error_line += 1
    print("Error lines: {}".format(error_line))

    weights_matrix = np.zeros((len(vocab), embed_size))
    words_found = 0

    for k, v in vocab.items():
        try:
            weights_matrix[v] = emb_dict[k]
            words_found += 1
        except KeyError:
            weights_matrix[v] = np.random.normal(size=(embed_size,))
    print("Totally find {} words in pre-trained embeddings.".format(words_found))
    np.save(weights_output_file, weights_matrix)
    print(weights_matrix.shape)

def build_news_embeddings(news_dict, output_file):
    title_len = len(news_dict['<pad>']['title'])
    title_matrix = np.zeros((len(news_dict), title_len))

    for k, v in news_dict.items():
        title_matrix[v['idx']] = np.array(v['title'])
    
    print('news embedding', title_matrix.shape)
    np.save(output_file, title_matrix)