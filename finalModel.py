import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from torch import nn
from tqdm import tqdm

from chu_liu_edmonds import decode_mst

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"found device \t:\t {device}")


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def get_df(filename):
    f = open(filename)
    file = f.read().split("\n\n")
    sentences_df = []
    for sen in file:
        if len(sen) == 0:
            continue
        df = pd.DataFrame([row.split('\t') for row in sen.split('\n')],
                          columns=["Token Counter", "Token", "-1", "Token POS", "-2", "-3", "Token Head",
                                   "Dependency Label", "-4", "-5"])
        sentences_df.append(df[["Token", "Token POS", "Token Head"]])
    return sentences_df


class PosEncoding:
    pos_enc_dict = {'JJ': 0, 'CD': 1, ':': 2, 'MD': 3, 'NNS': 4, '.': 5, 'PRP$': 6, '``': 7, 'UH': 8, 'DT': 9, '(': 10,
                    'RB': 11, 'JJR': 12, ')': 13, 'VBG': 14, 'NN': 15, ',': 16, 'RBR': 17, 'NNPS': 18, 'WDT': 19,
                    'VBN': 20, 'VBZ': 21, 'JJS': 22, 'TO': 23, 'IN': 24, 'PRP': 25, 'VB': 26, 'VBP': 27, 'SYM': 28,
                    'NNP': 29, 'WP$': 30, 'FW': 31, "''": 32, 'CC': 33, 'RBS': 34, 'LS': 35, 'VBD': 36, 'RP': 37,
                    'EX': 38, '#': 39, '$': 40, 'PDT': 41, 'WRB': 42, 'POS': 43, 'WP': 44}
    embedding_dim = len(pos_enc_dict)

    def __getitem__(self, item):
        try:
            lis = 45 * [0]
            lis[self.pos_enc_dict[item]] = 1
            return np.array(lis)
        except Exception as e:
            print("pos encoding error")
            print(e)


class BiLstm(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, batch_first=True,
                            num_layers=self.num_layers, bidirectional=True)

    def forward(self, sentence):
        lstm_out, _ = self.lstm(torch.stack([torch.tensor(word, dtype=torch.float32).to(device) for word in sentence]))
        return lstm_out


class DependencyParser(nn.Module):

    def __init__(self, hidden_dim):
        super(DependencyParser, self).__init__()
        self.GLOVE_FILE = 'glove.6B.300d.txt'
        self.word_embedding = dict(get_coefs(*o.strip().split()) for o in tqdm(open(self.GLOVE_FILE,
                                                                                    encoding='utf-8')))  # TODO: Implement embedding layer for words (can be new or pretrained - word2vec/glove)
        self.pos_embedding = PosEncoding()
        self.hidden_dim = hidden_dim
        self.encoder = BiLstm(embedding_dim=345,
                              hidden_dim=self.hidden_dim,
                              num_layers=4)  # Implement BiLSTM module which is fed with word embeddings and outputs hidden representations # TODO: change num_layers to 4

        self.in_linear1 = nn.Linear(self.hidden_dim * 4, 256) # TODO: change the network architecture
        self.nl1 = nn.ReLU()
        self.mid_linear1 = nn.Linear(256, 128)
        self.nl2 = nn.ReLU()
        self.mid_linear2 = nn.Linear(128, 64)
        self.nl3 = nn.ReLU()
        self.out_linear = nn.Linear(64, 1)
        self.network = nn.Sequential(self.in_linear1,
                                     self.nl1,
                                     self.mid_linear1,
                                     self.nl2,
                                     self.mid_linear2,
                                     self.nl3,
                                     self.out_linear).to(device)
        self.mst_predictor = decode_mst
        self.loss_fn = torch.nn.MSELoss().to(device)  # Implement the loss function described above

    def forward(self, sentence_forward):
        words = sentence_forward["Token"].values
        poses = sentence_forward["Token POS"].values
        true_tree_heads = sentence_forward["Token Head"].values
        true_tree_heads = torch.tensor([-1] + [int(x) for x in true_tree_heads])

        # Pass word and pos through their embedding layer
        emb_words = [
            self.word_embedding.get(word) if self.word_embedding.get(word) is not None else self.word_embedding['word']
            for word in words]  # TODO : try to change the embedding to deal with OOV
        emb_poses = [self.pos_embedding[pos] for pos in poses]

        # Get Bi-LSTM hidden representation for each word in sentence
        lstm_output = self.encoder(
            [np.concatenate((emb_word, emb_pos)) for (emb_word, emb_pos) in zip(emb_words, emb_poses)])

        available_words = torch.row_stack((torch.zeros(self.hidden_dim*2).to(device), lstm_output.to(device))) # zeros is the representation of the [ROOT] # TODO : the [ROOT] does not go through the lstm, you can try passing it through

        # Get score for each possible edge in the parsing graph, construct score matrix
        crit = self.loss_fn
        loss_forward = None
        score_matrix = torch.zeros((len(available_words), len(available_words)), dtype=torch.float32)
        for idx1, word1 in enumerate(available_words):
            for idx2, word2 in enumerate(available_words):
                target_output = torch.tensor([100.]).to(device) if idx2 == true_tree_heads[idx1] else torch.tensor(
                    [0.]).to(device)
                network_input = torch.cat((word1, word2)).to(device)
                output = self.network(network_input)
                if loss_forward is None:
                    loss_forward = crit(output.to(device), target_output.to(device))
                else:
                    loss_forward += crit(output.to(device), target_output.to(device))
                score_matrix[idx1, idx2] = output.item()

        loss_forward = loss_forward / (len(available_words) ** 2)
        score_matrix = softmax(score_matrix, axis=1)

        # get the Max-spanning tree
        T_forward = decode_mst(score_matrix.T, len(available_words), has_labels=False)

        return loss_forward, T_forward


train_ds = get_df(r'train.labeled')
eval_ds = get_df(r'test.labeled')
EPOCHS = range(20)
model = DependencyParser(250).to(device) # TODO :  change the embedding dim from 100 (200/250 for example)
optim = torch.optim.Adam(model.parameters(), lr=0.01)
# model.load_state_dict(torch.load(r'night_model_after_3_epoch', map_location=torch.device('cpu'))) # TODO : remove this ! we want to train from scartch

for epoch in EPOCHS:
    losses = []
    accuracies = []
    for idx, sentence in tqdm(enumerate(train_ds)):
        optim.zero_grad()
        loss, T = model(sentence)
        loss.backward()
        optim.step()

        accuracy = np.sum(T[0][1:] == np.array([int(x) for x in sentence["Token Head"].values])) / T[0].size

        losses.append(loss.item())
        accuracies.append(accuracy)
        if idx % 100 == 99:
            # plt.plot(losses[-100:], label="losses")
            # plt.show()
            # plt.plot(accuracies[-100:], label="accuracies")
            # plt.show()
            print("\n*****************************************************************")
            print(f"average loss in the last 100 sentences loss is : {sum(losses[-100:]) / 100}")
            print(f"average accuracy in the last 100 sentences loss is : {sum(accuracies[-100:]) / 100}")
            print(f" last tree was {T}")
            print("*****************************************************************")
    torch.save(model.state_dict(), f"final_model_after_{epoch}_epoch")
    accuracies = []
    for idx, sentence in tqdm(enumerate(eval_ds)):
        loss, T = model(sentence)
        accuracy = np.sum(T[0][1:] == np.array([int(x) for x in sentence["Token Head"].values])) / T[0].size
        accuracies.append(accuracy)

    total_acc = sum(accuracies) / len(accuracies)
    if total_acc > 0.7:
        print(f"V-V-V-V-V epoch {epoch} with accuracy \t {total_acc} ***** w7sh")
    else:
        print(f"*X*X*X*X epoch {epoch} with accuracy \tonly got {total_acc}")
