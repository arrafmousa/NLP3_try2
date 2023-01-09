import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
from scipy.special import softmax
from chu_liu_edmonds import decode_mst

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, device, optimizer, train_dataset, val_dataset):
    accuracies = []
    for phase in ["train", "validation"]:
        phase_accuracy = []
        if phase == "train":
            model.train(True)
        else:
            model.train(False)  # or model.evel()
        correct = 0.0
        count = 0
        accuracy = None
        dataset = train_dataset if phase == "train" else val_dataset
        t_bar = tqdm(dataset)
        idx = 0
        phase_buffer = []
        for sentence in t_bar:
            accuracy = 0.0
            if phase == "train":
                loss, T = model(sentence)
                accuracy = np.sum(T[0][1:] == np.array([int(x) for x in sentence["Token Head"].values])) / T[0].size
                phase_buffer.append(accuracy)
                try:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                except Exception as e:
                    print("----------------------------------------")
                    print(e)
                    print(sentence)
                    print("was not backpropagated ")
                    print("----------------------------------------")
                if idx % 10 == 9:
                    phase_accuracy.append(sum(phase_buffer[-10:]) / 10)
                    phase_buffer = []
                if idx % 100 == 99:
                    # plt.plot(phase_accuracy)
                    # plt.show()
                    print(f"phase accuracy \t:\t {phase_accuracy[-10:]}")
                    t_bar.set_description(f"{phase} accuracy: {phase_accuracy[-1]:.2f}")
                # if idx % 500 == 499:
                #     print("BEGIN DEBUG")
                idx += 1
            else:
                with torch.no_grad():
                    loss, T = model(sentence)
        accuracies += [accuracy]
        print(sum(phase_accuracy) / len(phase_accuracy))
    return accuracies


class Scorer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.in_linear = nn.Linear(2 * self.hidden_dim, 100).to(device)
        self.nl1 = nn.Sigmoid().to(device)
        self.out_linear = nn.Linear(100, 1).to(device)

    def forward(self, sentence, target_scores: torch.tensor):
        crit = nn.L1Loss()
        score_matrix = torch.zeros((len(sentence), len(sentence) + 1))
        potential_matches = torch.row_stack((torch.zeros(self.hidden_dim), sentence))  # add the embedding of zero
        loss = None
        for idx1 in range(len(sentence)):
            for idx2, _ in enumerate(potential_matches):
                target_output = torch.tensor([1]) if idx2 == target_scores[idx1] else torch.tensor([0])
                network_input = torch.cat((sentence[idx1], potential_matches[idx2])).to(device)
                x = self.in_linear(network_input)
                x = self.nl1(x)
                output = self.out_linear(x)
                score_matrix[idx1, idx2] = output.item()
                if loss is None:
                    loss = crit(output, target_output)
                else:
                    loss += crit(output, target_output)

        torch_softmax = nn.Softmax(dim=1)
        score_matrix = torch_softmax(score_matrix)

        return loss / (len(potential_matches) * len(sentence)), score_matrix


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
                            num_layers=self.num_layers)

    def forward(self, sentence):
        lstm_out, _ = self.lstm(torch.stack([torch.tensor(word, dtype=torch.float32) for word in sentence]))
        return lstm_out


class DependencyParser(nn.Module):
    def get_coefs(self, word, *arr):
        return word, np.asarray(arr, dtype='float32')

    def __init__(self, hidden_dim):
        super(DependencyParser, self).__init__()
        self.GLOVE_FILE = 'glove.6B.300d.txt'
        self.word_embedding = dict(self.get_coefs(*o.strip().split()) for o in tqdm(open(self.GLOVE_FILE,
                                                                                         encoding='utf-8')))  # TODO: copy embedding model # Implement embedding layer for words (can be new or pretrained - word2vec/glove)
        self.hidden_dim = hidden_dim
        # self.pos_embedding = PosEncoding()
        # self.encoder = BiLstm(embedding_dim=345,
        self.encoder = BiLstm(embedding_dim=300,
                              hidden_dim=self.hidden_dim,
                              num_layers=2)  # Implement BiLSTM module which is fed with word embeddings and outputs hidden representations
        self.edge_scorer = Scorer(
            self.hidden_dim)  # Scorer model# Implement a sub-module to calculate the scores for all possible edges in sentence dependency graph
        self.mst_predictor = decode_mst
        self.loss_fn = nn.NLLLoss()  # Implement the loss function described above

    def forward(self, sentence):
        words = sentence["Token"].values
        poses = sentence["Token POS"].values
        true_tree_heads = sentence["Token Head"].values
        true_tree_heads = torch.tensor([-1] + [int(x) for x in true_tree_heads])
        # target_score_matrix = np.array(
        #     [[1. if true_tree_heads[word] == i else 0. for i in range(len(true_tree_heads))] for word
        #      in true_tree_heads]).T.tolist()

        # Pass word_idx through their embedding layer
        emb_words = [
            self.word_embedding.get(word) if self.word_embedding.get(word) is not None else self.word_embedding['word']
            for word in words]
        # emb_poses = [self.pos_embedding[pos] for pos in poses]

        # Get Bi-LSTM hidden representation for each word in sentence

        # lstm_output = self.encoder(
        #     [np.concatenate((emb_word, emb_pos)) for (emb_word, emb_pos) in zip(emb_words, emb_poses)])
        lstm_output = self.encoder(emb_words)
        lstm_output = torch.tensor(emb_words)

        # Get score for each possible edge in the parsing graph, construct score matrix
        # test_tree_heads = torch.tensor([1]+[int(0) for x in range(len(true_tree_heads))])
        loss, score_mat = self.edge_scorer(lstm_output, true_tree_heads[1:])

        # get the Max-spanning tree
        T = decode_mst(np.row_stack((np.zeros(len(sentence) + 1), np.array(score_mat.clone().detach()))).T,
                       len(sentence) + 1, has_labels=False)

        return loss, T, score_mat


class Trainer:
    model = DependencyParser(hidden_dim=300)
    optimizer = Adam(model.edge_scorer.parameters(), lr=0.01)
    EPOCHS = 10

    def get_df(self, filename):
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

    def train(self):
        train_ds = self.get_df(r'train.labeled')
        eval_ds = self.get_df(r'test.labeled')

        # train scorer
        optimizer = Adam(self.model.edge_scorer.parameters(), lr=0.1)

        for epoch in range(self.EPOCHS):
            accuracies = []
            self.model.edge_scorer.train()
            # self.model.load_state_dict(torch.load("basic_epoch_4_samples_3999"))
            for idx, sentence in tqdm(enumerate(train_ds)):
                optimizer.zero_grad()
                loss, T, score_mat = self.model(sentence)
                loss.backward()
                optimizer.step()
                accuracy = np.sum(T[0][1:] == np.array([int(x) for x in sentence["Token Head"].values])) / T[0].size
                if accuracy > 0.7:
                    print(f"******   {accuracy}")
                    print(sentence)
                    print(T)
                accuracies.append(accuracy)
                idx += 1
                if idx % 1000 == 999:
                    torch.save(self.model.state_dict(), f"basic_epoch_{epoch}_samples_{idx}")
                    plt.plot(accuracies)
                    plt.plot(len(accuracies) * [sum(accuracies[-100:]) / len(accuracies[-100:])])
                    plt.show()
                # if idx % 500 == 499:
                #     print("BEGIN DEBUG")



trainer = Trainer()
trainer.train()
