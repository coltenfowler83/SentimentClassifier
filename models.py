# models.py

import time
import random
import math
import torch
from torch import nn
import embeddings as E
from sentiment_data import *
from collections import Counter, defaultdict


class FeatureExtractor:

    def extract_features(self, ex_words: List[str]) -> List[int]:
        raise NotImplementedError()


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """

    def extract_features(self, ex_words):
        """
        Q1: Implement the unigram feature extractor.
        Hint: You may want to use the Counter class.
        """
        cnt = Counter()
        cnt['<bias>'] = 1
        for word in ex_words:
            cnt[word.lower()] += 1
        return cnt


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def extract_features(self, ex_words):
        """
        Q3: Implement the unigram feature extractor.
        Hint: You may want to use the Counter class.
        """

        cnt = Counter()
        cnt['<bias>'] = 1
        # use unigram counts and bigram counts as features
        for word in ex_words:
            cnt[word.lower()] += 1
        for word1, word2 in zip(ex_words, ex_words[1:]):
            cnt[(word1.lower(), word2.lower())] += 1
        return cnt

class SentimentClassifier(object):

    def featurize(self, ex):
        raise NotImplementedError()

    def forward(self, feat):
        raise NotImplementedError()

    def extract_pred(self, output):
        raise NotImplementedError()

    def update_parameters(self, output, feat, ex, lr):
        raise NotImplementedError()

    def run_train(self, train_data: List[SentimentExample], dev_data: List[SentimentExample], lr=1e-3, epoch=10):
        """
        Training loop.
        """
        train_data = train_data[:]
        for ep in range(epoch):
            start = time.time()
            random.shuffle(train_data)

            if isinstance(self, nn.Module):
                self.train()

            acc = []
            for ex in train_data:
                feat = self.featurize(ex)
                output = self.forward(feat)
                self.update_parameters(output, feat, ex, lr)
                predicted = self.extract_pred(output)
                acc.append(predicted == ex.label)
            acc = sum(acc) / len(acc)

            if isinstance(self, nn.Module):
                self.eval()

            dev_acc = []
            for ex in dev_data:
                feat = self.featurize(ex)
                output = self.forward(feat)
                predicted = self.extract_pred(output)
                dev_acc.append(predicted == ex.label)
            dev_acc = sum(dev_acc) / len(dev_acc)
            print('epoch {}: train acc = {}, dev acc = {}, time = {}'.format(ep, acc, dev_acc, time.time() - start))

    def predict(self, ex: SentimentExample) -> int:
        feat = self.featurize(ex)
        output = self.forward(feat)
        return self.extract_pred(output)


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, ex: SentimentExample) -> int:
        return 1

    def run_train(self, train_data: List[SentimentExample], dev_data: List[SentimentExample], lr=None, epoch=None):
        pass


class PerceptronClassifier(SentimentClassifier):
    """
    Q1: Implement the perceptron classifier.
    """

    def __init__(self, feat_extractor):
        self.feat_extractor = feat_extractor
        self.weights = Counter()
        self.activation_threshold = 0.5

    def featurize(self, ex):
        """
        Converts an example into features.
        """
        return self.feat_extractor.extract_features(ex.words)

    def forward(self, feat) -> float:
        # compute the activation of the perceptron
        sum = 0
        for word, count in feat.items():
            sum += count * self.weights[word]
        return sum

    def extract_pred(self, output) -> int:
        # compute the prediction of the perceptron given the activation
        return 1 if output > self.activation_threshold else 0

    def update_parameters(self, output, feat, ex, lr):
        # update the weight of the perceptron given its activation, the input features, the example, and the learning rate
        for word, count in feat.items():
            self.weights[word] += lr * (ex.label - output) * count


class FNNClassifier(SentimentClassifier, nn.Module):
    """
    Q4: Implement the multi-layer perceptron classifier.
    """

    def __init__(self, args):
        super().__init__()
        self.glove = E.GloveEmbedding('wikipedia_gigaword', 300, default='zero')
        ### Start of your code

        self.fc1 = nn.Linear(300, 100)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()

        ### End of your code

        # do not touch this line below
        self.optim = torch.optim.Adam(self.parameters(), args.learning_rate)

    def featurize(self, ex):
        # You do not need to change this function
        # return a [T x D] tensor where each row i contains the D-dimensional embedding for the ith word out of T words
        embs = [self.glove.emb(w.lower()) for w in ex.words]
        return torch.Tensor(embs)

    def forward(self, feat) -> torch.Tensor:
        # compute the activation of the FNN
        feat = feat.unsqueeze(0)

        sum = torch.sum(feat, dim=1)
        hidden = self.fc1(sum)
        tanh = self.tanh(hidden)
        output = self.fc2(tanh)
        output = self.sigmoid(output)
        return output

    def extract_pred(self, output) -> int:
        # compute the prediction of the FNN given the activation
        return 1 if output.item() > 0.5 else 0

    def update_parameters(self, output, feat, ex, lr):
        # update the weight of the perceptron given its activation, the input features, the example, and the learning rate
        target = torch.Tensor([[ex.label]])
        self.optim.zero_grad()
        loss = nn.functional.binary_cross_entropy(output, target)
        loss.backward()
        self.optim.step()


class RNNClassifier(FNNClassifier):

    """
    Q5: Implement the RNN classifier.
    """

    def __init__(self, args):
        super().__init__(args)
        # Start of your code

        self.lstm = nn.LSTM(300, 20, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(40, 1)
        self.sigmoid = nn.Sigmoid()

        # End of your code
        self.optim = torch.optim.Adam(self.parameters(), args.learning_rate)

    def forward(self, feat):
        feat = feat.unsqueeze(0)

        lstm_out, _ = self.lstm(feat)
        pool = torch.max(lstm_out, dim=1).values
        fc = self.fc(pool)
        output = self.sigmoid(fc)

        return output


class MyNNClassifier(FNNClassifier):

    """
    Q6: Implement the your own classifier.
    """

    def __init__(self, args):
        super().__init__(args)
        # Start of your code

        self.lstm = nn.LSTM(300, 30, 2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(60, 1)
        self.sigmoid = nn.Sigmoid()

        # End of your code
        self.optim = torch.optim.Adam(self.parameters(), args.learning_rate)

    def forward(self, feat):
        feat = feat.unsqueeze(0)

        lstm_out, _ = self.lstm(feat)
        pool = torch.max(lstm_out, dim=1).values
        fc = self.fc(pool)
        output = self.sigmoid(fc)


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You don't need to change this.
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor()
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor()
    else:
        raise Exception("Pass in UNIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = PerceptronClassifier(feat_extractor)
    elif args.model == "FNN":
        model = FNNClassifier(args)
    elif args.model == 'RNN':
        model = RNNClassifier(args)
    elif args.model == 'MyNN':
        model = MyNNClassifier(args)
    else:
        raise NotImplementedError()

    model.run_train(train_exs, dev_exs, lr=args.learning_rate, epoch=args.epoch)
    return model
