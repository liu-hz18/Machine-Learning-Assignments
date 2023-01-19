import copy
import argparse
import numpy as np
from tqdm import tqdm


def softmax2d(x):
    """
    :param x: input array of shape [D, K], where K is the feature num
    :return: softmax on the last dimension, shape = [D, K]
    """
    x_max = np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x - x_max)
    f_x = e_x / np.sum(e_x, axis=1, keepdims=True)
    return f_x



class Dataset(object):

    def __init__(self, vocab_file: str, dataset_file: str):
        self.vocab_dict = []
        self.t = []
        # load vocab file
        with open(vocab_file, 'r') as fv:
            self.vocab_num = 0
            for line in fv:
                words = line.strip().split()
                self.vocab_dict.append(words[1])
                self.vocab_num += 1
        # load document file
        with open(dataset_file, 'r') as fd:
            self.doc_num = 0
            for line in fd:
                doc = np.zeros(self.vocab_num, dtype=float)
                words = line.strip().split()[1:]
                for word in words:
                    tmp = word.split(":")
                    word_id = int(tmp[0])
                    word_count = int(tmp[1])
                    doc[word_id] = word_count
                self.t.append(doc)
                self.doc_num += 1
        self.t = np.array(self.t)

    @property
    def T(self):
        return self.t

    @property
    def D(self):
        return self.doc_num

    @property
    def W(self):
        return self.vocab_num



class MixtureOfMultinomialEM(object):

    def __init__(self, d: int, w: int, k: int=10):
        self._d, self._w, self._k = d, w, k
        self.pi = np.random.random([k]) + 1e-30
        self.mu = np.random.random([k, w]) + 1e-30
        self.pi = self.pi / np.sum(self.pi)  # shape = [K]
        self.mu = self.mu / np.sum(self.mu, axis=1, keepdims=True) # shape = [K, W]

    def step_e(self, t: np.array):
        """
        :param t: the corpus word occurrence matrix T_{dw}, shape = [D, W]
        :return:  gamma, shape = [D, K]
        """
        logits = np.dot(t, np.log(self.mu.T + 1e-30)) + np.log(self.pi) # shape = [D, K]
        gamma = softmax2d(logits)
        return gamma

    def step_m(self, t: np.array, gamma: np.array):
        """
        :param t: the corpus word occurrence matrix T_{dw}, shape = [D, W]
        :param gamma: the gamma output of E step, shape = [D, K]
        :return: pi, shape = [K]
        :return: mu, shape = [K, W]
        """
        self.pi = np.sum(gamma, axis=0) / np.sum(gamma)
        tmp = np.dot(gamma.T, t) # shape = [K, W]
        self.mu = tmp / np.sum(tmp, axis=1, keepdims=True)
        return self.pi, self.mu


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EM algorithm for mixture of multinomial distribution.")
    parser.add_argument("--vocab", type=str, default="./20news/20news.vocab", help="vocabe file for training EM.")
    parser.add_argument("--corpus", type=str, default="./20news/20news.libsvm", help="corpus file for training EM.")
    parser.add_argument("--seed", type=int, default=23333333, help="random seed for EM algorithm.")
    parser.add_argument("--k", type=int, default=10, help="number of topics.")
    parser.add_argument("--maxiter", type=int, default=50, help="max number limit of iterations")
    parser.add_argument("--topk", type=int, default=5, help="most frequently word to be shown")
    args = parser.parse_args()
    
    # initialization
    np.random.seed(args.seed)
    print("Loading dataset...")
    dataset = Dataset(vocab_file=args.vocab, dataset_file=args.corpus)
    model = MixtureOfMultinomialEM(d=dataset.D, w=dataset.W, k=args.k)
    
    print("(D, W) = ", dataset.T.shape)

    # train
    print(f"Begin Training for K={args.k}")
    pi_old = np.zeros([args.k], dtype=float)
    epoches = 0
    for i in tqdm(range(args.maxiter)):
        gamma = model.step_e(dataset.T)
        pi, mu = model.step_m(dataset.T, gamma)
        norm = np.linalg.norm(pi - pi_old)
        pi_old = copy.deepcopy(pi)
        epoches += 1
        if norm < 1e-3: # convergence condition
            break
    print(f"Train Finished.")
    print(f"Norm value for (\pi - \pi_old) = {norm:.8f} after {epoches} iterations.")

    # evaluation
    topics = np.argsort(model.pi)[::-1]
    for k in range(args.k):
        topic = topics[k]
        print(f"Topic {k+1} (ratio = {pi[topic]:.4f}) :")
        words = np.argsort(model.mu[topic])[::-1]
        for i in range(min(len(words), args.topk)):
            print(f"\t[rank {i+1}]: {dataset.vocab_dict[words[i]]} ({mu[topic][words[i]]:.5f})")
