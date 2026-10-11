from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import numpy as np

import librosa

np.random.seed(0)

def load_wavs(data_dir, sr=16000):
    train_data = {}
    test_data = {}

    for path in sorted(Path(data_dir).glob("*.wav")):
        # filename: label_speaker_index.wav
        digit, speaker, index = path.stem.split("_")

        y, _ = librosa.load(path, sr=sr, mono=True)

        if int(index) < 5:
            test_data[path.stem] = y
        else:
            train_data[path.stem] = y

    return train_data, test_data

train_data, test_data = load_wavs("recordings")

def to_mfcc(y, sr=16000, n_mfcc=13):
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T            # (T, 13)

def to_melspec(y, sr=16000, n_mels=40):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    return librosa.power_to_db(S).T                                      # (T, 40)

#안 쓰는 쪽을 주석처리
#FEAT = to_mfcc
FEAT = to_melspec


train_data = {k: FEAT(v) for k, v in train_data.items()}
test_data  = {k: FEAT(v) for k, v in test_data.items()}

class GMM:
    def __init__(self, dim, ncomp, mu, std):
        self.dim = dim
        self.ncomp = ncomp
        self.mu = np.zeros([ncomp, dim])
        self.std = np.zeros([ncomp, dim])

        # initialize each parameters for this GMM
        self.mu = self.mu + mu + 0.01*(std**2)*np.random.randn(ncomp, 1)
        self.std = self.std + std
        self.phi = np.ones([ncomp]) / ncomp

    def train(self, data):
        z = self._e_step(data)
        self._m_step(data, z)

    def _compute_ll(self, data, mu, std):  # compute likelihood
        ll = -self.elog(std) - 0.5 * np.log(2*np.pi) - (data - mu)**2 / (2*std**2)
        return ll.sum()

    def elog(self, x):
        res = np.log(x, where=(x!=0))
        res[np.where(x==0)] = -(10.0**8)
        return res

    def loglike(self, data):
        ll = 0
        for t in range(data.shape[0]):
            ll_t = np.array(
                [
                    np.log(self.phi[k]) + \
                        self._compute_ll(data[t], self.mu[k], self.std[k])
                    for k in range(self.ncomp)
                ]
            )
            ll_t = self._logSumExp(ll_t)
            ll += ll_t
        return ll

    def _logSumExp(self, x, axis=None, keepdims=False):
        x_max = np.max(x, axis=axis, keepdims=keepdims)
        x_diff = x - x_max
        sumexp = np.exp(x_diff).sum(axis=axis, keepdims=keepdims)
        return (x_max + np.log(sumexp))

    def _e_step(self, data):  # data dimension -> N x self.dim
        est_z = np.zeros([len(data), self.ncomp])

        for i in range(len(data)):
            log_z = np.log(self.phi)
            for k in range(self.ncomp):
                log_z[k] += self._compute_ll(data[i], self.mu[k], self.std[k])
            likelihood = np.exp(log_z - log_z.max())
            est_z[i] = likelihood / (np.sum(likelihood))
        return est_z

    def _m_step(self, data, z):
        est_phi = np.mean(z, 0)
        self.est_phi = est_phi / sum(est_phi)
        denom = np.sum(z, 0, keepdims=True).T + 1e-6
        mu_num = np.zeros_like(self.mu)

        for k in range(self.ncomp):
            mu_num[k] = np.sum(z[:, k:k+1] * data, 0)
        self.mu = mu_num / denom

        std_num = np.zeros_like(self.std)
        for k in range(self.ncomp):
            std_num[k] = np.sum(z[:, k:k+1] * (data - self.mu[k])**2, 0)
        self.std = (std_num / denom)**0.5
        self.phi = self.est_phi 

# Preprocessing
X_train = defaultdict(list)
for it, (key, value) in enumerate(train_data.items()):
    label = key.split('_')[0] 
    X_train[label].append(value)

ncomp = 8 # the number of Gaussian components
GMMs = dict()

# training
for label, feats in X_train.items():
    data = np.vstack(feats)
    init_mu = np.mean(data, axis=0, keepdims=True)
    init_std = np.std(data, axis=0, keepdims=True)
    gmm = GMM(data.shape[-1], ncomp, init_mu, init_std)
    GMMs[label] = gmm

pbar = tqdm(range(10)) # training iterations
for i in pbar:
    total_ll = 0
    for label, feats in X_train.items():
        data = np.vstack(feats)
        GMMs[label].train(data)
        total_ll += np.mean(GMMs[label].loglike(data))
    pbar.set_description("Log Likelihood ({:.2f})".format(total_ll / 11))


# test
# Preprocessing
acc = 0
total = 0
pbar = tqdm(test_data.items())
for key, value in pbar:
    label = key.split('_')[0]
    likelihoods = []
    for i, key in enumerate(GMMs):
        if key == label:
            target_id = i
        likelihoods.append(GMMs[key].loglike(value))
    pred = np.argmax(likelihoods, 0)
    total += 1
    acc += (pred == target_id).sum()
    pbar.set_description("ACC : {:.2f}".format(acc / total * 100))

