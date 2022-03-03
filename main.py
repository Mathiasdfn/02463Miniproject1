# Start writing code here...
import numpy as np
import matplotlib.pyplot as plt
import GPyOpt
from torchvision import datasets, transforms, utils
from sklearn import model_selection
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
from scipy import stats

randomstate = 42069
# Set random seed
np.random.seed(randomstate)
random.seed(randomstate)
torch.manual_seed(randomstate)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')


def mcnemar(n11, n12, n21, n22):
    n = n11 + n12 + n21 + n22

    etheta = (n12 - n21) / n
    accA = (n11 + n12) / n
    accB = (n11 + n21) / n

    Q = (((n * n) * (n + 1) * (etheta + 1) * (1 - etheta)) /
         (
                 (n * (n12 + n21)) -
                 ((n12 - n21) * (n12 - n21)))
         )

    f = ((etheta + 1) / 2) * (Q - 1)
    g = ((1 - etheta) / 2) * (Q - 1)
    alpha = 0.05

    theta_lower = (2 * (stats.beta.ppf(alpha / 2, f, g))) - 1
    theta_upper = (2 * (stats.beta.ppf(1 - (alpha / 2), f, g))) - 1

    p = 2 * stats.binom.cdf((min(n12, n21)), (n12 + n21), 0.5)

    print("McNemar for difference in accuracy")
    print(f'Classifier A (BO) accuracy = {accA}, Classifier B (Random) accuracy = {accB}')
    print(f'etheta = {etheta}, Q = {Q}, f = {f}, g = {g}')
    print(f'CI = [{theta_lower};{theta_upper}]')
    print(f'p = {p}')



class Dataset(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, Xs, ys):
        """Initialization"""
        self.Xs = Xs
        self.ys = ys

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.Xs)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Load data and get label
        Xs = self.Xs[index]
        ys = self.ys[index]

        return Xs, ys


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, learning_rate):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, int(hidden_size / 2))
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(int(hidden_size / 2), num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu1(out)
        out = self.l2(out)
        out = self.relu2(out)
        out = self.l3(out)
        return out


def load_MNIST():
    """
    Function to load the MNIST training and test set with corresponding labels.

    :return: training_examples, training_labels, test_examples, test_labels
    """

    # we want to flat the examples

    training_set = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=None)

    Xtrain = training_set.data.numpy().reshape(-1, 28 * 28)
    Xtest = test_set.data.numpy().reshape(-1, 28 * 28)

    ytrain = training_set.targets.numpy()
    ytest = test_set.targets.numpy()

    return Xtrain, ytrain, Xtest, ytest


## Load MNIST data
Xtrain, ytrain, Xtest, ytest = load_MNIST()
X = np.vstack((Xtrain, Xtest))
y = np.hstack((ytrain, ytest))

print('Information about the new datasets')
print('Training set shape:', Xtrain.shape)
print('Test set shape', Xtest.shape)
print('All set shape', X.shape, y.shape)

input_size = Xtrain.shape[1]
num_classes = 10
n_epochs = 10
max_iter = 25

Xtrain = torch.Tensor(Xtrain)
ytrain = torch.Tensor(ytrain)
Xtest = torch.Tensor(Xtest)
ytest = torch.Tensor(ytest)

## Hyperparameter tuning using BO
## define the domain of the considered parameters
hidden_size = tuple(np.arange(2, 100, 2, dtype=np.int))
# learning_rate = tuple(np.arange(1e-5, 1e-3, 1e-5, dtype=np.float))        # Handled by GPyOpt explicitly

# define the dictionary for GPyOpt
domain = [{'name': 'hidden_size', 'type': 'discrete', 'domain': hidden_size},
          {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-5, 1e-2), 'dimensionality': 1}]

## we have to define the function we want to maximize --> validation accuracy,
## note it should take a 2D ndarray but it is ok that it assumes only one point
## in this setting

acc_current = 0
modelBOBest = 0

def objective_function(x):
    # print(x)
    param = x[0]

    # create the model
    model = NeuralNet(hidden_size=int(param[0]), learning_rate=float(param[1]), input_size=input_size,
                      num_classes=num_classes).to(device)

    ## fit the model
    xin = Xtrain.float().to(device)
    yin = torch.squeeze(ytrain.long()).to(device)

    for epoch in range(n_epochs):
        # forwards
        outputs = model(xin)
        loss = model.criterion(outputs, yin.squeeze())

        # backwards
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

    ## evaluate the model
    with torch.no_grad():
        xin = Xtest.float().to(device)
        yin = torch.squeeze(ytest.long()).to(device)
        outputs = model(xin)
        _, predictions = torch.max(outputs, 1)
        n_samples = yin.shape[0]
        n_correct = (predictions == yin).sum().item()
        acc = n_correct / n_samples
        global acc_current
        # global modelBOBest
        if acc > acc_current:
            acc_current = acc
            # modelBOBest = model
    global modelslist
    modelslist = np.append(modelslist, model)
    BOacciterlist_temp.append(acc_current)

    # print(acc)
    return - acc


#ewlist = np.arange(1e-2, 11e-2, 1e-2)
ewlist = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
accs = np.zeros(len(ewlist))
# modelBOBest = np.array([NeuralNet]*len(ewlist))
bestmodelsBO = np.array([])
x_best = np.zeros((len(ewlist),2))
BOacciterlist = []

for i, ew in enumerate(ewlist):
    # Set random seed
    np.random.seed(randomstate)
    random.seed(randomstate)
    torch.manual_seed(randomstate)

    modelslist = np.array([])
    BOacciterlist_temp = []
    acc_current = 0
    opt = GPyOpt.methods.BayesianOptimization(f=objective_function,  # function to optimize
                                              domain=domain,  # box-constrains of the problem
                                              acquisition_type='EI',  # Select acquisition function MPI, EI, LCB
                                              initial_design_numdata=5  # Set initial points sampled
                                              )
    ## Exploration weight is called jitter for EI
    opt.acquisition.jitter = ew

    # Set random seed
    np.random.seed(randomstate)
    random.seed(randomstate)
    torch.manual_seed(randomstate)
    opt.run_optimization(max_iter=max_iter)

    l_x_best = opt.X[np.argmin(opt.Y)]
    bestmodelsBO = np.append(bestmodelsBO, modelslist[np.argmin(opt.Y)])
    x_best[i] = l_x_best
    accs[i] = acc_current
    print("EW: " + str(ew) + ". The best parameters obtained: hidden_size=" + str(x_best[i][0]) + ", learning_rate=" + str(
        x_best[i][1]), "acc:", str(acc_current))
    # modelBOBest[i] = NeuralNet(hidden_size=int(x_best[i][0]), learning_rate=float(x_best[i][1]), input_size=input_size,
    #                            num_classes=num_classes).to(device)
    BOacciterlist.append(BOacciterlist_temp)

## Define network with Bayesian-optimized hyperparameters (override as we're training from scratch)

# modelBOBest = NeuralNet(hidden_size=int(x_best[0]), learning_rate=float(x_best[1]),
#                         input_size=input_size, num_classes=num_classes).to(device)

## Random sampling hyperparameters
### Make sure we sample for max_iter+initial sample points and use same domains as for BO

acc_current = 0
modelRandomBest = 0
# Set random seed
np.random.seed(randomstate)
random.seed(randomstate)
torch.manual_seed(randomstate)
randomacciterlist = []

for i in range(max_iter + 5):       # as we have max_iter=max_iter and initial_design_numdata=5 for BO
    param = np.zeros(2)
    param[0] = np.random.choice(hidden_size)
    param[1] = (1e-2 - 1e-5) * np.random.random_sample() + 1e-5     # Random uniform sampling from 1e-5 to 1e-2

    model = NeuralNet(hidden_size=int(param[0]), learning_rate=float(param[1]), input_size=input_size,
                      num_classes=num_classes).to(device)

    ## fit the model
    xin = Xtrain.float().to(device)
    yin = torch.squeeze(ytrain.long()).to(device)

    for epoch in range(n_epochs):
        # forwards
        outputs = model(xin)
        loss = model.criterion(outputs, yin.squeeze())

        # backwards
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

    ## evaluate the model
    with torch.no_grad():
        xin = Xtest.float().to(device)
        yin = torch.squeeze(ytest.long()).to(device)
        outputs = model(xin)
        _, predictions = torch.max(outputs, 1)
        n_samples = yin.shape[0]
        n_correct = (predictions == yin).sum().item()
        acc = n_correct / n_samples
        print(acc)

    if acc > acc_current:
        acc_current = acc
        x_best_random = param
        modelRandomBest = model
        print(x_best_random)
    randomacciterlist.append(acc_current)

print("Best parameters:", x_best_random, "Best accuracy:", acc_current)

## Define network with randomly sampled hyperparameters (override as we're training from scratch)
# modelRandomBest = NeuralNet(hidden_size=int(x_best_random[0]), learning_rate=float(x_best_random[1]),
#                             input_size=input_size, num_classes=num_classes).to(device)

predictionslist = np.zeros((len(ewlist),Xtest.shape[0]))
xin = Xtest.float().to(device)
yin = torch.squeeze(ytest.long()).to(device)

for i, model in enumerate(bestmodelsBO):
    # testing BO
    with torch.no_grad():
        outputs = model(xin)
        _, predictions = torch.max(outputs, 1)
        predictionslist[i] = predictions
        # n_samples = yin.shape[0]
        # n_correct = (predictions == yin).sum().item()

# Set random seed
np.random.seed(randomstate)
random.seed(randomstate)
torch.manual_seed(randomstate)
# testing Random
with torch.no_grad():
    outputs = modelRandomBest(xin)
    _, predictions = torch.max(outputs, 1)
    predictionsRandom = predictions
    n_samples = yin.shape[0]
    n_correct = (predictions == yin).sum().item()
    acc = n_correct / n_samples
    print(acc)

for l, predictions in enumerate(predictionslist):
    n11 = len([i for i, j, k in zip(predictions, predictionsRandom, yin) if i == j == k])
    n12 = len([i for i, j, k in zip(predictions, predictionsRandom, yin) if (i == k and j != k)])
    n21 = len([i for i, j, k in zip(predictions, predictionsRandom, yin) if (i != k and j == k)])
    n22 = len([i for i, j, k in zip(predictions, predictionsRandom, yin) if (i != k and j != k)])
    print("McNemar for BO Exploration weight:", ewlist[l], "VS Random")
    print(n11, "\t", n12, "\n",
          n21, "\t", n22)
    mcnemar(n11, n12, n21, n22)


t = np.arange(1, max_iter+5+1, 1)

plt.figure(dpi=300)
for i, ew in enumerate(ewlist):
    plt.plot(t, BOacciterlist[i], '--', label=("EI, xi=" + str(np.round(ew, 2))))

plt.plot(t, randomacciterlist, '-k', lw=3, alpha=0.5, label="Random sampling")
plt.xlabel("Iteration")
plt.ylabel("Test accuracy")
plt.legend()
plt.show()

print(BOacciterlist)




# for i, model in enumerate(modelBOBest):
#     # Set random seed
#     np.random.seed(randomstate)
#     random.seed(randomstate)
#     torch.manual_seed(randomstate)
#
#     for epoch in range(n_epochs):
#         # forwards
#         outputs = model(xin)
#         loss = model.criterion(outputs, yin.squeeze())
#
#         # backwards
#         model.optimizer.zero_grad()
#         loss.backward()
#         model.optimizer.step()
#     # testing BO
#     with torch.no_grad():
#         outputs = model(xin)
#         _, predictions = torch.max(outputs, 1)
#         predictionslist[i] = predictions
#         # n_samples = yin.shape[0]
#         # n_correct = (predictions == yin).sum().item()
#
# # Set random seed
# np.random.seed(randomstate)
# random.seed(randomstate)
# torch.manual_seed(randomstate)
# for epoch in range(n_epochs):
#     # forwards
#     outputs = modelRandomBest(xin)
#     loss = modelRandomBest.criterion(outputs, yin.squeeze())
#
#     # backwards
#     modelRandomBest.optimizer.zero_grad()
#     loss.backward()
#     modelRandomBest.optimizer.step()
# # testing Random
# with torch.no_grad():
#     outputs = modelRandomBest(xin)
#     _, predictions = torch.max(outputs, 1)
#     predictionsRandom = predictions
#     n_samples = yin.shape[0]
#     n_correct = (predictions == yin).sum().item()
#     acc = n_correct / n_samples
#     print(acc)


## Make graph for BO vs random sampling
#
#
# def objective_function2(x):
#     # print(x)
#     param = x[0]
#
#     # create the model
#     model = NeuralNet(hidden_size=int(param[0]), learning_rate=float(param[1]), input_size=input_size,
#                       num_classes=num_classes).to(device)
#
#     ## fit the model
#     xin = Xtrain.float().to(device)
#     yin = torch.squeeze(ytrain.long()).to(device)
#
#     for epoch in range(n_epochs):
#         # forwards
#         outputs = model(xin)
#         loss = model.criterion(outputs, yin.squeeze())
#
#         # backwards
#         model.optimizer.zero_grad()
#         loss.backward()
#         model.optimizer.step()
#
#     ## evaluate the model
#     with torch.no_grad():
#         xin = Xtest.float().to(device)
#         yin = torch.squeeze(ytest.long()).to(device)
#         outputs = model(xin)
#         _, predictions = torch.max(outputs, 1)
#         n_samples = yin.shape[0]
#         n_correct = (predictions == yin).sum().item()
#         acc = n_correct / n_samples
#         global acc_current
#         # global modelBOBest
#         if acc > acc_current:
#             acc_current = acc
#             # modelBOBest = model
#         BOacciterlist.append(acc_current)
#
#     # print(acc)
#     return - acc
#
#
# def objective_function3(x):
#     # print(x)
#     param = x[0]
#
#     # create the model
#     model = NeuralNet(hidden_size=int(param[0]), learning_rate=float(param[1]), input_size=input_size,
#                       num_classes=num_classes).to(device)
#
#     ## fit the model
#     xin = Xtrain.float().to(device)
#     yin = torch.squeeze(ytrain.long()).to(device)
#
#     for epoch in range(n_epochs):
#         # forwards
#         outputs = model(xin)
#         loss = model.criterion(outputs, yin.squeeze())
#
#         # backwards
#         model.optimizer.zero_grad()
#         loss.backward()
#         model.optimizer.step()
#
#     ## evaluate the model
#     with torch.no_grad():
#         xin = Xtest.float().to(device)
#         yin = torch.squeeze(ytest.long()).to(device)
#         outputs = model(xin)
#         _, predictions = torch.max(outputs, 1)
#         n_samples = yin.shape[0]
#         n_correct = (predictions == yin).sum().item()
#         acc = n_correct / n_samples
#         global acc_current
#         # global modelBOBest
#         if acc > acc_current:
#             acc_current = acc
#             # modelBOBest = model
#         BOacciterlist2.append(acc_current)
#
#     # print(acc)
#     return - acc


# randomstate = 420
# max_iter=25
# ## EW = 0.01
# acc_current = 0
# BOacciterlist = []
# opt = GPyOpt.methods.BayesianOptimization(f=objective_function2,  # function to optimize
#                                               domain=domain,  # box-constrains of the problem
#                                               acquisition_type='EI',  # Select acquisition function MPI, EI, LCB
#                                               initial_design_numdata=5  # Set initial points sampled
#                                               )
# ## Exploration weight is called jitter for EI
# opt.acquisition.jitter = 1e-2
# # Set random seed
# np.random.seed(randomstate)
# random.seed(randomstate)
# torch.manual_seed(randomstate)
# opt.run_optimization(max_iter=max_iter)
# print(BOacciterlist)
#
# ## EW = 0.02
# acc_current = 0
# BOacciterlist2 = []
# opt = GPyOpt.methods.BayesianOptimization(f=objective_function3,  # function to optimize
#                                               domain=domain,  # box-constrains of the problem
#                                               acquisition_type='EI',  # Select acquisition function MPI, EI, LCB
#                                               initial_design_numdata=5  # Set initial points sampled
#                                               )
# ## Exploration weight is called jitter for EI
# opt.acquisition.jitter = 10e-2
# # Set random seed
# np.random.seed(randomstate)
# random.seed(randomstate)
# torch.manual_seed(randomstate)
# opt.run_optimization(max_iter=max_iter)
# print(BOacciterlist2)
#
# ## Random sampling hyperparameters
# acc_current = 0
# modelRandomBest = 0
# randomacciterlist = []
# # Set random seed
# np.random.seed(randomstate)
# random.seed(randomstate)
# torch.manual_seed(randomstate)
#
# for i in range(max_iter + 5):       # as we have max_iter=max_iter and initial_design_numdata=5 for BO
#     param = np.zeros(2)
#     param[0] = np.random.choice(hidden_size)
#     # param[1] = (1e-2 - 1e-5) * np.random.random_sample() + 1e-5     # Random uniform sampling from 1e-5 to 1e-2
#     param[1] = np.random.uniform(low=1e-5, high=1e-2)
#
#     model = NeuralNet(hidden_size=int(param[0]), learning_rate=float(param[1]), input_size=input_size,
#                       num_classes=num_classes).to(device)
#
#     ## fit the model
#     xin = Xtrain.float().to(device)
#     yin = torch.squeeze(ytrain.long()).to(device)
#
#     for epoch in range(n_epochs):
#         # forwards
#         outputs = model(xin)
#         loss = model.criterion(outputs, yin.squeeze())
#
#         # backwards
#         model.optimizer.zero_grad()
#         loss.backward()
#         model.optimizer.step()
#
#     ## evaluate the model
#     with torch.no_grad():
#         xin = Xtest.float().to(device)
#         yin = torch.squeeze(ytest.long()).to(device)
#         outputs = model(xin)
#         _, predictions = torch.max(outputs, 1)
#         n_samples = yin.shape[0]
#         n_correct = (predictions == yin).sum().item()
#         acc = n_correct / n_samples
#
#     if acc > acc_current:
#         acc_current = acc
#         x_best_random = param
#         modelRandomBest = model
#     randomacciterlist.append(acc_current)
#
# print(randomacciterlist)
#
# t = np.arange(1, max_iter+5+1, 1)
#
# plt.plot(t, BOacciterlist, '-', label="EI, xi=0.01")
# plt.plot(t, BOacciterlist2, '-', label="EI, xi=0.1")
# plt.plot(t, randomacciterlist, '-', label="Random sampling")
# plt.xlabel("Iteration")
# plt.ylabel("Test accuracy")
# plt.legend()
#
# plt.show()

#
# ## Test the two networks against each others
# K = 10
# CV = model_selection.KFold(K, shuffle=True, random_state=69420)
#
# n11 = 0
# n12 = 0
# n21 = 0
# n22 = 0
#
# predictions = np.zeros((7,1))
#
# for k, (train_index, test_index) in enumerate(CV.split(X, y)):
#     X_train = torch.Tensor(X[train_index, :])
#     y_train = torch.Tensor(y[train_index])
#     X_test = torch.Tensor(X[test_index, :])
#     y_test = torch.Tensor(y[test_index])
#
#     batch_size = X.shape[0]
#
#     train_loader = torch.utils.data.DataLoader(Dataset(X_train, y_train), batch_size=batch_size,
#                                                shuffle=True)
#     test_loader = torch.utils.data.DataLoader(Dataset(X_test, y_test), batch_size=batch_size,
#                                               shuffle=False)
#     for i, ew in enumerate(x_best):
#         modelBOBest[i] = NeuralNet(hidden_size=int(x_best[i][0]), learning_rate=float(x_best[i][1]),
#                                 input_size=input_size, num_classes=num_classes).to(device)
#     modelRandomBest = NeuralNet(hidden_size=int(x_best_random[0]), learning_rate=float(x_best_random[1]),
#                                 input_size=input_size, num_classes=num_classes).to(device)
#
#     for epoch in range(n_epochs):
#         for i, (xin, yin) in enumerate(train_loader):
#             xin = xin.float().to(device)
#             yin = torch.squeeze(yin.long()).to(device)
#
#             # Random
#             # forward
#             outputsRandom = modelRandomBest(xin)
#             lossRandom = modelRandomBest.criterion(outputsRandom, yin.squeeze())
#             # backwards
#             lossRandom.backward()
#             modelRandomBest.optimizer.zero_grad()
#             modelRandomBest.optimizer.step()
#
#             # BO
#             outputsBO = np.zeros(11)
#             lossBO = np.zeros(11)
#
#             for i, ew in enumerate(x_best):
#                 outputsBO[i] = modelBOBest[i](xin)
#                 lossBO[i] = modelBOBest[i].criterion(outputsBO[i], yin.squeeze())
#                 modelBOBest[i].optimizer.zero_grad()
#                 lossBO[i].backward()
#                 modelBOBest[i].optimizer.step()
#
#     ## Testing
#     with torch.no_grad():
#         for xin, yin in test_loader:
#             xin = xin.float().to(device)
#             yin = torch.squeeze(yin.long()).to(device)
#             outputsBO = np.zeros(6)
#             for i, ew in enumerate(x_best):
#                 outputsBO[i] = modelBOBest[i](xin)
#             outputsRandom = modelRandomBest(xin)
#             predictionsBO = np.zeros(6)
#             for i, ew in enumerate(x_best):
#                 _, predictionsBO[i] = torch.max(outputsBO[i], 1)
#                 predictions = np.append(predictions[i], predictionsBO[i])
#             _, predictionsRandom = torch.max(outputsRandom, 1)
#             # predictions = np.append(predictions[-1], predictionsRandom)
#             # n11 += len([i for i, j, k in zip(predictionsBO, predictionsRandom, yin) if i == j == k])
#             # n12 += len([i for i, j, k in zip(predictionsBO, predictionsRandom, yin) if (i == k and j != k)])
#             # n21 += len([i for i, j, k in zip(predictionsBO, predictionsRandom, yin) if (i != k and j == k)])
#             # n22 += len([i for i, j, k in zip(predictionsBO, predictionsRandom, yin) if (i != k and j != k)])
#             print(predictions)
# #     print("k:", str(k))
# #     print(n11, "\t", n12, "\n",
# #           n21, "\t", n22)
# #
# # print("Final:")
# # print(n11, "\t", n12, "\n",
# #       n21, "\t", n22)
#
# ## McNemar
# n = n11 + n12 + n21 + n22
#
# etheta = (n12 - n21) / n
# accA = (n11 + n12) / n
# accB = (n11 + n21) / n
#
# Q = (((n * n) * (n + 1) * (etheta + 1) * (1 - etheta)) /
#      (
#              (n * (n12 + n21)) -
#              ((n12 - n21) * (n12 - n21)))
#      )
#
# f = ((etheta + 1) / 2) * (Q - 1)
# g = ((1 - etheta) / 2) * (Q - 1)
# alpha = 0.05
#
# theta_lower = (2 * (stats.beta.ppf(alpha / 2, f, g))) - 1
# theta_upper = (2 * (stats.beta.ppf(1 - (alpha / 2), f, g))) - 1
#
# p = 2 * stats.binom.cdf((min(n12, n21)), (n12 + n21), 0.5)
#
# print("McNemar for difference in accuracy")
# print(f'Classifier A (BO) accuracy = {accA}, Classifier B (Random) accuracy = {accB}')
# print(f'etheta = {etheta}, Q = {Q}, f = {f}, g = {g}')
# print(f'CI = [{theta_lower};{theta_upper}]')
# print(f'p = {p}')
