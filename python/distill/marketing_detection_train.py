import time

import torch
import torch.optim as optim
import torch.nn.functional as F

from models import LSTMRegressionModel
from dataset import *


def train(dataset, model, loss_fn, optimizer, device='cpu'):
    model.train()
    for batch, (X, y) in enumerate(dataset, 1):
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 200 == 0:
            loss, current = loss.item(), (batch) * len(X)
            print(f"| batch: [{batch:>5d}], loss: {loss:>7f}")

def evaluate(dataset, model, loss_fn, device='cpu'):
    model.eval()
    test_loss, correct = 0., 0.
    num_batches, size = 0, 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataset, 1):
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += ((pred > 0.5).type(torch.float)== y).type(torch.float).sum().item()

            num_batches, size = batch, size + len(X) 
        test_loss /= num_batches
        correct /= size
    print(f"\n Validation Accuracy: {(100*correct):>0.1f}%, Validation loss: {test_loss:>8f} \n")

def train_and_save(train_data_path, valid_data_path):
    start_time = time.time()
    vocab = build_vocab([train_data_path, valid_data_path], charTokenize)
    esplased_time = time.time() - start_time
    print(f'Build vocab... esplased: {time.time() - start_time:.2f}s')
    print(f'vocab size is {len(vocab)}')

    train_pipe = build_data_pipe(train_data_path, vocab=vocab)
    print('Load train set...')
    # for texts, labels in train_pipe:
    #    print(texts)
    #    print(type(texts), texts.shape)
    #    print(type(labels), labels.shape)
    #    break

    valid_pipe = build_data_pipe(valid_data_path, vocab=vocab)
    print('Load valid set...')
    # for texts, labels in valid_pipe:
    #    print(texts)
    #    print(type(texts), texts.shape)
    #    print(type(labels), labels.shape)
    #    break

    epochs = 3
    learning_rate = 1e-3
    embedding_dim = 100
    hidden_dim = 128
    vocab_size = len(vocab)
    model = LSTMRegressionModel(embedding_dim, hidden_dim, vocab_size)
    loss_fn = F.binary_cross_entropy
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print('Train model...')
    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f"Epoch: {epoch}")
        train(train_pipe, model, loss_fn, optimizer)
        evaluate(valid_pipe, model, loss_fn)
        print("-" * 59)
        print(f"| end of epoch {epoch:3d} | time: {time.time()-epoch_start_time:5.2f}s")
        print("-" * 59)


if __name__ == '__main__':
    train_and_save(train_data_path='../data/train.txt', valid_data_path='../data/valid.txt')