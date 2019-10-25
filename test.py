import os
import torch
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
from RNN import RNN
#from CNN import CNN
from alexnet import alexnet
from vocab_build import vocab_build
from torch.autograd import Variable
from torchvision import transforms
from dataloader import dataloader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-img', type=str, default='3.jpg')
    parser.add_argument('-train_time', type=int, default=3)
    parser.add_argument('-epoch', type=int, default=4)
    args = parser.parse_args()
    img= args.img
    train_time=args.train_time
    epoch=args.epoch

    dir = 'train'
    embedding_size = 512
    hidden_size = 512
    gpu_device = None

    with open(os.path.join(dir, 'vocab.pkl'), 'rb') as file:
        vocab = pickle.load(file)
        print('vocab loaded')

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])
    img = transform(Image.open(img))
    image = img.unsqueeze(0)
    #image = Variable(image).cuda()
    image = Variable(image)

    cnn = alexnet(embedding_dim=embedding_size)
    rnn = RNN(embedding_dim=embedding_size, hidden_dim=hidden_size, vocab_size=vocab.index)
    #cnn.cuda()
    #rnn.cuda()

    #cnn_file = str(train_time) + '_iter_' + str(epoch) + '_cnn.pkl'
    #rnn_file = str(train_time) + '_iter_' + str(epoch) + '_rnn.pkl'
    cnn_file = 'alex_iter_' + str(epoch) + '_cnn.pkl'
    rnn_file = 'alex_iter_' + str(epoch) + '_rnn.pkl'
    cnn.load_state_dict(torch.load(os.path.join('train_file', cnn_file), map_location='cpu'))
    rnn.load_state_dict(torch.load(os.path.join('train_file', rnn_file), map_location='cpu'))

    cnn_out = cnn(image)
    word_id = rnn.search(cnn_out)
    sentence= vocab.get_sentence(word_id)
    print(sentence)

    showimage = Image.open(args.img)
    plt.imshow(np.asarray(showimage))
