# code by Tae Hwan Jung @graykode
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

dtype = torch.FloatTensor

sentences = [ "i like dog", "i love coffee", "i hate milk"]

word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}
number_dict = {i: w for i, w in enumerate(word_list)}
n_class = len(word_dict) # number of Vocabulary

# NNLM Parameter
n_step = 2 # n-1 in paper
n_hidden = 2 # h in paper
m = 2 # m in paper m:表示词向量C(w)的维度，一般是50到100

def make_batch(sentences):
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()
        input = [word_dict[n] for n in word[:-1]]
        target = word_dict[word[-1]]

        input_batch.append(input)
        target_batch.append(target)

    return input_batch, target_batch

# Model
class NNLM(nn.Module):
    def __init__(self):
        super(NNLM, self).__init__()
        self.C = nn.Embedding(n_class, m)  #C：词向量C(w)存在于矩阵C(|V|*m)中，矩阵C的行数表示词汇表的大小；列数表示词向量C(w)的维度。矩阵C的某一行对应一个单词的词向量表示。
        self.H = nn.Parameter(torch.randn(n_step * m, n_hidden).type(dtype)) #H: 隐藏层的权重(h*(n-1)m)
        self.W = nn.Parameter(torch.randn(n_step * m, n_class).type(dtype)) #W:输入层到输出层权重(|V|*(n-1)m)
        self.d = nn.Parameter(torch.randn(n_hidden).type(dtype)) #d:隐藏层偏置bias(h)
        self.U = nn.Parameter(torch.randn(n_hidden, n_class).type(dtype)) #U:隐藏层到输出层的权重(|V|*h)
        self.b = nn.Parameter(torch.randn(n_class).type(dtype)) #b:输出层的偏置bias(|V|)

    def forward(self, X):
        X = self.C(X)
        X = X.view(-1, n_step * m) # [batch_size, n_step * n_class]
        tanh = torch.tanh(self.d + torch.mm(X, self.H)) # [batch_size, n_hidden]
        output = self.b + torch.mm(X, self.W) + torch.mm(tanh, self.U) # [batch_size, n_class]
        return output

model = NNLM()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

input_batch, target_batch = make_batch(sentences)
input_batch = Variable(torch.LongTensor(input_batch))
target_batch = Variable(torch.LongTensor(target_batch))

# Training
for epoch in range(5000):

    optimizer.zero_grad()
    output = model(input_batch)

    # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
    loss = criterion(output, target_batch)
    if (epoch + 1)%1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    loss.backward()
    optimizer.step()

# Predict
predict = model(input_batch).data.max(1, keepdim=True)[1]

# Test
print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])
