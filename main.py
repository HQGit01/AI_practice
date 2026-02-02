from nt import W_OK
from numpy.polynomial.chebyshev import chebpts2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

with open('./src/datas.txt', 'r') as file:
    words = file.read().lower().splitlines()

chars = sorted(list(set(c for word in words for c in word)))
print(chars)
print(len(chars))

# Create a mapping from characters to integers
stoi = {ch:i+1 for i, ch in enumerate[str](chars)}
itos = {i+1:ch for i, ch in enumerate(chars)}

stoi['.'] = 0
itos[0] = '.'
print(f'{stoi=}')

# # 先生成训练数据
# N = torch.zeros((27,27),dtype=torch.int32)
# data = {}
# for word in words:
#     context = ['.'] + list(word) + ['.']
#     next = context[1:]
#     for ch1, ch2 in zip(context, next):
#         x = stoi[ch1]
#         y = stoi[ch2]
#         N[x, y] += 1
# # 通过imshow显示N数组的值
# plt.imshow(N)
# plt.show()
# # 计算概率
# p = (N).float()
# p = p / p.sum(1, keepdim=True)

# print(p)

g = torch.Generator().manual_seed(2147483647)

# ix = 0
# out = []
# while True:
#     prob = p[ix]
#     ix = torch.multinomial(prob, num_samples=1, replacement=True, generator=g).item()
#     out.append(itos[ix])
#     if ix == 0:
#         break
# print(''.join(out))

## 下面尝试使用神经网络实现上面的功能

# 考虑一层神经元，神经元个数为27，输入为27，输出为27

# 构造一个weight张量
W = torch.randn((28,28), generator=g, requires_grad=True)

xs = []
ys = []

for word in words:
    # 先给定输入数据
    context = ['.'] + list(word) + ['.']
    for ch1, ch2 in zip(context, context[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print(f'{num=}')
    # 对x和y进行独热编码
print(f'{torch.argmax(xs)=}')
xenc = nn.functional.one_hot(xs, num_classes=28).float()
for k in range(1000):
    log_prob = xenc @ W
    counts = log_prob.exp()
    probs =  counts / counts.sum(1, keepdim=True)
    loss = -probs[torch.arange(num, dtype=torch.int32), ys].log().mean()
    W.grad=None
    loss.backward()
    # lr = 30.0 / (1 + k * 0.01) 
    lr = 10
    W.data -= lr * W.grad
print(f'{loss=}')

print(W.shape)
for idx in range(10):
    ix = 0
    out = []
    while True:
        xenc = nn.functional.one_hot(torch.tensor([ix]), num_classes=28).float()
        # 计算 logits
        logits = xenc @ W
        # 转换为概率分布（softmax）
        counts = logits.exp()
        prob = counts / counts.sum(1, keepdim=True)
        ix = torch.multinomial(prob, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))
    
    


