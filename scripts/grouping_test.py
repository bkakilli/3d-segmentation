import numpy as np
import torch


x = np.array([
    [0,1],
    [0,1],
    [1,2],
    [1,2],
    [1,2],
    [2,3],
    [2,3],
    [2,3],
    [3,4],
], dtype=np.float32)
x = torch.tensor(x, requires_grad=True)

groups = [[0,1], [2,3,4], [5,6,7,8]]
g_t = []
for g in groups:
    g_t.append(torch.mean(3*x[g]**2, dim=0, keepdims=True))

gr = (torch.cat(g_t, dim=0))
gr = gr**2+2

temp = torch.zeros_like(x)
for i, g in enumerate(groups):
    temp[g] = gr[i]

loss = temp.mean()
loss.backward()

print(x.grad)