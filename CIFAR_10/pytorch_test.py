import torch

m = torch.nn.Linear(20,30)
input = torch.randn(128,20)
print(input.size)
output = m(input)
print(output.size)