import torch
import math
import matplotlib.pyplot as plt

print(torch.cuda.is_available())

dtype = torch.float
device = torch.device("cuda")

x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = x**2 + 2 * x + 1


plt.plot(x.cpu(), y.cpu(), '-r', label='y = x² + 2x + 1')
plt.title('Originalna funkcija: y = x² + 2x + 1')
plt.xlabel('x', color='#1C2833')
plt.ylabel('y', color='#1C2833')
plt.legend(loc='upper left')
plt.grid()
plt.show()


p = torch.tensor([1, 2, 3], device=device)
xx = x.unsqueeze(-1).pow(p)


model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
).cuda()

loss_fn = torch.nn.MSELoss(reduction='sum').cuda()

learning_rate = 1e-6

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
   
for t in range(2001):
    y_pred = model(xx)

    loss = loss_fn(y_pred, y)
    if t % 100 == 0:
        print(t, loss.item())

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    linear_layer = model[0]

    a = linear_layer.bias.item()
    b = linear_layer.weight[:, 0].item()
    c = linear_layer.weight[:, 1].item()
    d = linear_layer.weight[:, 2].item()

    if t % 500 == 0:
        y_graph = a + b * x + c * x ** 2 + d * x ** 3
        plt.plot(x.cpu().detach(), y_graph.cpu().detach(), '-r', label='t = ' + str(t) + ' y=a + b * x + c * x ** 2 + d * x ** 3')
        plt.title('Graph of y = a + bx + cx^2 + dx^3')
        plt.xlabel('x', color='#1C2833')
        plt.ylabel('y', color='#1C2833')
        plt.legend(loc='upper left')
        plt.grid()
        plt.show()

print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')