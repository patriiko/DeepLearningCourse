import pandas as pd
import torch
import matplotlib.pyplot as plt

df = pd.read_csv("test.csv")

x_np = df["x"].values
y_np = df["y"].values

x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(1)  # (n, 1)
y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)

model = torch.nn.Linear(1, 1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for t in range(5001):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)

    if t % 1000 == 0:
        print(f"Epoch {t}: Loss = {loss.item():.4f}")

        y_pred_np = y_pred.detach().numpy()
        plt.scatter(x_np, y_np, label="Stvarni podaci", alpha=0.6)
        plt.plot(x_np, y_pred_np, color='red', label=f"Predikcija (epoch {t})")
        plt.title(f"Regresija – Epoch {t}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid()
        plt.show()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


a = model.bias.item()
b = model.weight.item()
print(f"\nNaučeni model: y = {a:.4f} + {b:.4f}x")
