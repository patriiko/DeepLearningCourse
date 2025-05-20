import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("sales_data_sample.csv", encoding="unicode_escape")

data.fillna(data.mean(numeric_only=True), inplace=True)

data.drop_duplicates(inplace=True)

if 'PRICEEACH' in data.columns:
    usd_to_eur = 0.85
    data['PRICEEACH'] = data["PRICEEACH"] * 0.85

print(data.head())

if 'PRICEEACH' in data.columns:
    plt.figure(figsize=(8, 5))
    data.boxplot(column='PRICEEACH')
    plt.title("Boxplot (Euri)")
    plt.ylabel("Cijena")
    plt.show()


if 'QUANTITYORDERED' in data.columns:
    plt.figure(figsize=(8, 5))
    data['QUANTITYORDERED'].hist(bins=20)
    plt.title("Histogram naručenih količina")
    plt.xlabel("Količina")
    plt.ylabel("Broj narudžbi")
    plt.show()


if 'PRICEEACH' in data.columns and 'QUANTITYORDERED' in data.columns:
    plt.figure(figsize=(8, 5))
    plt.scatter(data['QUANTITYORDERED'], data['PRICEEACH'])
    plt.title("Odnos cijene i količine")
    plt.xlabel("Količina")
    plt.ylabel("Cijena (EUR)")
    plt.show()