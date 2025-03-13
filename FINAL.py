import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv('dane_finansowe.csv')
prices = data['Zamkniecie'].values

def taylor_log(x, terms=10):
    if x <= 0:
        raise ValueError("Logarytm naturalny jest zdefiniowany tylko dla x > 0.")
    z = x - 1  # Transformacja do ln(1 + z)
    result = 0
    for n in range(1, terms + 1):
        result += ((-1) ** (n + 1)) * (z ** n) / n  # Szereg Taylora
    return result

log_returns = [taylor_log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]

X = np.arange(len(log_returns)).reshape(-1, 1)  # Indeksy czasowe
y = np.array(log_returns)

model = LinearRegression()
model.fit(X, y)

future_steps = 500
X_all = np.arange(len(log_returns) + future_steps).reshape(-1, 1)
predicted_log_returns = model.predict(X_all)

# Obliczanie wartości portfela
initial_value = 100000

real_values = [initial_value]
for r in log_returns:
    real_values.append(real_values[-1] * np.exp(r))

np.random.seed(42)
mu = np.mean(log_returns)
sigma = np.std(log_returns)
random_log_returns = np.random.normal(mu, sigma, future_steps)

monte_carlo_values = [real_values[-1]]
for r in random_log_returns:
    monte_carlo_values.append(monte_carlo_values[-1] * np.exp(r))

cumulative_log_returns = np.cumsum(predicted_log_returns)
portfolio_values_regression = initial_value * np.exp(cumulative_log_returns)

plt.figure(figsize=(12, 6))

plt.plot(range(len(real_values)), real_values, color='orange', label='Rzeczywiste wartości portfela', linestyle='--', alpha=0.7, zorder=1)

plt.plot(range(len(real_values) - 1, len(real_values) - 1 + len(monte_carlo_values)), monte_carlo_values,
         color='red', label='Monte Carlo (losowe dane)', linestyle='-', alpha=0.9, linewidth=2, zorder=3)

plt.plot(range(len(portfolio_values_regression)), portfolio_values_regression, color='blue', linewidth=2, label='Linia regresji (rzeczywiste dane + prognozy)', zorder=2)

plt.axvline(len(real_values) - 1, color='gray', linestyle=':', label='Koniec danych rzeczywistych', zorder=0)

plt.title('Wartości portfela: rzeczywiste dane, regresja i symulacja Monte Carlo')
plt.xlabel('Czas (indeks)')
plt.ylabel('Wartość portfela [PLN]')
plt.legend()
plt.grid(True)
plt.show()

beta_0 = model.intercept_
beta_1 = model.coef_[0]
print("Współczynnik beta_0: ", beta_0, "Współczynnik beta_1: ", beta_1)
