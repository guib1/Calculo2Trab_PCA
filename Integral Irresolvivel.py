import numpy as np

def f(x):
    return np.exp(x) / x
    
n1 = 1
n2 = 2
rn = 1000000

x_random = np.random.uniform(n1, n2, rn)
f_random = f(x_random)
f_mean = np.mean(f_random)

integral = (n2 - n1) * f_mean

print("Calculo de uma Integral 'irresolvivel' usando Monte Carlo")
print("\nIntegral: I = l(e^x / x) dx de 1 a 2")
print("Quantidade de valores aleatorios (rn):", rn)
print("Intervalo de integracao: [", n1, ",", n2, "]")
print("\nValor aproximado da integral: {:.5f}".format(integral))