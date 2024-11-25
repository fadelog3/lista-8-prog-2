#Questão 1

import numpy as np
import matplotlib.pyplot as plt

class AjustePolinomio:
    def __init__(self, pontos_x, pontos_y, d):

        self.pontos_x = np.array(pontos_x)
        self.pontos_y = np.array(pontos_y)
        self.grau = d
        self.coeficientes = None
    
    def ajustar(self):
        #utiliza o metodo dos minimos quadsrados para achar o polinomio
        X = np.vander(self.pontos_x, self.grau + 1, increasing=True)
        self.coeficientes = np.linalg.lstsq(X, self.pontos_y, rcond=None)[0]
        
    def prever(self, x):
        X = np.vander(x, self.grau + 1, increasing=True)
        return np.dot(X, self.coeficientes)
    
    def plotar_ajuste(self):
        plt.scatter(self.pontos_x, self.pontos_y, color='blue', label='Pontos reais')
        
        x_range = np.linspace(min(self.pontos_x), max(self.pontos_x), 1000)
        y_range = self.prever(x_range)
        
        plt.plot(x_range, y_range, color='red', label=f'Polinômio de grau {self.grau}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()

if __name__=='__main__':
    pontos_x = [1, 2, 3, 4, 5]
    pontos_y = [2.2, 2.8, 3.6, 5.1, 8.5]
    grau = 7

    ajuste = AjustePolinomio(pontos_x, pontos_y, grau)
    ajuste.ajustar()

    ajuste.plotar_ajuste()

    novos_x = np.array([6, 7])
    previsoes = ajuste.prever(novos_x)
    print(f"Previsões para x = {novos_x}: {previsoes}")

#Questão 2

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class AjustePolinomioAutomatico:
    def __init__(self, pontos_x, pontos_y, max_grau=10):
        self.pontos_x = np.array(pontos_x)
        self.pontos_y = np.array(pontos_y)
        self.max_grau = max_grau
        self.grau_otimo = None
        self.coeficientes = None

    def ajustar_polinomio(self, grau, pontos_x, pontos_y):
        X = np.vander(pontos_x, grau + 1, increasing=True)
        coeficientes = np.linalg.lstsq(X, pontos_y, rcond=None)[0]
        return coeficientes

    def erro_mse(self, coeficientes, grau, pontos_x, pontos_y):
        X = np.vander(pontos_x, grau + 1, increasing=True)
        previsoes = np.dot(X, coeficientes)
        return mean_squared_error(pontos_y, previsoes)

    def selecionar_grau_otimo(self):
        X_treino, X_val, y_treino, y_val = train_test_split(self.pontos_x, self.pontos_y, test_size=0.3, random_state=42)
        
        erros = []
        
        for grau in range(1, self.max_grau + 1):
            coeficientes = self.ajustar_polinomio(grau, X_treino, y_treino)
            
            mse = self.erro_mse(coeficientes, grau, X_val, y_val)
            erros.append(mse)

        self.grau_otimo = np.argmin(erros) + 1 
        self.coeficientes = self.ajustar_polinomio(self.grau_otimo, self.pontos_x, self.pontos_y)

    def plotar_ajuste(self):
        """
        Plota os pontos originais e o polinômio ajustado.
        """
        plt.scatter(self.pontos_x, self.pontos_y, color='blue', label='Pontos reais')

        x_range = np.linspace(min(self.pontos_x), max(self.pontos_x), 1000)
        X = np.vander(x_range, self.grau_otimo + 1, increasing=True)
        y_range = np.dot(X, self.coeficientes)
        
        plt.plot(x_range, y_range, color='red', label=f'Polinômio de grau {self.grau_otimo}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()

if __name__=='__main__':
      pontos_x = np.array([1, 2, 3, 4, 5.6, 6, 13, 8, 9])
      pontos_y = np.array([1, -4, 9, 6, 25, -36, 59, 69, 81]) 

      ajuste_automatico = AjustePolinomioAutomatico(pontos_x, pontos_y)
      ajuste_automatico.selecionar_grau_otimo()

      ajuste_automatico.plotar_ajuste()

      print(f"O grau ótimo do polinômio é: {ajuste_automatico.grau_otimo}")

#Questão 3
#item a)
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def erro_absoluto(params, x, y):
    a, b = params
    return np.sum(np.abs(a * x + b - y))

def ajuste_linear(x, y):
    params_iniciais = [0, 0]
    
    resultado = minimize(erro_absoluto, params_iniciais, args=(x, y))

    a_otimo, b_otimo = resultado.x
    return a_otimo, b_otimo

def plotar_ajuste(x, y, a, b):

    plt.scatter(x, y, color='blue', label='Pontos reais')
    plt.plot(x, a * x + b, color='red', label=f'Linha ajustada: y = {a:.2f}x + {b:.2f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

if __name__=='__main__':
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2.3, 3.1, 3.9, 5.2, 6.9])

    a, b = ajuste_linear(x, y)

    plotar_ajuste(x, y, a, b)

    print(f"Coeficiente a: {a:.2f}")
    print(f"Coeficiente b: {b:.2f}")

#item b)
import numpy as np

def generate_points(m):
    np.random.seed(1)
    a = 6
    b = -3
    x = np.linspace(0, 10, m)
    y = a*x + b + np.random.standard_cauchy(size=m)
    return (x, y)

def save_points(points, path='test_points.txt'):
    with open(path, 'wt') as f:
        for x, y in zip(points[0], points[1]):
            f.write(f'{x} {y}\n')

if __name__ == "__main__":
    # Gerar e salvar os conjuntos de pontos para diferentes tamanhos de dados
    for m in [64, 128, 256, 512, 1024]:
        points = generate_points(m)
        save_points(points, f'CodigosExcercicios/xy_{m}.txt')
        print(f"Conjunto de {m} pontos salvo em: CodigosExcercicios/xy_{m}.txt")

def carregar_pontos(arquivo):
    """
    Carrega os pontos de um arquivo de texto.
    :param arquivo: Caminho para o arquivo com os pontos.
    :return: Arrays x e y.
    """
    pontos = np.loadtxt(arquivo)
    return pontos[:, 0], pontos[:, 1]

# Executar o ajuste para diferentes tamanhos de dados
for m in [64, 128, 256, 512, 1024]:
    arquivo = f'CodigosExcercicios/xy_{m}.txt'
    x, y = carregar_pontos(arquivo)
    
    # Ajuste da linha
    a, b = ajuste_linear(x, y)
    
    # Exibir o gráfico com a linha ajustada
    plotar_ajuste(x, y, a, b, m)
    
    # Exibir os coeficientes a e b ajustados
    print(f"Para {m} pontos:")
    print(f"Coeficiente a: {a:.2f}")
    print(f"Coeficiente b: {b:.2f}")
    print("-" * 40)

#item c e d

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def generate_points(m):
    np.random.seed(1)
    a = 6
    b = -3
    x = np.linspace(0, 10, m)
    y = a * x + b + np.random.standard_cauchy(size=m) 
    return x, y

def erro_quadratico(params, x, y):
    a, b = params
    return np.sum((a * x + b - y) ** 2)

def ajuste_linear_mínimos_quadrados(x, y):
    params_iniciais = [0, 0]
    resultado = minimize(erro_quadratico, params_iniciais, args=(x, y))
    a_otimo, b_otimo = resultado.x
    return a_otimo, b_otimo

def plotar_ajuste(x, y, a, b, m):
    plt.scatter(x, y, color='blue', label='Pontos reais', s=10)
    plt.plot(x, a * x + b, color='red', label=f'Linha ajustada: y = {a:.2f}x + {b:.2f}', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Ajuste Linear - {m} Pontos')
    plt.legend()
    plt.grid(True)
    plt.show()


for m in [64, 128, 256, 512, 1024]:
    x, y = generate_points(m)

    a, b = ajuste_linear_mínimos_quadrados(x, y)
    
    plotar_ajuste(x, y, a, b, m)
    
    print(f"Para {m} pontos:")
    print(f"Coeficiente a (erros quadráticos): {a:.2f}")
    print(f"Coeficiente b (erros quadráticos): {b:.2f}")
    print("-" * 40)

#item e)
#Quando se utiliza a soma das diferenças absolutas, grandes desvios nos valores de y
#y têm um impacto menor no modelo final do que na minimização dos erros quadráticos. 
#Isso ocorre porque a função absoluta cresce linearmente, enquanto o erro quadrático aumenta exponencialmente.

#A técnica de mínimos quadrados pode ser resolvida de forma simples e rápida, sem a necessidade de otimização numérica, como acontece na minimização dos erros absolutos. 
#Isso faz com que a implementação da solução seja mais simples
