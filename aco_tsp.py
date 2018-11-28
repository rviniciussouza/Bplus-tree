import numpy as np
import random as rand
import pylab as gr
import math
import timeit

#ACO aplicado ao problema do caixeiro viajante (TSP)

ncidades = 20
iteracoes = 40
alfa = 1
beta = 1
p = 0.3
feromonioInicial = 0.1
Q = 1

class Formiga:
    def __init__(self, cidade):
        self.origem = cidade
        self.caminho = []
        self.custo = None
        
    def setSolucao(self, solucao, custo):
        if not self.custo:
            self.custo = custo
            caminho = solucao[:]
        else:
            if self.custo > custo:
                self.custo = custo
                caminho = solucao[:]

    def getCusto(self):
        return self.custo

    def getCaminho(self):
        return self.caminho

#funcao conferida
def gerarGrafo():
    G = np.zeros((ncidades, ncidades), int)
    for i in range(ncidades):
        for j in range(i):
            G[i][j] = rand.randrange(1, 50) 
            G[j][i] = G[i][j] 
    return G

def solve(k, G, F, T, formigas):
    visitados = np.empty(ncidades)
    visitados.fill(0)

    distancia_percorrida = 0
    x = k
    caminho = [k]

    for q in range(ncidades-1):
        visitados[x] = 1
        nao_visitada = []
        prob = []
        sumF = 0

        for i in range(ncidades):
            if visitados[i] == 0:
                nao_visitada.append(i) #adiciona cidade nao visitada
                sumF += math.pow(F[x][i], alfa) * math.pow(1.0 / G[x][i], beta)

        for i in range(ncidades):
            if visitados[i] == 0:
                prob.append((math.pow(F[x][i], alfa) * math.pow(1.0 / G[x][i], beta)) / sumF)

        somaPr = sum(prob)
        soma_cur = 0
        sorteio = rand.uniform(0, somaPr)

        for i in range(len(nao_visitada)):
            soma_cur += prob[i]
            if soma_cur >= sorteio:
                distancia_percorrida += G[x][nao_visitada[i]]
                caminho.append(nao_visitada[i])
                x = nao_visitada[i]
                break

    caminho.append(k)
    distancia_percorrida += G[x][k]

    for i in range(len(caminho) - 1):
        T[caminho[i]][caminho[i+1]] += Q/distancia_percorrida
        T[caminho[i+1]][caminho[i]] += Q/distancia_percorrida

    formigas[k].setSolucao(caminho, distancia_percorrida)

def atualizarFeromonio(F, T):
    for i in range(ncidades):
        for j in range(i):
            F[i][j] = (1 - p) * F[i][j]
            if T[i][j] != 0:
                F[i][j] += T[i][j]            
            F[j][i] = F[i][j]

def main():
    
    # G = gerarGrafo()


    #Grafo utilizados nos experimentos do relatorio

    G = [[ 0,  3, 31,  8, 40, 47,  9, 36, 10, 38, 43, 12, 25, 32, 49, 39, 13, 20, 44,  1],
 [ 3,  0, 20, 30, 43, 30, 44,  6, 26, 30, 36, 29, 36, 44, 41, 44, 12, 37, 47,  7],
 [31, 20,  0, 22, 10, 14, 36,  1,  1, 40, 13, 37, 26, 49, 26, 35,  1, 48, 26, 11],
 [ 8, 30, 22,  0, 12,  8,  8, 27,  2,  6, 40,  7, 43,  7, 13, 22, 37, 14, 46, 44],
 [40, 43, 10, 12,  0, 35, 21, 29, 41, 15, 17, 33, 43, 22, 27, 14, 23, 11, 34, 36],
 [47, 30, 14,  8, 35,  0,  6, 15, 20,  1, 11, 20, 39, 25, 20, 46, 16, 33, 45, 10],
 [ 9, 44, 36,  8, 21,  6,  0, 20, 27,  5, 11, 41,  3, 25,  7, 46, 31, 29, 14, 38],
 [36,  6,  1, 27, 29, 15, 20,  0, 30,  5, 32,  6, 18,  3,  5, 33, 24, 19, 40, 24],
 [10, 26,  1,  2, 41, 20, 27, 30,  0, 10, 42, 34,  8, 29, 35, 37, 42, 12, 48, 48],
 [38, 30, 40,  6, 15,  1,  5,  5, 10,  0, 29, 35,  7, 27, 24,  9, 11, 23, 44, 11],
 [43, 36, 13, 40, 17, 11, 11, 32, 42, 29,  0, 41, 16,  7, 47, 27, 23, 28, 37, 37],
 [12, 29, 37,  7, 33, 20, 41,  6, 34, 35, 41,  0, 34, 40,  6, 31,  2,  5, 27, 44],
 [25, 36, 26, 43, 43, 39,  3, 18,  8,  7, 16, 34,  0, 37, 17, 28, 28, 11, 46, 23],
 [32, 44, 49,  7, 22, 25, 25,  3, 29, 27,  7, 40, 37,  0, 42, 15, 16, 29, 47, 39],
 [49, 41, 26, 13, 27, 20,  7,  5, 35, 24, 47,  6, 17, 42,  0, 40,  3, 21,  8, 40],
 [39, 44, 35, 22, 14, 46, 46, 33, 37,  9, 27, 31, 28, 15, 40,  0, 10, 25, 30,  7],
 [13, 12,  1, 37, 23, 16, 31, 24, 42, 11, 23,  2, 28, 16,  3, 10,  0, 21, 42, 49],
 [20, 37, 48, 14, 11, 33, 29, 19, 12, 23, 28,  5, 11, 29, 21, 25, 21,  0, 36, 38],
 [44, 47, 26, 46, 34, 45, 14, 40, 48, 44, 37, 27, 46, 47,  8, 30, 42, 36,  0,  8],
 [ 1,  7, 11, 44, 36, 10, 38, 24, 48, 11, 37, 44, 23, 39, 40,  7, 49, 38,  8,  0]]


    resultado = []

    # inicio = timeit.default_timer()
    F = np.full((ncidades,ncidades), feromonioInicial)
    formigas = [Formiga(i) for i in range(0, ncidades)]

    for i in range(iteracoes):
        T = np.zeros((ncidades, ncidades))
        for j in range(ncidades):
            solve(j, G, F, T, formigas)
        atualizarFeromonio(F, T)

        menor = 1999999 #INF
        for formiga in formigas:
            if formiga.custo < menor:
                menor = formiga.custo

        resultado.append(menor)

    # fim = timeit.default_timer()
    # print ('duracao: %f' % (fim - inicio))

    # print(resultado)

    gr.plot(resultado)
    gr.ylabel("Resultados")
    gr.xlabel("Iterações")
    gr.show()


if __name__ == '__main__':
    main()
