// Cortes Corona Jonathan Manuel, Lara Jimenez Pablo Cesar, Martinez Delgado Miguel Angel

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define TAM        10   // tamano del arreglo por hilo
#define NUM_HILOS   4   // numero de hilos a utilizar

// Arreglo compartido a nivel de memoria, pero cada hilo SOLO escribe en su
// propia fila (arreglos[hilo]), por lo que funciona como arreglo privado/local
// por hilo, evitando corromper datos aunque exista condicion de carrera en
// el orden de ejecucion e impresion.
int arreglos[NUM_HILOS][TAM];

// Verifica si un valor ya fue generado por ese mismo hilo (evita repetidos)
int existeValor(int *arreglo, int cantidadActual, int valor) {
    for (int i = 0; i < cantidadActual; i++) {
        if (arreglo[i] == valor) return 1;
    }
    return 0;
}

// Generador de numeros pseudoaleatorios propio (LCG), ya que rand_r no
// existe en Windows/MinGW. Cada hilo mantiene su propia "semilla" (estado),
// por lo que no hay conflicto entre hilos al generar numeros.
unsigned int siguienteAleatorio(unsigned int *semilla) {
    *semilla = (*semilla) * 1103515245u + 12345u;
    return (*semilla) % 1000; // rango 0 a 999
}

int main() {
    // Primera instruccion: nombres del equipo
    printf("=================================================\n");
    printf(" Practica: Condicion de carrera con OpenMP\n");
    printf(" Integrantes: Cortes Corona Jonathan Manuel, Lara Jimenez Pablo Cesar,\n");
    printf("              Martinez Delgado Miguel Angel\n");
    printf("=================================================\n\n");

    // 1. Configuracion inicial de OpenMP
    omp_set_num_threads(NUM_HILOS);

    // 2. Region paralela: cada hilo genera sus propios numeros aleatorios unicos
    #pragma omp parallel
    {
        int hilo = omp_get_thread_num();
        // Semilla distinta por hilo para que rand_r no genere la misma secuencia
        unsigned int semilla = (unsigned int) time(NULL) ^ ((hilo + 1) * 7919);
        int contador = 0;
        int valor;

        printf("[Hilo %d] INICIADO\n", hilo);

        while (contador < TAM) {
            valor = siguienteAleatorio(&semilla);

            if (!existeValor(arreglos[hilo], contador, valor)) {
                arreglos[hilo][contador] = valor;
                contador++;

                double porcentaje = (contador / (double) TAM) * 100.0;
                printf("[Hilo %d] Valor %3d guardado en posicion %d -> Avance: %.1f%%\n",
                       hilo, valor, contador - 1, porcentaje);
            }
        }

        printf("[Hilo %d] TERMINADO\n", hilo);

        // Se espera a que todos los hilos terminen de llenar su arreglo
        #pragma omp barrier

        // 3. Impresion del arreglo: solo un hilo imprime, evitando conflictos
        #pragma omp single
        {
            printf("\n---------- IMPRESION DE ARREGLOS POR HILO ----------\n");
            for (int h = 0; h < NUM_HILOS; h++) {
                printf("Hilo %d: [ ", h);
                for (int j = 0; j < TAM; j++) {
                    printf("%d ", arreglos[h][j]);
                }
                printf("]\n");
            }
            printf("-----------------------------------------------------\n");
        }
        // 4. Finalizacion: OpenMP cierra la region paralela automaticamente
    }

    // Ultima instruccion: nombres del equipo
    printf("\n=================================================\n");
    printf(" Fin de la practica - Integrantes: Cortes Corona Jonathan Manuel,\n");
    printf(" Lara Jimenez Pablo Cesar, Martinez Delgado Miguel Angel\n");
    printf("=================================================\n");

    return 0;
}
