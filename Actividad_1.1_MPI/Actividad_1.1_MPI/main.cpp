// Cortes Corona Jonathan Manuel, Lara Jimenez Pablo Cesar, Martinez Delgado Miguel Angel
//
// Practica: Modelo de programacion y memoria en MPI (MPI + OpenMP hibrido)
//
// Compilacion:
//   Linux:   mpic++ -fopenmp -O2 mpi_openmp_arreglo.cpp -o mpi_openmp_arreglo
//   Windows: ver README (MS-MPI + /openmp)
//
// Ejecucion:
//   PRIMERA EJECUCION  (tamano aleatorio entre 20 y 50, sin argumento):
//       mpiexec -n 4 ./mpi_openmp_arreglo
//   SEGUNDA EJECUCION  (10,000,000 de elementos, se pasa como argumento):
//       mpiexec -n 4 ./mpi_openmp_arreglo 10000000

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cstring>
#include <stdexcept>
#include <mpi.h>
#include <omp.h>

// Rango de los valores aleatorios que se generan en el arreglo
const int VALOR_MIN = 0;
const int VALOR_MAX = 999;

// ============================================================================
// CLASE GeneradorArreglo
// Encapsula el arreglo dinamico de cada proceso MPI: lo reserva, lo llena en
// paralelo con OpenMP mostrando el avance de 10% en 10%, lo imprime y libera
// la memoria en el destructor.
// NO se utiliza vector ni ninguna estructura equivalente: unicamente memoria
// reservada con new[] y liberada con delete[].
// ============================================================================
class GeneradorArreglo {
private:
    int* datos;           // arreglo dinamico
    long long tamano;     // cantidad de elementos
    int rank;             // numero de proceso MPI
    int totalProcesos;    // cantidad total de procesos MPI
    char equipo[MPI_MAX_PROCESSOR_NAME]; // nombre del equipo (nodo)

public:
    // Constructor: reserva el arreglo dinamico
    GeneradorArreglo(long long n, int rankProceso, int procesos, const char* nombreEquipo)
        : datos(nullptr), tamano(n), rank(rankProceso), totalProcesos(procesos) {

        std::strncpy(equipo, nombreEquipo, MPI_MAX_PROCESSOR_NAME - 1);
        equipo[MPI_MAX_PROCESSOR_NAME - 1] = '\0';

        // Manejo de excepciones: si no hay memoria suficiente se informa el error
        try {
            datos = new int[tamano];
        } catch (const std::bad_alloc& e) {
            std::cerr << "[ERROR][Proceso " << rank << " | Equipo " << equipo
                      << "] No se pudo reservar memoria para " << tamano
                      << " elementos: " << e.what() << std::endl;
            throw;
        }
    }

    // Destructor: libera la memoria dinamica reservada
    ~GeneradorArreglo() {
        delete[] datos;
        datos = nullptr;
    }

    long long obtenerTamano() const { return tamano; }

    // ------------------------------------------------------------------------
    // Llena el arreglo dinamico con valores aleatorios usando OpenMP dentro del
    // nodo, mostrando el avance real de 10% en 10% (10, 20, ... 100).
    //
    // Semilla: se usa srand()/rand() como pide la practica para obtener una
    // semilla base distinta en cada proceso MPI. Como rand() mantiene un estado
    // global que NO es seguro cuando varios hilos lo llaman a la vez, cada hilo
    // deriva de esa semilla base su propio estado independiente y genera sus
    // valores con el, evitando condiciones de carrera sobre el generador.
    // ------------------------------------------------------------------------
    void llenarParalelo() {
        // Semilla base distinta por proceso MPI (requisito: srand y rand)
        srand(static_cast<unsigned int>(time(NULL)) + rank * 7919);
        unsigned int semillaBase = static_cast<unsigned int>(rand());

        long long completados = 0; // contador compartido de elementos ya escritos
        int ultimoDecilImpreso = 0; // ultimo porcentaje (en decenas) ya mostrado

        #pragma omp parallel
        {
            int hilo = omp_get_thread_num();

            // Estado propio de cada hilo, derivado de la semilla base del proceso
            unsigned int estado = semillaBase + static_cast<unsigned int>(hilo) * 2654435761u + 1u;

            #pragma omp for schedule(static)
            for (long long i = 0; i < tamano; i++) {
                // Generador congruencial lineal con el estado privado del hilo
                estado = estado * 1103515245u + 12345u;
                datos[i] = VALOR_MIN + static_cast<int>((estado >> 16) % (VALOR_MAX - VALOR_MIN + 1));

                // Se incrementa el contador compartido de forma atomica y se
                // captura el valor resultante, para saber cuantos elementos
                // llevamos escritos sin que dos hilos se pisen.
                long long actual;
                #pragma omp atomic capture
                actual = ++completados;

                // Se imprime solo cuando se cruza un multiplo de 10%.
                // Comparando el decil de "actual" contra el de "actual-1" se
                // garantiza que cada porcentaje (10,20,...,100) se muestre
                // exactamente una vez y refleje el avance real del llenado.
                int decilActual   = static_cast<int>((actual * 10) / tamano);
                int decilAnterior = static_cast<int>(((actual - 1) * 10) / tamano);

                if (decilActual > decilAnterior && decilActual >= 1 && decilActual <= 10) {
                    // critical evita que dos hilos escriban al mismo tiempo.
                    // Ademas, dentro de el se imprimen en ORDEN todos los
                    // porcentajes pendientes hasta el alcanzado por este hilo.
                    // Asi, aunque un hilo que llego al 50% gane el turno antes
                    // que otro que llego al 30%, la salida siempre sera
                    // 10, 20, 30, ... 100, cada uno una sola vez, y solo se
                    // imprime un porcentaje cuando realmente ya se alcanzo.
                    #pragma omp critical(salida)
                    {
                        while (ultimoDecilImpreso < decilActual) {
                            ultimoDecilImpreso++;
                            std::cout << "[Equipo: " << equipo
                                      << " | Proceso MPI: " << rank << "/" << totalProcesos
                                      << " | Hilo OpenMP: " << hilo
                                      << "] Avance del llenado: " << (ultimoDecilImpreso * 10) << "%"
                                      << std::endl;
                        }
                    }
                }
            }
        } // fin de la region paralela: aqui terminan todos los hilos de OpenMP
    }

    // ------------------------------------------------------------------------
    // Imprime el contenido del arreglo identificando equipo y proceso.
    // Si el arreglo es muy grande se muestra una parte representativa, ya que
    // imprimir millones de valores haria ilegible la salida.
    // ------------------------------------------------------------------------
    void imprimir(long long limiteCompleto = 100) const {
        #pragma omp critical(salida)
        {
            std::cout << "\n----- Contenido del arreglo | Equipo: " << equipo
                      << " | Proceso MPI: " << rank << "/" << totalProcesos
                      << " | Elementos: " << tamano << " -----\n";

            if (tamano <= limiteCompleto) {
                std::cout << "[ ";
                for (long long i = 0; i < tamano; i++) {
                    std::cout << datos[i];
                    if (i < tamano - 1) std::cout << ", ";
                }
                std::cout << " ]\n";
            } else {
                std::cout << "(Arreglo demasiado grande para imprimirse completo; "
                          << "se muestran los primeros y ultimos 10 valores)\n";
                std::cout << "Primeros 10: [ ";
                for (long long i = 0; i < 10; i++) std::cout << datos[i] << " ";
                std::cout << "]\nUltimos 10:  [ ";
                for (long long i = tamano - 10; i < tamano; i++) std::cout << datos[i] << " ";
                std::cout << "]\n";
            }
            std::cout << std::flush;
        }
    }
};

// ============================================================================
// MAIN
// ============================================================================
int main(int argc, char* argv[]) {
    // ---------------- Inicializacion de MPI ----------------
    MPI_Init(&argc, &argv);

    int rank = 0, totalProcesos = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   // numero de este proceso
    MPI_Comm_size(MPI_COMM_WORLD, &totalProcesos); // total de procesos

    char equipo[MPI_MAX_PROCESSOR_NAME];
    int longitudNombre = 0;
    MPI_Get_processor_name(equipo, &longitudNombre); // nombre del equipo/nodo

    // Primera instruccion: nombres del equipo de trabajo (solo el proceso 0)
    if (rank == 0) {
        std::cout << "=========================================================\n";
        std::cout << " Practica: Modelo de programacion y memoria en MPI\n";
        std::cout << " Integrantes: Cortes Corona Jonathan Manuel,\n";
        std::cout << "              Lara Jimenez Pablo Cesar,\n";
        std::cout << "              Martinez Delgado Miguel Angel\n";
        std::cout << "=========================================================\n" << std::flush;
    }
    MPI_Barrier(MPI_COMM_WORLD); // todos esperan a que se imprima el encabezado

    // ---------------- Determinacion del tamano del arreglo ----------------
    // Sin argumento  -> PRIMERA EJECUCION: tamano aleatorio entre 20 y 50.
    // Con argumento  -> SEGUNDA EJECUCION: el tamano indicado (ej. 10000000).
    long long tamano = 0;
    if (argc > 1) {
        tamano = atoll(argv[1]);
        if (tamano <= 0) {
            if (rank == 0) std::cerr << "[ERROR] Tamano invalido.\n";
            MPI_Finalize();
            return 1;
        }
    } else {
        // El proceso 0 sortea el tamano y lo comparte para que todos usen el mismo
        if (rank == 0) {
            srand(static_cast<unsigned int>(time(NULL)));
            tamano = 20 + (rand() % 31); // rango 20 a 50
        }
        MPI_Bcast(&tamano, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        std::cout << "\n>>> Tamano del arreglo dinamico por proceso: " << tamano
                  << " elementos\n";
        std::cout << ">>> Procesos MPI: " << totalProcesos
                  << " | Hilos OpenMP por proceso: " << omp_get_max_threads() << "\n\n"
                  << std::flush;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // ---------------- Generacion y llenado del arreglo ----------------
    double inicio = MPI_Wtime();
    try {
        GeneradorArreglo generador(tamano, rank, totalProcesos, equipo);
        generador.llenarParalelo();

        double fin = MPI_Wtime();

        // Los procesos imprimen su arreglo en orden para que la salida sea legible
        for (int p = 0; p < totalProcesos; p++) {
            if (p == rank) {
                generador.imprimir();
                std::cout << "[Equipo: " << equipo << " | Proceso MPI: " << rank
                          << "] Tiempo de llenado: " << std::fixed << std::setprecision(6)
                          << (fin - inicio) << " segundos\n" << std::flush;
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        // Al salir de este bloque el destructor libera la memoria dinamica
    } catch (const std::bad_alloc&) {
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Ultima instruccion: nombres del equipo de trabajo
    if (rank == 0) {
        std::cout << "\n=========================================================\n";
        std::cout << " Fin de la ejecucion - Integrantes: Cortes Corona Jonathan Manuel,\n";
        std::cout << " Lara Jimenez Pablo Cesar, Martinez Delgado Miguel Angel\n";
        std::cout << "=========================================================\n" << std::flush;
    }

    // ---------------- Finalizacion de MPI ----------------
    MPI_Finalize();
    return 0;
}
