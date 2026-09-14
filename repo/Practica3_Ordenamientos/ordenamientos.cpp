// Cortes Corona Jonathan Manuel, Lara Jimenez Pablo Cesar, Martinez Delgado Miguel Angel

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <omp.h>

// ============================================================================
// PRIMERA EJECUCION: dejar TAMANO en 100, RANGO_MIN/RANGO_MAX en 0/200
//                     (se imprime el arreglo antes y despues de ordenar).
// SEGUNDA EJECUCION: cambiar TAMANO a 10000000 y RANGO_MAX a 2000000, y volver
//                     a compilar (no se imprime el arreglo, solo tiempos).
// ============================================================================
const int TAMANO    = 100;
const int RANGO_MIN = 0;
const int RANGO_MAX = 200;

// Umbral para dejar de crear tareas nuevas en el merge sort paralelo: por
// debajo de este tamano de tramo, el costo de crear una tarea es mayor que
// el beneficio de paralelizar un pedazo tan chico.
const int UMBRAL_TAREA = 1000;

// ----------------------------------------------------------------------------
// LLENADO DEL ARREGLO (paralelo, con semilla independiente por hilo)
// ----------------------------------------------------------------------------
void llenarArreglo(int* arreglo, int n, double& tiempo) {
    double inicio = omp_get_wtime();

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        // thread_local: esta variable se crea UNA SOLA VEZ por cada hilo (la
        // primera vez que ese hilo pasa por aqui) y conserva su estado en las
        // siguientes iteraciones que le toquen a ese mismo hilo. Al sembrarla
        // con el numero de hilo, cada hilo genera una secuencia distinta.
        thread_local std::mt19937 generador(
            static_cast<unsigned int>(time(NULL)) ^ (omp_get_thread_num() * 2654435761u + 40503u));
        std::uniform_int_distribution<int> distribucion(RANGO_MIN, RANGO_MAX);
        arreglo[i] = distribucion(generador);
    }

    double fin = omp_get_wtime();
    tiempo = fin - inicio;
}

void copiarArreglo(const int* origen, int* destino, int n) {
    for (int i = 0; i < n; i++) destino[i] = origen[i];
}

void imprimirArreglo(const int* arreglo, int n) {
    std::cout << "[ ";
    for (int i = 0; i < n; i++) {
        std::cout << arreglo[i];
        if (i < n - 1) std::cout << ", ";
    }
    std::cout << " ]\n";
}

bool verificarOrdenado(const int* arreglo, int n) {
    for (int i = 1; i < n; i++) {
        if (arreglo[i - 1] > arreglo[i]) return false;
    }
    return true;
}

// ============================================================================
// ORDENAMIENTO ITERATIVO: Shell Sort (secuencial y paralelo)
// ============================================================================
// Se eligio Shell Sort y no Burbuja/Seleccion/Insercion porque estas ultimas
// son O(n^2): con 10,000,000 de elementos tardarian horas en terminar. Shell
// Sort es mucho mas eficiente y ademas se paraleliza de forma natural.

// Version secuencial
void shellSecuencial(int* arreglo, int n) {
    for (int salto = n / 2; salto > 0; salto /= 2) {
        for (int i = salto; i < n; i++) {
            int temp = arreglo[i];
            int j = i;
            while (j >= salto && arreglo[j - salto] > temp) {
                arreglo[j] = arreglo[j - salto];
                j -= salto;
            }
            arreglo[j] = temp;
        }
    }
}

// Version paralela: para un mismo "salto", el arreglo se divide en "salto"
// sub-secuencias independientes (columna 0: indices 0, salto, 2*salto, ...;
// columna 1: indices 1, salto+1, 2*salto+1, ...; etc.). Cada columna no
// comparte indices con las demas, por lo que se pueden ordenar (con insercion)
// en paralelo sin condicion de carrera. No se necesita critical ni atomic
// aqui porque no hay dato compartido en conflicto.
// Lo que SI es indispensable es sincronizar entre un valor de "salto" y el
// siguiente: el paso con salto mas chico depende de que TODAS las columnas
// del salto anterior hayan terminado. Esa sincronizacion la da el barrier
// IMPLICITO al final de "#pragma omp parallel for" (cada valor de salto abre
// y cierra su propia region paralela antes de pasar al siguiente).
void shellParalelo(int* arreglo, int n) {
    for (int salto = n / 2; salto > 0; salto /= 2) {
        #pragma omp parallel for
        for (int col = 0; col < salto; col++) {
            for (int i = col + salto; i < n; i += salto) {
                int temp = arreglo[i];
                int j = i;
                while (j >= col + salto && arreglo[j - salto] > temp) {
                    arreglo[j] = arreglo[j - salto];
                    j -= salto;
                }
                arreglo[j] = temp;
            }
        }
        // Barrier implicito: termina aqui la region paralela de este "salto"
        // antes de que el for exterior pase al siguiente salto (mas chico).
    }
}

// ============================================================================
// ORDENAMIENTO RECURSIVO: Merge Sort (secuencial) / Merge Sort con tareas
// ============================================================================

void mezclar(int* arreglo, int inicio, int medio, int fin, int* temporal) {
    int i = inicio, j = medio + 1, k = inicio;
    while (i <= medio && j <= fin) {
        if (arreglo[i] <= arreglo[j]) temporal[k++] = arreglo[i++];
        else                          temporal[k++] = arreglo[j++];
    }
    while (i <= medio) temporal[k++] = arreglo[i++];
    while (j <= fin)   temporal[k++] = arreglo[j++];
    for (int x = inicio; x <= fin; x++) arreglo[x] = temporal[x];
}

// Version secuencial
void mergeSortSecuencial(int* arreglo, int inicio, int fin, int* temporal) {
    if (inicio >= fin) return;
    int medio = inicio + (fin - inicio) / 2;
    mergeSortSecuencial(arreglo, inicio, medio, temporal);
    mergeSortSecuencial(arreglo, medio + 1, fin, temporal);
    mezclar(arreglo, inicio, medio, fin, temporal);
}

// Version paralela mediante tareas: cada mitad se lanza como una tarea
// independiente (#pragma omp task). #pragma omp taskwait espera a que ambas
// mitades terminen antes de mezclarlas, porque la mezcla necesita las dos
// mitades ya ordenadas (es una dependencia real de datos, no un requisito
// artificial). No se usa taskgroup porque cada nivel de la recursion ya
// espera a sus propias tareas hijas con taskwait, lo cual es suficiente para
// garantizar que toda la recursion termine correctamente.
void mergeSortParalelo(int* arreglo, int inicio, int fin, int* temporal) {
    if (inicio >= fin) return;

    if (fin - inicio < UMBRAL_TAREA) {
        mergeSortSecuencial(arreglo, inicio, fin, temporal);
        return;
    }

    int medio = inicio + (fin - inicio) / 2;

    #pragma omp task shared(arreglo, temporal)
    mergeSortParalelo(arreglo, inicio, medio, temporal);

    #pragma omp task shared(arreglo, temporal)
    mergeSortParalelo(arreglo, medio + 1, fin, temporal);

    #pragma omp taskwait
    mezclar(arreglo, inicio, medio, fin, temporal);
}

// Punto de entrada para el merge sort paralelo: se abre una region paralela
// y SOLO UN hilo (#pragma omp single) inicia el arbol de tareas; si no se
// usara single, cada uno de los hilos de la region intentaria iniciar su
// propio arbol de tareas completo, duplicando el trabajo en vez de repartirlo.
void ejecutarMergeSortParalelo(int* arreglo, int n, int* temporal) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            mergeSortParalelo(arreglo, 0, n - 1, temporal);
        }
    }
}

// ============================================================================
// MAIN: menu ciclado
// ============================================================================
int main() {
    // Primera instruccion: nombres del equipo
    std::cout << "Integrantes: Cortes Corona Jonathan Manuel, Lara Jimenez Pablo Cesar, "
              << "Martinez Delgado Miguel Angel\n";

    bool mostrarArreglo = (TAMANO <= 1000);

    int* original  = nullptr;
    int* trabajo   = nullptr;
    int* temporal  = nullptr;

    // Manejo de excepciones: si falla la reserva de memoria dinamica
    try {
        original = new int[TAMANO];
        trabajo  = new int[TAMANO];
        temporal = new int[TAMANO];
    } catch (const std::bad_alloc& e) {
        std::cerr << "[ERROR] No se pudo reservar memoria: " << e.what() << "\n";
        return 1;
    }

    bool arregloListo = false;
    double tLlenado = 0, tIterSec = 0, tIterPar = 0, tRecSec = 0, tRecPar = 0;
    bool listoIterSec = false, listoIterPar = false, listoRecSec = false, listoRecPar = false;

    int opcion = 0;
    do {
        std::cout << "\n=========================================================\n";
        std::cout << "   MENU - ORDENAMIENTOS CON OPENMP (tamano: " << TAMANO << ")\n";
        std::cout << "=========================================================\n";
        std::cout << "1. Llenar arreglo\n";
        std::cout << "2. Ordenamiento iterativo secuencial (Shell Sort)\n";
        std::cout << "3. Ordenamiento iterativo paralelo (Shell Sort)\n";
        std::cout << "4. Ordenamiento recursivo secuencial (Merge Sort)\n";
        std::cout << "5. Ordenamiento recursivo paralelo (Merge Sort con tareas)\n";
        std::cout << "6. Mostrar resultados y speedup\n";
        std::cout << "7. Salir\n";
        std::cout << "Seleccione una opcion: ";

        // Manejo de excepciones: entrada no numerica
        if (!(std::cin >> opcion)) {
            std::cin.clear();
            std::cin.ignore(10000, '\n');
            std::cout << "[X] Entrada invalida. Ingrese un numero del 1 al 7.\n";
            continue;
        }

        switch (opcion) {
            case 1: {
                llenarArreglo(original, TAMANO, tLlenado);
                arregloListo = true;
                listoIterSec = listoIterPar = listoRecSec = listoRecPar = false;
                std::cout << "\n[!] Arreglo llenado. Tiempo de llenado: "
                          << std::fixed << std::setprecision(8) << tLlenado << " segundos.\n";
                if (mostrarArreglo) {
                    std::cout << "Arreglo desordenado:\n";
                    imprimirArreglo(original, TAMANO);
                }
                break;
            }
            case 2: {
                if (!arregloListo) { std::cout << "\n[X] Primero llena el arreglo (opcion 1).\n"; break; }
                copiarArreglo(original, trabajo, TAMANO);
                double t0 = omp_get_wtime();
                shellSecuencial(trabajo, TAMANO);
                double t1 = omp_get_wtime();
                tIterSec = t1 - t0;
                listoIterSec = true;
                std::cout << "\n--- ITERATIVO SECUENCIAL (Shell Sort) ---\n";
                std::cout << "Ordenado correctamente: " << (verificarOrdenado(trabajo, TAMANO) ? "Si" : "No") << "\n";
                std::cout << "Tiempo: " << std::fixed << std::setprecision(8) << tIterSec << " segundos.\n";
                if (mostrarArreglo) { std::cout << "Arreglo ordenado:\n"; imprimirArreglo(trabajo, TAMANO); }
                break;
            }
            case 3: {
                if (!arregloListo) { std::cout << "\n[X] Primero llena el arreglo (opcion 1).\n"; break; }
                copiarArreglo(original, trabajo, TAMANO);
                double t0 = omp_get_wtime();
                shellParalelo(trabajo, TAMANO);
                double t1 = omp_get_wtime();
                tIterPar = t1 - t0;
                listoIterPar = true;
                std::cout << "\n--- ITERATIVO PARALELO (Shell Sort) ---\n";
                std::cout << "Ordenado correctamente: " << (verificarOrdenado(trabajo, TAMANO) ? "Si" : "No") << "\n";
                std::cout << "Tiempo: " << std::fixed << std::setprecision(8) << tIterPar << " segundos.\n";
                if (mostrarArreglo) { std::cout << "Arreglo ordenado:\n"; imprimirArreglo(trabajo, TAMANO); }
                break;
            }
            case 4: {
                if (!arregloListo) { std::cout << "\n[X] Primero llena el arreglo (opcion 1).\n"; break; }
                copiarArreglo(original, trabajo, TAMANO);
                double t0 = omp_get_wtime();
                mergeSortSecuencial(trabajo, 0, TAMANO - 1, temporal);
                double t1 = omp_get_wtime();
                tRecSec = t1 - t0;
                listoRecSec = true;
                std::cout << "\n--- RECURSIVO SECUENCIAL (Merge Sort) ---\n";
                std::cout << "Ordenado correctamente: " << (verificarOrdenado(trabajo, TAMANO) ? "Si" : "No") << "\n";
                std::cout << "Tiempo: " << std::fixed << std::setprecision(8) << tRecSec << " segundos.\n";
                if (mostrarArreglo) { std::cout << "Arreglo ordenado:\n"; imprimirArreglo(trabajo, TAMANO); }
                break;
            }
            case 5: {
                if (!arregloListo) { std::cout << "\n[X] Primero llena el arreglo (opcion 1).\n"; break; }
                copiarArreglo(original, trabajo, TAMANO);
                double t0 = omp_get_wtime();
                ejecutarMergeSortParalelo(trabajo, TAMANO, temporal);
                double t1 = omp_get_wtime();
                tRecPar = t1 - t0;
                listoRecPar = true;
                std::cout << "\n--- RECURSIVO PARALELO (Merge Sort con tareas) ---\n";
                std::cout << "Ordenado correctamente: " << (verificarOrdenado(trabajo, TAMANO) ? "Si" : "No") << "\n";
                std::cout << "Tiempo: " << std::fixed << std::setprecision(8) << tRecPar << " segundos.\n";
                if (mostrarArreglo) { std::cout << "Arreglo ordenado:\n"; imprimirArreglo(trabajo, TAMANO); }
                break;
            }
            case 6: {
                std::cout << "\n===================== RESULTADOS =====================\n";
                std::cout << std::left << std::setw(24) << "Algoritmo"
                          << std::setw(12) << "Version"
                          << std::setw(16) << "Tiempo (s)"
                          << "Speedup\n";
                std::cout << "--------------------------------------------------------\n";
                if (listoIterSec)
                    std::cout << std::left << std::setw(24) << "Iterativo (Shell Sort)" << std::setw(12) << "Secuencial"
                              << std::setw(16) << std::fixed << std::setprecision(8) << tIterSec << "---\n";
                if (listoIterPar)
                    std::cout << std::left << std::setw(24) << "Iterativo (Shell Sort)" << std::setw(12) << "Paralelo"
                              << std::setw(16) << std::fixed << std::setprecision(8) << tIterPar
                              << (listoIterSec ? std::to_string(tIterSec / tIterPar) : std::string("---")) << "\n";
                if (listoRecSec)
                    std::cout << std::left << std::setw(24) << "Recursivo (MergeSort)" << std::setw(12) << "Secuencial"
                              << std::setw(16) << std::fixed << std::setprecision(8) << tRecSec << "---\n";
                if (listoRecPar)
                    std::cout << std::left << std::setw(24) << "Recursivo (MergeSort)" << std::setw(12) << "Paralelo"
                              << std::setw(16) << std::fixed << std::setprecision(8) << tRecPar
                              << (listoRecSec ? std::to_string(tRecSec / tRecPar) : std::string("---")) << "\n";
                if (!listoIterSec && !listoIterPar && !listoRecSec && !listoRecPar)
                    std::cout << "(Todavia no se ha ejecutado ningun ordenamiento. Usa las opciones 2-5.)\n";
                std::cout << "=========================================================\n";
                break;
            }
            case 7:
                std::cout << "\nSaliendo del programa...\n";
                break;
            default:
                std::cout << "\n[X] Opcion invalida. Intente de nuevo.\n";
                break;
        }

    } while (opcion != 7);

    delete[] original;
    delete[] trabajo;
    delete[] temporal;

    // Ultima instruccion: nombres del equipo
    std::cout << "\nIntegrantes: Cortes Corona Jonathan Manuel, Lara Jimenez Pablo Cesar, "
              << "Martinez Delgado Miguel Angel\n";

    return 0;
}
