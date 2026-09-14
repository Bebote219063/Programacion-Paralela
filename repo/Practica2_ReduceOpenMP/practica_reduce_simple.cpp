// Cortes Corona Jonathan Manuel, Lara Jimenez Pablo Cesar, Martinez Delgado Miguel Angel

#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <random>
#include <ctime>
#include <omp.h>

// ============================================================================
// PRIMERA EJECUCION: dejar TAMANO en 100 (arreglo ascendente, se imprime).
// SEGUNDA EJECUCION: cambiar TAMANO a 10000000 y volver a compilar
//                     (arreglo aleatorio de 1 a 2,000,000, no se imprime).
// ============================================================================
const int TAMANO = 100;

class OperacionesArreglos {
public:
    // Llena el arreglo con valores ascendentes: 1, 2, 3, ... n
    void llenarAscendente(int* arreglo, int n) {
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            arreglo[i] = i + 1;
        }
    }

    // Llena el arreglo con valores aleatorios en el rango [minVal, maxVal]
    void llenarAleatorio(int* arreglo, int n, int minVal, int maxVal) {
        std::mt19937 generador(static_cast<unsigned int>(time(NULL)));
        std::uniform_int_distribution<int> distribucion(minVal, maxVal);
        for (int i = 0; i < n; i++) {
            arreglo[i] = distribucion(generador);
        }
    }

    // Sumatoria de todos los elementos del arreglo (reduce +)
    long long sumatoria(const int* arreglo, int n) {
        long long resultado = 0;
        #pragma omp parallel for reduction(+:resultado)
        for (int i = 0; i < n; i++) {
            resultado += arreglo[i];
        }
        return resultado;
    }

    // Promedio de todos los elementos del arreglo
    double promedio(const int* arreglo, int n) {
        return static_cast<double>(sumatoria(arreglo, n)) / n;
    }

    // Maximo de todos los elementos del arreglo (reduce max)
    int maximo(const int* arreglo, int n) {
        int resultado = arreglo[0];
        #pragma omp parallel for reduction(max:resultado)
        for (int i = 0; i < n; i++) {
            if (arreglo[i] > resultado) resultado = arreglo[i];
        }
        return resultado;
    }

    // Minimo de todos los elementos del arreglo (reduce min)
    int minimo(const int* arreglo, int n) {
        int resultado = arreglo[0];
        #pragma omp parallel for reduction(min:resultado)
        for (int i = 0; i < n; i++) {
            if (arreglo[i] < resultado) resultado = arreglo[i];
        }
        return resultado;
    }

    // Imprime el arreglo completo
    void imprimirArreglo(const int* arreglo, int n) {
        std::cout << "[ ";
        for (int i = 0; i < n; i++) {
            std::cout << arreglo[i];
            if (i < n - 1) std::cout << ", ";
        }
        std::cout << " ]\n";
    }
};

int main() {
    // Primera instruccion: nombres del equipo
    std::cout << "Integrantes: Cortes Corona Jonathan Manuel, Lara Jimenez Pablo Cesar, "
              << "Martinez Delgado Miguel Angel\n\n";

    OperacionesArreglos ops;
    // Si el arreglo es pequeno (demostracion 1) se imprime; si es masivo (demostracion 2), no.
    bool imprimirArreglo = (TAMANO <= 1000);

    int* arreglo = nullptr;

    // Manejo de excepciones: si falla la reserva de memoria dinamica
    try {
        arreglo = new int[TAMANO];
    } catch (const std::bad_alloc& e) {
        std::cerr << "[ERROR] No se pudo reservar memoria: " << e.what() << "\n";
        return 1;
    }

    if (imprimirArreglo) {
        ops.llenarAscendente(arreglo, TAMANO);
    } else {
        ops.llenarAleatorio(arreglo, TAMANO, 1, 2000000);
    }

    int opcion = 0;
    double inicio = 0.0, fin = 0.0;

    do {
        std::cout << "\n===================================\n";
        std::cout << "     MENU (tamano del arreglo: " << TAMANO << ")\n";
        std::cout << "===================================\n";
        std::cout << "1. Calcular la sumatoria\n";
        std::cout << "2. Calcular el promedio\n";
        std::cout << "3. Calcular el maximo\n";
        std::cout << "4. Calcular el minimo\n";
        std::cout << "5. Salir del programa\n";
        std::cout << "Seleccione una opcion: ";

        // Manejo de excepciones: entrada no numerica
        if (!(std::cin >> opcion)) {
            std::cin.clear();
            std::cin.ignore(10000, '\n');
            std::cout << "[X] Entrada invalida. Ingrese un numero del 1 al 5.\n";
            continue;
        }

        if (imprimirArreglo && opcion >= 1 && opcion <= 4) {
            std::cout << "\nArreglo antes de la operacion:\n";
            ops.imprimirArreglo(arreglo, TAMANO);
        }

        switch (opcion) {
            case 1: {
                inicio = omp_get_wtime();
                long long resultado = ops.sumatoria(arreglo, TAMANO);
                fin = omp_get_wtime();
                std::cout << "\n--- SUMATORIA ---\n";
                std::cout << "Resultado: " << resultado << "\n";
                std::cout << "Tiempo del calculo en paralelo: "
                          << std::fixed << std::setprecision(8) << (fin - inicio) << " segundos.\n";
                break;
            }
            case 2: {
                inicio = omp_get_wtime();
                double resultado = ops.promedio(arreglo, TAMANO);
                fin = omp_get_wtime();
                std::cout << "\n--- PROMEDIO ---\n";
                std::cout << "Resultado: " << std::fixed << std::setprecision(4) << resultado << "\n";
                std::cout << "Tiempo del calculo en paralelo: "
                          << std::fixed << std::setprecision(8) << (fin - inicio) << " segundos.\n";
                break;
            }
            case 3: {
                inicio = omp_get_wtime();
                int resultado = ops.maximo(arreglo, TAMANO);
                fin = omp_get_wtime();
                std::cout << "\n--- MAXIMO ---\n";
                std::cout << "Resultado: " << resultado << "\n";
                std::cout << "Tiempo del calculo en paralelo: "
                          << std::fixed << std::setprecision(8) << (fin - inicio) << " segundos.\n";
                break;
            }
            case 4: {
                inicio = omp_get_wtime();
                int resultado = ops.minimo(arreglo, TAMANO);
                fin = omp_get_wtime();
                std::cout << "\n--- MINIMO ---\n";
                std::cout << "Resultado: " << resultado << "\n";
                std::cout << "Tiempo del calculo en paralelo: "
                          << std::fixed << std::setprecision(8) << (fin - inicio) << " segundos.\n";
                break;
            }
            case 5:
                std::cout << "\nSaliendo del programa...\n";
                break;
            default:
                std::cout << "\n[X] Opcion invalida. Intente de nuevo.\n";
                break;
        }

    } while (opcion != 5);

    delete[] arreglo;

    // Ultima instruccion: nombres del equipo
    std::cout << "\nIntegrantes: Cortes Corona Jonathan Manuel, Lara Jimenez Pablo Cesar, "
              << "Martinez Delgado Miguel Angel\n";

    return 0;
}
