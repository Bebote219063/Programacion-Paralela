// Cortes Corona Jonathan Manuel, Lara Jimenez Pablo Cesar, Martinez Delgado Miguel Angel
// Practica 1.3: Reduce en MPI con OpenMP
//
// Compilacion:  mpic++ -fopenmp -O2 reduce_mpi.cpp -o reduce_mpi
// Ejecucion local:       mpirun -np 5 ./reduce_mpi 40
// Ejecucion distribuida: mpirun --hostfile hostfile -np 5 ./reduce_mpi 4000000

#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>

// ============================================================================
// CLASE Registro: escribe simultaneamente en pantalla y en el log local del nodo
// Cada nodo crea su propio archivo log_equipo_[NOMBRE_PC]_nodo_[RANK].txt
// ============================================================================
class Registro {
private:
    std::ofstream archivo;
    std::string nombreArchivo;
public:
    void abrir(const char* equipo, int rank) {
        std::ostringstream n;
        n << "log_equipo_" << equipo << "_nodo_" << rank << ".txt";
        nombreArchivo = n.str();
        archivo.open(nombreArchivo.c_str(), std::ios::out | std::ios::trunc);
    }
    // Escribe la linea en consola y en el archivo de log del nodo
    void escribir(const std::string& linea) {
        std::cout << linea << std::endl;
        if (archivo.is_open()) { archivo << linea << "\n"; archivo.flush(); }
    }
    // Solo al log (para no saturar la consola)
    void soloLog(const std::string& linea) {
        if (archivo.is_open()) { archivo << linea << "\n"; archivo.flush(); }
    }
    void cerrar() { if (archivo.is_open()) archivo.close(); }
    const std::string& nombre() const { return nombreArchivo; }
};

// ============================================================================
// CLASE OperacionesArreglos
// Maneja la seccion local de los arreglos dinamicos A, B y C de cada proceso.
// Todas las operaciones usan OpenMP dentro del nodo y reportan equipo,
// proceso MPI e hilo OpenMP.
// NO se usa vector ni estructuras equivalentes: solo punteros con new[]/delete[].
// ============================================================================
class OperacionesArreglos {
private:
    long long* A;        // seccion local del arreglo A
    long long* B;        // seccion local del arreglo B
    long long* C;        // seccion local del resultado
    long long  nLocal;   // elementos de la seccion local
    long long  nGlobal;  // elementos totales del arreglo global
    int   rank;          // numero de proceso MPI
    int   totalProcs;    // total de procesos MPI
    char  equipo[MPI_MAX_PROCESSOR_NAME];
    Registro* log;
    bool  detalle;       // true = imprime elemento a elemento (primera ejecucion)

    // Construye un mensaje con el formato pedido por la practica
    std::string msg(int hilo, long long pos, const char* operacion,
                    const std::string& valor) const {
        std::ostringstream s;
        s << "[Equipo: " << equipo << "] [Proceso MPI: " << rank << "]";
        if (hilo >= 0) s << " [Hilo OpenMP: " << hilo << "]";
        if (pos >= 0)  s << " [Posicion: " << pos << "]";
        s << " [Operacion: " << operacion << "] [Valor: " << valor << "]";
        return s.str();
    }
    static std::string num(long long v) { std::ostringstream s; s << v; return s.str(); }
    static std::string dec(double v) { std::ostringstream s; s << std::fixed << std::setprecision(4) << v; return s.str(); }

public:
    OperacionesArreglos(int r, int tp, const char* eq, Registro* lg)
        : A(nullptr), B(nullptr), C(nullptr), nLocal(0), nGlobal(0),
          rank(r), totalProcs(tp), log(lg), detalle(true) {
        std::strncpy(equipo, eq, MPI_MAX_PROCESSOR_NAME - 1);
        equipo[MPI_MAX_PROCESSOR_NAME - 1] = '\0';
    }

    ~OperacionesArreglos() { liberar(); }

    void liberar() {
        delete[] A; delete[] B; delete[] C;
        A = B = C = nullptr;
    }

    bool listo() const { return A != nullptr; }
    long long tamLocal() const { return nLocal; }
    long long tamGlobal() const { return nGlobal; }
    void setDetalle(bool d) { detalle = d; }
    long long* ptrA() { return A; }
    long long* ptrC() { return C; }

    // ---------- 1. Crear arreglos: memoria dinamica para la seccion local ----
    void crearArregloMPI(long long nGlob) {
        liberar();
        nGlobal = nGlob;
        nLocal  = nGlob / totalProcs;      // cada proceso toma su porcion
        try {
            A = new long long[nLocal];
            B = new long long[nLocal];
            C = new long long[nLocal];
        } catch (const std::bad_alloc& e) {
            log->escribir(msg(-1, -1, "ERROR de memoria", e.what()));
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (long long i = 0; i < nLocal; i++) { A[i] = B[i] = C[i] = 0; }
        std::ostringstream s;
        s << nLocal << " elementos locales (global " << nGlobal << ")";
        log->escribir(msg(-1, -1, "Crear arreglos", s.str()));
    }

    // ---------- 6. Llenar secuencial: valores deterministas por proceso -----
    void llenarSecuencial() {
        long long inicio = (long long)rank * nLocal;   // desplazamiento global
        #pragma omp parallel for
        for (long long i = 0; i < nLocal; i++) {
            A[i] = inicio + i + 1;
            B[i] = (inicio + i + 1) * 2;
            if (detalle) {
                #pragma omp critical
                {
                    std::ostringstream v;
                    v << "A=" << A[i] << " B=" << B[i];
                    log->escribir(msg(omp_get_thread_num(), inicio + i,
                                      "Llenar secuencial", v.str()));
                }
            }
        }
        log->soloLog(msg(-1, -1, "Llenar secuencial", "seccion completa"));
    }

    // ---------- 7. Llenar aleatorio: rango 1 a 1,000,000 --------------------
    // Cada nodo inicializa su semilla con srand(tiempo + rank) para que las
    // secuencias no se repitan entre nodos. Como rand() tiene estado global y
    // no es seguro entre hilos, cada hilo deriva de esa semilla su propio
    // estado y genera con el, evitando condiciones de carrera.
    void llenarAleatorio() {
        srand((unsigned int)time(NULL) + (unsigned int)rank * 7919u);
        unsigned int base = (unsigned int)rand();
        long long inicio = (long long)rank * nLocal;

        #pragma omp parallel
        {
            int hilo = omp_get_thread_num();
            unsigned int estado = base + (unsigned int)hilo * 2654435761u + 1u;
            #pragma omp for
            for (long long i = 0; i < nLocal; i++) {
                estado = estado * 1103515245u + 12345u;
                A[i] = 1 + (long long)((estado >> 8) % 1000000);
                estado = estado * 1103515245u + 12345u;
                B[i] = 1 + (long long)((estado >> 8) % 1000000);
                if (detalle) {
                    #pragma omp critical
                    {
                        std::ostringstream v;
                        v << "A=" << A[i] << " B=" << B[i];
                        log->escribir(msg(hilo, inicio + i, "Llenar aleatorio", v.str()));
                    }
                }
            }
        }
        log->soloLog(msg(-1, -1, "Llenar aleatorio", "seccion completa"));
    }

    // ---------- 2 a 5. Operaciones aritmeticas elemento a elemento ----------
    // op: 1 suma, 2 resta, 3 multiplicacion, 4 cuadrado de A
    void operacionElemento(int op) {
        const char* nombre = (op == 1) ? "Suma A+B" : (op == 2) ? "Resta A-B"
                           : (op == 3) ? "Multiplicacion A*B" : "Cuadrado de A";
        long long inicio = (long long)rank * nLocal;

        #pragma omp parallel for
        for (long long i = 0; i < nLocal; i++) {
            switch (op) {
                case 1: C[i] = A[i] + B[i]; break;
                case 2: C[i] = A[i] - B[i]; break;
                case 3: C[i] = A[i] * B[i]; break;
                default: C[i] = A[i] * A[i]; break;
            }
            if (detalle) {
                #pragma omp critical
                log->escribir(msg(omp_get_thread_num(), inicio + i, nombre, num(C[i])));
            }
        }
        log->soloLog(msg(-1, -1, nombre, "seccion procesada"));
    }

    // ---------- Nivel 1: reducciones locales con OpenMP ---------------------
    long long sumaLocal() const {
        long long s = 0;
        #pragma omp parallel for reduction(+:s)
        for (long long i = 0; i < nLocal; i++) s += A[i];
        return s;
    }
    long long maxLocal() const {
        long long m = A[0];
        #pragma omp parallel for reduction(max:m)
        for (long long i = 0; i < nLocal; i++) if (A[i] > m) m = A[i];
        return m;
    }
    long long minLocal() const {
        long long m = A[0];
        #pragma omp parallel for reduction(min:m)
        for (long long i = 0; i < nLocal; i++) if (A[i] < m) m = A[i];
        return m;
    }

    void reportarLocal(const char* operacion, long long valor) {
        log->escribir(msg(omp_get_thread_num(), -1, operacion, num(valor)));
    }
    void reportar(const char* operacion, const std::string& valor) {
        log->escribir(msg(-1, -1, operacion, valor));
    }
    void reportarTiempo(const char* operacion, double t) {
        std::ostringstream s; s << std::fixed << std::setprecision(6) << t << " s";
        log->escribir(msg(-1, -1, operacion, s.str()));
    }

    // ---------- Impresion de la seccion local (solo primera ejecucion) ------
    void imprimirSeccion(const char* etiqueta, long long* arr) {
        if (!detalle) return;
        std::ostringstream s;
        s << "[Equipo: " << equipo << "] [Proceso MPI: " << rank << "] " << etiqueta << ": [ ";
        for (long long i = 0; i < nLocal; i++) { s << arr[i]; if (i < nLocal - 1) s << ", "; }
        s << " ]";
        log->escribir(s.str());
    }
};

// ============================================================================
// MAIN
// ============================================================================
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, totalProcs, len;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &totalProcs);
    char equipo[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(equipo, &len);

    // PRIMERA LINEA: nombres del equipo
    if (rank == 0) {
        std::cout << "=========================================================\n"
                  << " Practica 1.3: Reduce en MPI con OpenMP\n"
                  << " Integrantes: Cortes Corona Jonathan Manuel,\n"
                  << "              Lara Jimenez Pablo Cesar,\n"
                  << "              Martinez Delgado Miguel Angel\n"
                  << "=========================================================" << std::endl;
    }

    // Cada nodo abre su archivo de log local
    Registro log;
    log.abrir(equipo, rank);
    log.soloLog("=== Log del nodo " + std::string(equipo) + " proceso " +
                std::to_string(rank) + " ===");

    // Tamano global: argumento o 40 por omision
    long long N = (argc > 1) ? atoll(argv[1]) : 40;
    if (N % totalProcs != 0) N = (N / totalProcs + 1) * totalProcs; // divisible
    bool detalle = (N <= 1000);   // impresiones elemento a elemento solo si es chico

    if (rank == 0) {
        std::cout << "\n>>> Elementos totales: " << N
                  << " | Procesos MPI: " << totalProcs
                  << " | Elementos por proceso: " << N / totalProcs
                  << " | Hilos OpenMP: " << omp_get_max_threads()
                  << "\n>>> Modo detallado: " << (detalle ? "SI" : "NO (solo tiempos)")
                  << std::endl;
    }

    OperacionesArreglos ops(rank, totalProcs, equipo, &log);
    ops.setDetalle(detalle);

    int opcion = 0;
    do {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << "\n=========================================================\n"
                      << "        MENU - REDUCE EN MPI CON OPENMP (N = " << N << ")\n"
                      << "=========================================================\n"
                      << " 1. Crear arreglos\n"
                      << " 2. Sumar arreglos\n"
                      << " 3. Restar arreglos\n"
                      << " 4. Multiplicar arreglos\n"
                      << " 5. Calcular el cuadrado de un arreglo\n"
                      << " 6. Llenar secuencial\n"
                      << " 7. Llenar aleatorio\n"
                      << " 8. Sumatoria\n"
                      << " 9. Promedio\n"
                      << "10. Maximo\n"
                      << "11. Minimo\n"
                      << "12. Salir\n"
                      << "Seleccione una opcion: " << std::flush;
            if (!(std::cin >> opcion)) opcion = 12;   // EOF o entrada invalida -> salir
        }
        // El maestro difunde la opcion a todos los procesos
        MPI_Bcast(&opcion, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Las opciones 2 a 11 requieren arreglos creados
        if (opcion >= 2 && opcion <= 11 && !ops.listo()) {
            if (rank == 0) std::cout << "\n[X] Primero cree los arreglos (opcion 1).\n";
            continue;
        }

        double t0 = MPI_Wtime();

        switch (opcion) {
            case 1:
                ops.crearArregloMPI(N);
                break;

            case 2: case 3: case 4: case 5: {
                const char* nom = (opcion==2)?"SUMA":(opcion==3)?"RESTA":
                                  (opcion==4)?"MULTIPLICACION":"CUADRADO";
                ops.operacionElemento(opcion - 1);
                ops.imprimirSeccion("Resultado C", ops.ptrC());

                // --- Recoleccion de resultados: version Scatter/Gather ------
                long long* total = nullptr;
                if (rank == 0) total = new long long[N];
                double g0 = MPI_Wtime();
                MPI_Gather(ops.ptrC(), ops.tamLocal(), MPI_LONG_LONG,
                           total, ops.tamLocal(), MPI_LONG_LONG, 0, MPI_COMM_WORLD);
                double g1 = MPI_Wtime();

                // --- Recoleccion de resultados: version punto a punto -------
                double s0 = MPI_Wtime();
                if (rank == 0) {
                    long long* buf = new long long[ops.tamLocal()];
                    for (int p = 1; p < totalProcs; p++)
                        MPI_Recv(buf, ops.tamLocal(), MPI_LONG_LONG, p, 100,
                                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    delete[] buf;
                } else {
                    MPI_Send(ops.ptrC(), ops.tamLocal(), MPI_LONG_LONG, 0, 100,
                             MPI_COMM_WORLD);
                }
                double s1 = MPI_Wtime();

                if (rank == 0) {
                    std::cout << "\n--- " << nom << " (global) ---" << std::endl;
                    if (detalle) {
                        std::cout << "Arreglo resultado C: [ ";
                        for (long long i = 0; i < N; i++) {
                            std::cout << total[i]; if (i < N-1) std::cout << ", ";
                        }
                        std::cout << " ]" << std::endl;
                    }
                    std::cout << std::fixed << std::setprecision(6)
                              << "Recoleccion Scatter/Gather : " << (g1-g0) << " s\n"
                              << "Recoleccion Send/Recv      : " << (s1-s0) << " s"
                              << std::endl;
                    delete[] total;
                }
                log.soloLog("Gather=" + std::to_string(g1-g0) +
                            "s SendRecv=" + std::to_string(s1-s0) + "s");
                break;
            }

            case 6: ops.llenarSecuencial(); ops.imprimirSeccion("Arreglo A", ops.ptrA()); break;
            case 7: ops.llenarAleatorio();  ops.imprimirSeccion("Arreglo A", ops.ptrA()); break;

            // ---------- Reducciones: Nivel 1 OpenMP + Nivel 2 MPI ----------
            case 8: case 9: case 10: case 11: {
                long long local = 0, globalRed = 0, globalAll = 0;
                MPI_Op operador = MPI_SUM;
                const char* nom = "SUMATORIA";

                if (opcion == 8 || opcion == 9) {
                    local = ops.sumaLocal(); operador = MPI_SUM;
                    nom = (opcion == 8) ? "SUMATORIA" : "PROMEDIO";
                } else if (opcion == 10) {
                    local = ops.maxLocal(); operador = MPI_MAX; nom = "MAXIMO";
                } else {
                    local = ops.minLocal(); operador = MPI_MIN; nom = "MINIMO";
                }
                ops.reportarLocal((std::string(nom) + " Local").c_str(), local);

                // Variante A: MPI_Reduce -> el resultado solo llega al nodo 0
                double r0 = MPI_Wtime();
                MPI_Reduce(&local, &globalRed, 1, MPI_LONG_LONG, operador, 0, MPI_COMM_WORLD);
                double r1 = MPI_Wtime();

                // Variante B: MPI_Allreduce -> el resultado llega a TODOS
                double a0 = MPI_Wtime();
                MPI_Allreduce(&local, &globalAll, 1, MPI_LONG_LONG, operador, MPI_COMM_WORLD);
                double a1 = MPI_Wtime();

                // Todos los nodos conocen el resultado gracias a Allreduce
                if (opcion == 9) {
                    double prom = (double)globalAll / (double)N;
                    std::ostringstream s; s << std::fixed << std::setprecision(4) << prom;
                    ops.reportar("PROMEDIO global (Allreduce)", s.str());
                } else {
                    ops.reportar((std::string(nom) + " global (Allreduce)").c_str(),
                                 std::to_string(globalAll));
                }

                if (rank == 0) {
                    std::cout << "\n--- " << nom << " GLOBAL ---" << std::endl;
                    if (opcion == 9)
                        std::cout << "MPI_Reduce    (solo nodo 0): "
                                  << std::fixed << std::setprecision(4)
                                  << (double)globalRed / (double)N << std::endl;
                    else
                        std::cout << "MPI_Reduce    (solo nodo 0): " << globalRed << std::endl;
                    std::cout << std::fixed << std::setprecision(6)
                              << "Tiempo MPI_Reduce    : " << (r1-r0) << " s\n"
                              << "Tiempo MPI_Allreduce : " << (a1-a0) << " s" << std::endl;
                }
                break;
            }

            case 12:
                if (rank == 0) std::cout << "\nSaliendo del programa..." << std::endl;
                break;

            default:
                if (rank == 0) std::cout << "\n[X] Opcion invalida (1-12)." << std::endl;
                break;
        }

        if (opcion >= 1 && opcion <= 11) {
            double t1 = MPI_Wtime();
            ops.reportarTiempo("Tiempo total de la operacion", t1 - t0);
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank == 0)
                std::cout << "Tiempo total (MPI_Wtime): " << std::fixed
                          << std::setprecision(6) << (t1 - t0) << " s" << std::endl;
        }

    } while (opcion != 12);

    ops.liberar();

    // Cada nodo cierra su log de forma segura ANTES de MPI_Finalize
    log.soloLog("=== Fin del log del nodo " + std::to_string(rank) + " ===");
    std::string archivo = log.nombre();
    log.cerrar();

    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "[Equipo: " << equipo << "] [Proceso MPI: " << rank
              << "] Log guardado en: " << archivo << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);

    // ULTIMA LINEA: nombres del equipo
    if (rank == 0) {
        std::cout << "\n=========================================================\n"
                  << " Fin - Integrantes: Cortes Corona Jonathan Manuel,\n"
                  << " Lara Jimenez Pablo Cesar, Martinez Delgado Miguel Angel\n"
                  << "=========================================================" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
