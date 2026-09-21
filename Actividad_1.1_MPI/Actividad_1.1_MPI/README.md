# Modelo de programación y memoria en MPI (MPI + OpenMP híbrido)

Programa que utiliza una clase para llenar un arreglo dinámico con valores aleatorios,
mostrando el porcentaje de avance de 10% en 10%. Usa **MPI** para distribuir la ejecución
entre procesos/nodos y **OpenMP** para paralelizar el llenado dentro de cada nodo.

## Equipo 5 - Integrantes (orden alfabético por primer apellido)
- Cortes Corona Jonathan Manuel
- Lara Jiménez Pablo César
- Martínez Delgado Miguel Ángel

## Contenido del repositorio
| Archivo | Descripción |
|---|---|
| `main.cpp` | Código fuente completo (clase, MPI, OpenMP, arreglo dinámico). |
| `README.md` | Este archivo: instrucciones de compilación y ejecución. |
| `pruebas/` | Capturas de la primera ejecución (20 a 50 elementos) y de la segunda ejecución (10,000,000 de elementos), ambas con 4 procesos MPI. |

## Implementación

- **MPI:** `MPI_Init`, `MPI_Comm_rank` (identificador del proceso), `MPI_Comm_size` (total de
  procesos), `MPI_Get_processor_name` (nombre del equipo/nodo), `MPI_Bcast` (compartir el
  tamaño sorteado), `MPI_Barrier` (ordenar la salida), `MPI_Wtime` (medición de tiempo) y
  `MPI_Finalize`.
- **OpenMP:** `#pragma omp parallel` + `#pragma omp for` para repartir el llenado entre los
  núcleos del nodo, `#pragma omp atomic capture` para llevar el conteo real de avance sin
  condiciones de carrera, y `#pragma omp critical` para que las líneas de avance no se
  mezclen entre hilos.
- **Clase `GeneradorArreglo`:** encapsula el arreglo dinámico; lo reserva con `new[]` en el
  constructor y lo libera con `delete[]` en el destructor.
- **Arreglos dinámicos:** se usa exclusivamente `new int[n]` / `delete[]`. No se utiliza
  `vector`, `std::vector`, `ArrayList` ni ninguna estructura equivalente.
- **Valores aleatorios:** se usa `srand()` y `rand()` para obtener una semilla base distinta
  por proceso MPI; cada hilo deriva de ella su propio estado independiente, ya que el estado
  global de `rand()` no es seguro cuando varios hilos lo llaman simultáneamente.

## Compilación

### Linux
```bash
mpic++ -fopenmp -O2 main.cpp -o Hola_MPI
```

### Windows (MS-MPI + Visual Studio)
1. Instalar **MS-MPI** (Runtime + SDK).
2. Crear un proyecto de consola C++ y agregar `main.cpp`.
3. En *Propiedades del proyecto*:
   - **C/C++ > General > Directorios de inclusión adicionales:** `C:\Program Files (x86)\Microsoft SDKs\MPI\Include`
   - **Vinculador > General > Directorios de bibliotecas adicionales:** `C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64`
   - **Vinculador > Entrada > Dependencias adicionales:** agregar `msmpi.lib`
   - **C/C++ > Lenguaje > Compatibilidad con OpenMP:** `Sí (/openmp)`
4. Compilar en configuración **x64**.

## Ejecución

### Primera ejecución — tamaño aleatorio entre 20 y 50 elementos
Sin argumentos, el programa sortea un tamaño entre 20 y 50:
```bash
mpiexec -n 4 Hola_MPI.exe
```

### Segunda ejecución — 10,000,000 de elementos
El tamaño se pasa como argumento:
```bash
mpiexec -n 4 Hola_MPI.exe 10000000
```

### Tercera ejecución — distribuida en red (al menos 3 computadoras)
```bash
mpiexec -n 6 -hosts 3 NODO1 NODO2 NODO3 \\RUTA_COMPARTIDA\Hola_MPI.exe 10000000
```

### Controlar el número de hilos OpenMP
Por defecto se usan todos los hilos disponibles del nodo. Para fijarlos manualmente:
```bash
# Linux
OMP_NUM_THREADS=4 mpiexec -n 4 Hola_MPI.exe 10000000
# Windows
set OMP_NUM_THREADS=4
mpiexec -n 4 Hola_MPI.exe 10000000
```

## Salida esperada
Cada línea de avance identifica el equipo, el proceso MPI y el hilo OpenMP:
```
[Equipo: NODO1 | Proceso MPI: 0/4 | Hilo OpenMP: 2] Avance del llenado: 30%
```
Cada proceso muestra la secuencia completa 10%, 20%, ... 100%, y al terminar imprime el
contenido de su arreglo (completo si es pequeño, o una muestra si tiene millones de elementos).
