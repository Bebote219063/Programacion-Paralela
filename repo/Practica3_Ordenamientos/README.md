# Actividad 2.5: Ordenamientos (iterativo y recursivo con OpenMP)

## Integrantes (orden alfabético por primer apellido)
- Cortes Corona Jonathan Manuel
- Lara Jiménez Pablo César
- Martínez Delgado Miguel Ángel

## Algoritmos seleccionados
- **Iterativo: Shell Sort.** Se eligió en lugar de Burbuja/Selección/Inserción porque estas
  últimas son O(n²): con 10,000,000 de elementos tardarían horas en terminar. Shell Sort es
  mucho más eficiente y se paraleliza de forma natural dividiendo el arreglo en columnas
  independientes para cada valor de "salto".
- **Recursivo: Merge Sort.** Se paraleliza de forma natural con tareas de OpenMP: cada mitad
  del arreglo se ordena en una tarea independiente, y se combinan al final.

## Directivas OpenMP utilizadas
- `#pragma omp parallel for` — llenado del arreglo y ambas fases de Shell Sort (secuencial y
  paralela, en la paralela reparte las columnas independientes entre hilos).
- `thread_local` (C++) — semilla independiente por hilo para el llenado aleatorio, evitando
  secuencias repetidas entre hilos.
- Barrier implícito (al final de cada `#pragma omp parallel for`) — sincroniza entre un valor
  de "salto" y el siguiente en Shell Sort paralelo, ya que cada salto depende de que el
  anterior haya terminado por completo.
- `#pragma omp task` — cada mitad del arreglo en Merge Sort paralelo.
- `#pragma omp taskwait` — espera a que ambas mitades (tareas) terminen antes de mezclarlas.
- `#pragma omp parallel` + `#pragma omp single` — inicia el árbol de tareas de Merge Sort
  desde un único hilo, evitando que se dupliquen los árboles de tareas.
- `omp_get_wtime()` — medición de tiempos de llenado y de cada ordenamiento.

## Cómo compilar y ejecutar

```bash
g++ -fopenmp -std=c++17 ordenamientos.cpp -o ordenamientos
./ordenamientos
```

En Code::Blocks: agregar `-fopenmp` en *Build options > Compiler settings > Other options* Y
en *Build options > Linker settings > Other linker options* (aplicado al proyecto completo).

### Dos ejecuciones requeridas por la práctica

El tamaño del arreglo se controla con constantes al inicio del archivo:

```cpp
const int TAMANO    = 100;   // cambiar a 10000000 para la segunda ejecución
const int RANGO_MIN = 0;
const int RANGO_MAX = 200;   // cambiar a 2000000 para la segunda ejecución
```

- **Primera ejecución:** dejar `TAMANO = 100` y `RANGO_MAX = 200` (se imprime el arreglo).
- **Segunda ejecución:** cambiar a `TAMANO = 10000000` y `RANGO_MAX = 2000000`, y volver a
  compilar (no se imprime el arreglo, solo tiempos y speedup).

### Uso del menú

1. Llenar arreglo
2. Ordenamiento iterativo secuencial (Shell Sort)
3. Ordenamiento iterativo paralelo (Shell Sort)
4. Ordenamiento recursivo secuencial (Merge Sort)
5. Ordenamiento recursivo paralelo (Merge Sort con tareas)
6. Mostrar resultados y speedup
7. Salir
