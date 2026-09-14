# Programación Paralela - OpenMP

Integrantes (orden alfabético por primer apellido):
- Cortes Corona Jonathan Manuel
- Lara Jiménez Pablo César
- Martínez Delgado Miguel Ángel

Este repositorio contiene las prácticas de OpenMP desarrolladas para el curso, cada una en
su propia carpeta.


## Contenido

| Carpeta | Práctica | Descripción |
|---|---|---|
| `Practica1_CondicionDeCarrera` | Condición de carrera en OpenMP | Cada hilo llena un arreglo local con valores únicos, mostrando su porcentaje de avance. |
| `Practica2_ReduceOpenMP` | Operaciones con arreglos y reduce | Clase `OperacionesArreglos` con sumatoria, promedio, máximo y mínimo usando `reduction`. |
| `Practica3_Ordenamientos` | Actividad 2.5 - Ordenamientos | Shell Sort (iterativo) y Merge Sort (recursivo), cada uno secuencial y paralelo con tareas. |

## Compilación general

Todas las prácticas requieren soporte de OpenMP:

```bash
g++ -fopenmp -std=c++17 archivo.cpp -o programa
```

En Code::Blocks: agregar `-fopenmp` tanto en *Compiler settings > Other options* como en
*Linker settings > Other linker options*, en las opciones de compilación (Build options) del proyecto.

Ver el README de cada carpeta para detalles específicos de cada práctica.
