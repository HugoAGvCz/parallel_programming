from mpi4py import MPI 
import random
import time

class InsertionSort:
    def __init__(self, array):
        self.array = array

    def insertion_sort(self):
        # Versión iterativa del ordenamiento por inserción
        for i in range(1, len(self.array)):
            key = self.array[i]
            j = i - 1
            # Mover los elementos mayores que la clave una posición adelante
            while j >= 0 and self.array[j] > key:
                self.array[j + 1] = self.array[j]
                j -= 1
            self.array[j + 1] = key
        return self.array

    def parallel_insertion_sort(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Dividir el arreglo en partes iguales para cada proceso
        if rank == 0:
            chunk_size = len(self.array) // size
            # Crear los chunks para cada proceso
            chunks = [self.array[i * chunk_size:(i + 1) * chunk_size] for i in range(size)]
            # Si hay elementos sobrantes, los añadimos al último chunk
            if len(self.array) % size != 0:
                chunks[-1].extend(self.array[size * chunk_size:])
        else:
            chunks = None

        # Distribuir los subarreglos a los diferentes procesos
        subarray = comm.scatter(chunks, root=0)

        # Cada proceso ordena su subarreglo localmente
        sorter = InsertionSort(subarray)
        sorted_subarray = sorter.insertion_sort()

        # Reunir los subarreglos ordenados en el proceso principal
        gathered_subarrays = comm.gather(sorted_subarray, root=0)

        if rank == 0:
            # Combinar los subarreglos ordenados en un solo arreglo
            combined = sum(gathered_subarrays, [])
            # Ordenar el arreglo completo nuevamente en el proceso principal
            final_sorter = InsertionSort(combined)
            final_sorted_array = final_sorter.insertion_sort()

            return final_sorted_array
        else:
            return None

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Crear el arreglo original en el proceso principal
    if rank == 0:
        # Modificar el número de elementos para probar con diferentes tamaños
        num_elements = 500 
        array = [random.randint(-100, 100) for _ in range(num_elements)] 
        print("Arreglo original:", array)

        # Medir el tiempo del ordenamiento secuencial
        start_time_seq = time.time()
        sequential_sorter = InsertionSort(array.copy())
        sequential_sorted_array = sequential_sorter.insertion_sort()
        end_time_seq = time.time()
        sequential_duration = end_time_seq - start_time_seq
        print("Resultado del ordenamiento por inserción secuencial:", sequential_sorted_array)
        print(f"Tiempo de ejecución secuencial: {sequential_duration:.6f} segundos")

        # Medir el tiempo del ordenamiento paralelo
        start_time_par = time.time()
        parallel_sorter = InsertionSort(array.copy())
        parallel_sorted_array = parallel_sorter.parallel_insertion_sort()
        end_time_par = time.time()
        parallel_duration = end_time_par - start_time_par
        print("Resultado del ordenamiento por inserción paralelo:", parallel_sorted_array)
        print(f"Tiempo de ejecución paralelo: {parallel_duration:.6f} segundos")

        # Calcular el speedup y la eficiencia
        if parallel_sorted_array is not None:
            speedup = sequential_duration / parallel_duration if parallel_duration > 0 else float('inf')
            efficiency = speedup / size

            # Mostrar los resultados del speedup y eficiencia
            print(f"Speedup: {speedup:.2f}")
            print(f"Eficiencia: {efficiency:.2f}")

            # Comparar los tiempos de ejecución
            print(f"Diferencia de tiempo: {sequential_duration - parallel_duration:.6f} segundos")
    else:
        array = None
        # Ejecutar el método paralelo en los otros procesos
        parallel_sorter = InsertionSort(array)
        parallel_sorted_array = parallel_sorter.parallel_insertion_sort()

if __name__ == "__main__":
    main()
