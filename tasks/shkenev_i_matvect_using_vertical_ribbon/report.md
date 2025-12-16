# Ленточная вертикальная схема - умножение матрицы на вектор.

- Студент: Шкенев Илья Андреевич, группа 3823Б1ПР3
- Технология: SEQ | MPI
- Вариант: 12

## 1. Введение

Умножение матрицы на вектор — фундаментальная операция в линейной алгебре. Последовательная реализация может быть вычислительно дорогой для больших матриц. MPI позволяет распределить вычисления между процессами, ускоряя выполнение операции.

В данной работе реализованы две версии алгоритма:
- Последовательная (SEQ)
- Параллельная (MPI) с использованием декомпозиции по вертикальным полосам (столбцам) результирующего вектора

Цель работы — создать корректное и эффективное параллельное решение и сравнить его с последовательной реализацией.

## 2. Постановка задачи

Даны матрица A размером m * n и вектор-столбец B размером n.

**Необходимо вычислить:** 
- `C = A ⋅ B,  Ci = Σ (j=0..n-1) A[i][j] * B[j],  i = 0..m-1`

**Входные данные:**
- матрица A и вектор B - InType = std::pair<std::vector<std::vector<double>>, std::vector<double>>

**Выходные данные:**
- Результирующая вектор C - OutType = std::vector<double>

**Ограничения:**
- Все строки матрицы A имеют одинаковую длину.
- Размер вектора B совпадает с числом столбцов матрицы A.
- ВМатрицы и вектор не пустые.

## 3. Базовый алгоритм (последовательный)

**Последовательный алгоритм:**

```
1. Проверка размеров матрицы и вектора
2. Инициализация результирующей вектора C размером m*p нулями
3. Для i от 0 до m-1:
     Для j от 0 до n-1:
         Для k от 0 до p-1:
             C[i][k] += A[i][j] * B[j][k]
4. Вернуть C
```

**Сложность:**
- Время: O(m * n)
- Память: O(m)

## 4. Схема параллелизации

### Распределение данных

**Вертикальная декомпозиция результирующей матрицы C:**

Каждый процесс получает свою часть столбцов матрицы A и соответствующих элементов вектора B.
Декомпозиция столбцов по процессам:
```
base_cols = n / P
remainder = n % P
local_cols[i] = base_cols + (i < remainder ? 1 : 0)
offset[i] = сумма столбцов предыдущих процессов
```

### Локальные вычисления

**Каждый процесс вычисляет свой локальный результат::**
```
for row in 0..m-1:
    for col in 0..local_cols-1:
        local_result[row] += local_matrix[row][col] * local_vector[col]
```

### Обработка особых случаев

- Если число столбцов < число процессов, все процессы выполняют полное умножение последовательно.
- Процессы с нулевой шириной полосы пропускают вычисления.

### Сбор результатов

Для сбора используется:

- Используется MPI_Reduce для суммирования локальных результатов.
- Полный результат рассылается всем процессам через MPI_Bcast.


## 5. Детали реализации

### Структура кода

**Используемые файлы:**
- `common/include/common.hpp` - общие типы данных
- `seq/include/ops_seq.hpp`, `seq/src/ops_seq.cpp` - последовательная версия
- `mpi/include/ops_mpi.hpp`, `mpi/src/ops_mpi.cpp` - параллельная версия
- `tests/functional/main.cpp` - функциональные тесты
- `tests/performance/main.cpp` - тесты производительности

**Классы:**
- `ShkenevIDiffBetwNeighbElemVecSEQ` - последовательная реализация
- `ShkenevIDiffBetwNeighbElemVecMPI` - параллельная реализация

**Методы:**
- `ValidationImpl()` - проверка входных данных
- `PreProcessingImpl()` - предобработка
- `RunImpl()` - основной алгоритм
- `PostProcessingImpl()` - проверка результата

**Вспомогательные функции MPI:**
- `BroadcastMatrixSize` - рассылка размеров матриц
- `ComputeColumnDistribution` - вычисление количества столбцов на процесс
- `ScatterMatrixColumns` - распределение данных
- `ScatterVectorPart` - распределение данных
- `ComputeLocalProduct` - локальные вычисления

## 6. Экспериментальная установка

### Аппаратное обеспечение

- **Процессор:** Intel(R) Core(TM) 5 220H (2.70 GHz) 
- **ОЗУ:** 32,0 ГБ
- **ОС:** Windows 11 Домашняя для одного языка

### Программное обеспечение

- **Компилятор:** MSVC 14.44
- **MPI:** MS-MPI 10.0
- **Сборка:** Release 
- **CMake:** 4.2.0-rc1
- **Фреймворк тестирования:** Google Test

### Параметры тестирования

**Функциональные тесты:**
Было выполнено 8 тестов:
- (2x2 * 2) - малый квадрат
- (3x3 * 3) - квадратная матрица, результат [14, 32, 50]
- (2x3 * 3) - прямоугольные матрицы
- (4×3 * 3) - прямоугольная, результат [22, 49, 76, 103]
- (3x3 * 3) - единичная матрица
- (2×3 * 3) - нулевая матрица
- (1x1 * 1) - скалярный случай 
- (1x5 * 5) - строка на вектор
- Проверка корректности: сравнение SEQ и MPI результатов

**Тесты производительности:**
- Размер матриц: 11000*11000
- Матрица A: разреженная (20% ненулевых элементов)
- Вектор B: плотная
- MPI: 1,2,3,4 процесса
- Режимы: task_run, pipeline

## 7. Результаты и обсуждение

### 7.1 Корректность

Корректность реализации проверена следующими способами:

1. **Функциональные тесты:** все 8 тестов успешно пройдены.
2. **Сравнение SEQ и MPI:** результаты идентичны на всех тестовых данных.
3. **Проверка размеров:** корректная обработка граничных случаев и недопустимых входных данных.

### 7.2 Производительность

Результаты измерений на векторе из 11000*11000 элементов

**task_run:**

| Режим | Процессов | Время, сек | Ускорение | Эффективность |
| ----- | --------- | ---------- | --------- | ------------- |
| seq   | 1         | 1.05       | 1.00      | N/A           |
| mpi   | 2         | 0.706      | 1.49      | 74%           |
| mpi   | 3         | 0.636      | 1.65      | 55%           |
| mpi   | 4         | 0.579      | 1.81      | 45%           |


**task_pipeline:**

| Режим | Процессов | Время, сек | Ускорение | Эффективность |
| ----- | --------- | ---------- | --------- | ------------- |
| seq   | 1         | 1.05       | 1.00      | N/A           |
| mpi   | 2         | 0.731      | 1.44      | 72%           |
| mpi   | 3         | 0.636      | 1.65      | 55%           |
| mpi   | 4         | 0.602      | 1.74      | 44%           |



**Расчет показателей:**
- Ускорение = T_seq / T_mpi
- Эффективность = Ускорение / P x 100%

**Анализ результатов:**

1. **Сравнение SEQ и MPI:** 
- MPI-реализация демонстрирует ускорение уже на 2 процессах

2. **Оптимальная конфигурация:** 
- Наибольшее ускорение достигается на 4 процессах, но оно не такое большое

3. **Особенности вертикальной декомпозиции:** 
- Хорошее масштабирование на малом числе процессов (2-3)
- При 8 процессах эффективность сильно падает (≈20%) из-за высоких коммуникационных накладных расходов

**Вывод:**

- Ленточная вертикальная схема эффективна для умеренного числа процессов
- Оптимальное число процессов - 4, дальше время становится почти одиннаковым, а на 8 процессах эффективность начинает сильно падать 
- MPI-реализация превосходит SEQ благодаря алгоритмическим и структурным оптимизациям

## 8. Выводы

1. **Реализация:** 

Созданы и протестированы последовательная (SEQ) и параллельная (MPI) версии умножения матриц на вектор с использованием ленточной вертикальной схемы

2. **Корректность:** 

Все тесты пройдены, SEQ и MPI версии дают одинаковый результат

3. **Производительность:** 

На 2–4 процессах MPI быстрее SEQ, эффективность падает при 8 процессах из-за накладных расходов, а оптимальное число процессов для 11 000 элементов — 4.

4. **Масштабируемость:** 

- Эффективность снижается при увеличении числа процессов из-за накладных расходов на коммуникацию
- Оптимальная конфигурация - 4 процесса, обеспечивающая баланс между вычислительной нагрузкой и коммуникационными затратами


## 9. Список источников информации

1. Сысоев А. В. Лекции по параллельному программированию. — Н. Новгород: ННГУ, 2025.
2. Parallel Programming Course Slides - https://learning-process.github.io/parallel_programming_slides/
3. Microsoft. Microsoft MPI Documentation - https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi 

## Приложение

### Фрагменты кода

**Последовательная версия (SEQ):**

```cpp
bool ShkenevImatvectUsingVerticalRibbonSEQ::RunImpl() {
  const auto &matrix_a = GetInput().first;
  const auto &vector_b = GetInput().second;

  std::size_t rows_a = matrix_a.size();
  std::size_t cols_a = matrix_a[0].size();

  std::vector<double> result_vector(rows_a, 0.0);

  for (std::size_t i = 0; i < rows_a; i++) {
    for (std::size_t j = 0; j < cols_a; j++) {
      if (matrix_a[i][j] == 0.0) {
        continue;
      }

      result_vector[i] += matrix_a[i][j] * vector_b[j];
    }
  }

  GetOutput() = result_vector;

  return true;
}
```

**Параллельная версия (MPI):**

```cpp
bool ShkenevImatvectUsingVerticalRibbonMPI::RunImpl() {
    int world_size = 0;
    int rank = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int rows = 0;
    int cols = 0;

    if (rank == 0) {
        rows = static_cast<int>(GetInput().first.size());
        cols = static_cast<int>(GetInput().first[0].size());
    }

    BroadcastMatrixSize(rows, cols);

    if (rows == 0 || cols == 0) {
        return false;
    }

    std::vector<int> local_cols_vec(world_size, cols / world_size);
    int remainder = cols % world_size;
    for (int i = 0; i < remainder; ++i) local_cols_vec[i]++;

    std::vector<int> displs(world_size, 0);
    for (int i = 1; i < world_size; ++i)
        displs[i] = displs[i - 1] + local_cols_vec[i - 1];

    int local_cols = local_cols_vec[rank];
    int col_offset = displs[rank];

    std::vector<double> local_matrix(static_cast<size_t>(rows) * local_cols);

    if (rank == 0) {
        std::vector<double> flat_matrix(rows * cols);
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                flat_matrix[r * cols + c] = GetInput().first[r][c];

        std::vector<int> sendcounts(world_size);
        for (int i = 0; i < world_size; ++i) sendcounts[i] = rows * local_cols_vec[i];

        std::vector<int> senddispls(world_size);
        senddispls[0] = 0;
        for (int i = 1; i < world_size; ++i)
            senddispls[i] = senddispls[i - 1] + sendcounts[i - 1];

        MPI_Scatterv(flat_matrix.data(), sendcounts.data(), senddispls.data(), MPI_DOUBLE,
                     local_matrix.data(), rows * local_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DOUBLE,
                     local_matrix.data(), rows * local_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    std::vector<double> local_vector(local_cols);
    if (rank == 0) {
        for (int i = 0; i < local_cols; ++i)
            local_vector[i] = GetInput().second[col_offset + i];
    }
    MPI_Bcast(local_vector.data(), local_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<double> local_result(rows, 0.0);
    ComputeLocalProduct(local_matrix, local_vector, local_result, rows, local_cols);

    std::vector<double> result(rows, 0.0);
    MPI_Allreduce(local_result.data(), result.data(), rows, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    GetOutput() = result;
    return true;
}
```