# Нахождение наиболее отличающихся по значению соседних элементов вектора

- Студент: Шкенев Илья Андреевич, группа 3823Б1ПР3
- Технология: SEQ | MPI
- Вариант: 8

## 1. Введение

Задача нахождения максимальной разности между соседними элементами вектора является важной при обработке данных и вычислительной математике. Последовательная реализация выполняет линейный проход по массиву, что оптимально по сложности, но при обработке больших данных полезно применять параллельные вычисления. Технология MPI позволяет распределить данные между несколькими процессами и вычислить локальные разности, а затем объединить результаты в один глобальный максимум.

В данной работе реализованы две версии алгоритма:
- Последовательная (SEQ)
- Параллельная (MPI)

Цель работы - реализовать корректное и эффективное параллельное решение и сравнить ускорение.

## 2. Постановка задачи

Дан вектор целых чисел vec[0..N-1].

**Необходимо вычислить:** 
- `max |vec[i+1| - vec[i]|, где 0<=i<N-1`

**Входные данные:**
- Вектор целых чисел InType = std::vector<int>

**Выходные данные:**
- OutType = int - максимальная разность между соседними элементами

**Ограничения:**
- Если N < 2, то результат 0
- Данные могут быть любыми

## 3. Базовый алгоритм (последовательный)

**Последовательный алгоритм:**

```
1. Если N < 2, то вернуть 0
2. max_diff = 0
3. Для i от 0 до N-2:
     diff = abs(vec[i+1] - vec[i])
     если diff > max_diff, то обновить max_diff
4. Вернуть max_diff
```

**Сложность:**
- Время: O(N)
- Память: O(1)

## 4. Схема параллелизации

### Распределение данных

**Вектор делится на блоки для каждого процесса:**
```
base_size = N / P
remainder = N % P
cnt[i] = base_size + (i < remainder ? 1 : 0)
disp[i] = смещение начала блока процесса i
```
Процесс 0 копирует данные в свой локальный блок, остальные получают блок через MPI_Recv

### Локальные вычисления

Каждый процесс вычисляет максимум разности внутри своего блока с помощью функции LocalCompute(l_vec)

### Граничные элементы

**Чтобы корректно учесть пары, пересекающие границы блоков, используется BoundaryExchange(l_vec, world_rank, world_size):**
- Процесс i > 0 получает последний элемент процесса i-1
- Вычисляется abs(first_of_block - last_of_prev_block)
- Процесс i < world_size-1 отправляет свой последний элемент процессу i+1

### Сбор результатов

Для сбора используется:
```
MPI_Reduce(l_max, global_max, MPI_MAX) - Локальные максимумы
MPI_Bcast(global_max) - Результат рассылается всем процессам
```

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
- `HandleSmallVector` - обработка коротких векторов
- `ComputeCountsAndDispls` - расчет cnt и disp
- `ScatterData` - раздача данных
- `LocalCompute` - локальный максимум
- `BoundaryExchange` - обработка границ блоков

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
Было выполнено 10 тестов:
- {1,5} - 4 (результат)
- {8, 10, 2, 4} - 8 (результат)
- {1, 2, 3, 4, 5} - 1 (результат)
- {100, 5, 8, 3, 4} - 95 (результат)
- {-1, -5, 10, 8, -7} - 15 (результат)
- {1, 1, 1, 1, 1} - 0 (результат)
- {0, 0, 5, 0, 0, 0} - 5 (результат)
- {100, 50, 75, 25} - 50 (результат)
- {5} - 0 (результат)
- {} - 0 (результат)
- Проверка корректности: сравнение SEQ и MPI результатов

**Тесты производительности:**
- N = 500 000 000 случайных чисел
- Для создания вектора используется генератор случайных чисел:
```
std::uniform_int_distribution<int> small_dist(0, 100);
std::uniform_int_distribution<int> large_dist(1000, 10000);
```
Элементы вектора чередуются:
Чётные позиции (i % 2 == 0) - значения из small_dist
Нечётные позиции (i % 2 == 1) - значения из large_dist

Такой подход создаёт вектор с чередующимися малыми и большими числами, обеспечивая нагрузку на алгоритм и проверку корректности вычисления максимальной разницы соседних элементов

- MPI: 1,2,3,4 процесса
- Режимы: task_run, pipeline

## 7. Результаты и обсуждение

### 7.1 Корректность

Корректность реализации проверена следующими способами:

1. **Функциональные тесты:** Все 10 тестов успешно пройдены 
2. **Сравнение SEQ и MPI:** Результаты идентичны на всех тестовых данных
3. **Тесты производительности:** Для вектора из 500 000 000 элементов с чередующимися малыми (0–100) и большими (1000–10000) значениями проверено, что вычисленный максимальный разрыв соседних элементов соответствует ожидаемому
3. **Проверка инвариантов:** Результат всегда находится в корректном диапазоне, соответствует ожидаемой логике алгоритма

### 7.2 Производительность

Результаты измерений на векторе из 500 000 000 элементов.

**task_pipeline:**

| Режим | Процессов | Время, сек | Ускорение | Эффективность |
| ----- | --------- | ---------- | --------- | ------------- |
| seq   | 1         | 0.467      | 1.00      | N/A           |
| mpi   | 1         | 1.209      | 0.39      | 39%           |
| mpi   | 2         | 0.969      | 0.48      | 24%           |
| mpi   | 3         | 0.828      | 0.56      | 19%           |
| mpi   | 4         | 0.937      | 0.50      | 12%           |


**task_run:**

| Режим | Процессов | Время, сек | Ускорение | Эффективность |
| ----- | --------- | ---------- | --------- | ------------- |
| seq   | 1         | 0.469      | 1.00      | N/A           |
| mpi   | 1         | 1.446      | 0.32      | 32%           |
| mpi   | 2         | 0.934      | 0.50      | 25%           |
| mpi   | 3         | 0.875      | 0.54      | 18%           |
| mpi   | 4         | 0.735      | 0.64      | 16%           |


**Расчет показателей:**
- Ускорение = T_seq / T_mpi
- Эффективность = Ускорение / P x 100%

**Анализ результатов:**

1. **1 процесс MPI:** 
- В обоих режимах время значительно больше SEQ (ускорение < 1.0x, эффективность очень низкая)
- Это объясняется накладными расходами MPI (инициализация, коммуникации), которые сильно превышают полезную работу

2. **2 процесса:** 
- Незначительное улучшение времени, эффективность остаётся низкой
- Распределение данных ещё не компенсирует накладные расходы

3. **3 процесса:** 
- pipeline: небольшое ускорение, task_run: умеренное ускорение
- Масштабируемость всё ещё ограничена

4. **4 процесса:** 
- Некоторая выгода в task_run, pipeline остаётся нестабильным
- Общая эффективность MPI низкая из-за большой нагрузки на коммуникации

**Особенности:**

- SEQ значительно быстрее MPI из-за высокой нагрузки на коммуникации на каждом процессе
- task_run показывает более стабильное время при увеличении числа процессов, чем pipeline
- Масштабируемость ограничена. Для существенного ускорения требуется ещё больше данных или работы на элемент

**Вывод:**

- Для больших векторов (500 млн элементов) MPI пока не даёт значительного ускорения в нашем тесте
- Task_run показывает небольшое преимущество при 4 процессах
- Эффективность параллельной реализации ограничена накладными расходами на коммуникации

## 8. Выводы

1. **Реализация:** 

Созданы и протестированы последовательная (SEQ) и параллельная (MPI) версии поиска максимальной разницы между соседними элементами вектора

2. **Корректность:** 

Все тесты пройдены, SEQ и MPI версии дают одинаковый результат

3. **Производительность:** 

На векторе из 500 млн элементов SEQ версия остаётся быстрее MPI на малом числе процессов из-за накладных расходов на инициализацию и коммуникации MPI.
Task_run показывает более стабильное время по сравнению с pipeline, но ускорение MPI ограничено:
- 1 процесс MPI значительно медленнее SEQ (ускорение ~0.3–0.4×)
- 4 процесса MPI дают умеренное ускорение (task_run: ~0.64×), pipeline остаётся нестабильным

4. **Масштабируемость:** 

Эффективность MPI низкая, добавление процессов даёт небольшое улучшение для task_run, pipeline демонстрирует нестабильное поведение при росте числа процессов. Масштабируемость ограничена накладными расходами на коммуникации и синхронизацию

## 9. Список источников информации

1. Сысоев А. В. Лекции по параллельному программированию. — Н. Новгород: ННГУ, 2025.
2. Parallel Programming Course Slides - https://learning-process.github.io/parallel_programming_slides/
3. Microsoft. Microsoft MPI Documentation - https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi 

## Приложение

### Фрагменты кода

**Последовательная версия (SEQ):**

```cpp
bool ShkenevIDiffBetwNeighbElemVecSEQ::RunImpl() {
  const std::vector<int> &vec = GetInput();
  std::size_t n = vec.size();
  if (n < 2) {
    GetOutput() = 0;
    return true;
  }

  int max_diff = 0;
  for (std::size_t i = 0; i < n - 1; i++) {
    int diff = std::abs(vec[i + 1] - vec[i]);
    max_diff = std::max(diff, max_diff);
  }

  GetOutput() = max_diff;
  return true;
}
```

**Параллельная версия (MPI):**

```cpp
namespace {

int HandleSmallVector(const std::vector<int> &vec, int n) {
  int result = 0;
  for (int i = 0; i < n - 1; ++i) {
    int diff = std::abs(vec[i + 1] - vec[i]);
    result = std::max(result, diff);
  }
  return result;
}

void ComputeCountsAndDispls(int n, int world_size, std::vector<int> &cnt, std::vector<int> &disp) {
  int base_size = n / world_size;
  int rem = n % world_size;

  int shift = 0;
  for (int i = 0; i < world_size; ++i) {
    cnt[i] = base_size + (i < rem ? 1 : 0);
    disp[i] = shift;
    shift += cnt[i];
  }
}

void ScatterData(const std::vector<int> &vec, const std::vector<int> &cnt, const std::vector<int> &disp,
                 std::vector<int> &l_vec, int world_rank) {
  size_t l_n = cnt[world_rank];

  if (world_rank == 0) {
    if (l_n > 0) {
      for (size_t i = 0; i < l_n; ++i) {
        l_vec[i] = vec[i];
      }
    }

    for (size_t proc = 1; proc < cnt.size(); ++proc) {
      if (cnt[proc] > 0) {
        MPI_Send(vec.data() + disp[proc], cnt[proc], MPI_INT, static_cast<int>(proc), 0, MPI_COMM_WORLD);
      }
    }
  } else {
    if (l_n > 0) {
      MPI_Recv(l_vec.data(), static_cast<int>(l_n), MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
}

int LocalCompute(const std::vector<int> &l_vec) {
  int l_max = 0;
  int l_n = static_cast<int>(l_vec.size());

  for (int i = 0; i < l_n - 1; ++i) {
    int diff = std::abs(l_vec[i + 1] - l_vec[i]);
    l_max = std::max(l_max, diff);
  }

  return l_max;
}

int BoundaryExchange(const std::vector<int> &l_vec, int world_rank, int world_size) {
  int l_n = static_cast<int>(l_vec.size());
  int boundary = 0;

  if (world_rank > 0 && l_n > 0) {
    int prev_last = 0;
    MPI_Recv(&prev_last, 1, MPI_INT, world_rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    boundary = std::abs(l_vec[0] - prev_last);
  }

  if (world_rank < world_size - 1 && l_n > 0) {
    int my_last = l_vec[l_n - 1];
    MPI_Send(&my_last, 1, MPI_INT, world_rank + 1, 1, MPI_COMM_WORLD);
  }

  return boundary;
}
}  // namespace

ShkenevIDiffBetwNeighbElemVecMPI::ShkenevIDiffBetwNeighbElemVecMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool ShkenevIDiffBetwNeighbElemVecMPI::ValidationImpl() {
  return true;
}

bool ShkenevIDiffBetwNeighbElemVecMPI::PreProcessingImpl() {
  return true;
}

bool ShkenevIDiffBetwNeighbElemVecMPI::RunImpl() {
  int world_rank = 0;
  int world_size = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const std::vector<int> &vec = GetInput();
  int n = static_cast<int>(vec.size());

  if (n < 2) {
    GetOutput() = 0;
    return true;
  }

  if (world_size > n) {
    int result = (world_rank == 0 ? HandleSmallVector(vec, n) : 0);
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    GetOutput() = result;
    return true;
  }

  std::vector<int> cnt(world_size);
  std::vector<int> disp(world_size);
  ComputeCountsAndDispls(n, world_size, cnt, disp);

  std::vector<int> l_vec(cnt[world_rank]);
  ScatterData(vec, cnt, disp, l_vec, world_rank);

  int local_max = LocalCompute(l_vec);
  int boundary = BoundaryExchange(l_vec, world_rank, world_size);
  local_max = std::max(local_max, boundary);

  int global_max = 0;
  MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Bcast(&global_max, 1, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = global_max;
  return true;
}
```