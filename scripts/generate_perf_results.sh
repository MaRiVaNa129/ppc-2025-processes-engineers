#!/usr/bin/env bash
set -euo pipefail

# Создаём директорию для результатов
mkdir -p build/perf_stat_dir

# Запускаем тесты и сохраняем лог
scripts/run_tests.py --running-type="performance" | tee build/perf_stat_dir/perf_log.txt

# Создаём таблицу результатов
python3 scripts/create_perf_table.py --input build/perf_stat_dir/perf_log.txt --output build/perf_stat_dir
