Папка для задачи B: воспроизведение последней посылки A со score 90.67.

Главный файл прогноза:
- test.csv

Главный ноутбук:
- x5_stage2_final_solution_RU_FIXED.ipynb

Файлы, которые ноутбук использует как входные артефакты ансамбля:
- train_2.csv(не получилось загрузить из за размера файла)
- UPLOAD_9067_FINE_A014_A020_T0p75.csv
- UPLOAD_H2_RECENCY_WEIGHTED_REGION_TRANSFER_CANDIDATE.csv
- UPLOAD_9080_TRANSFER_BIAS_V1.csv
- UPLOAD_PLATEAU_MEDIAN_ENSEMBLE_V1.csv

Контроль воспроизводимости:
- seed: 2026
- итоговый md5 test.csv: 25232f8b8fd6171e3168d6d22e0c012d
- строк: 18657
- колонки: new_id,rto
- NaN: 0
- неположительных rto: 0

Важно: это exact-repro пакет для последнего подтвержденного public 90.67.
Он не является train-only решением, потому что финальный файл был собран из нескольких уже построенных прогнозов.
