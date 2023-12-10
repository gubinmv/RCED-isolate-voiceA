# Исследование нейросетевых методов Forward и Backward для выделения полезного речевого сигнала из зашумленного звукового потока

## Описание датасета
Датасет с речевыми сигналами и шумами размещен по адресу https://github.com/gubinmv/dataset-voices-noise.git

## Описание файлов
param_project.py - модуль настройки

my_loss_functions.py - модуль функций ошибки

create_model_conv1d-15skip.py - модуль создания нейронной сети с 15 скрытыми слоям

create_model_conv1d-9skip.py - модуль создания нейронной сети с 9 скрытыми слоям

create_dataset.py - модуль создания обучающей и тестирующей выборок

start_train.py - модуль обучучения нейронной сети

test_Forward_Backward.py - модуль окончательного тестирования обученных нейронных сетей

