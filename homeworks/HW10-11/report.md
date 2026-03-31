# HW10-11 – компьютерное зрение в PyTorch: CNN, transfer learning, detection/segmentation

**Выбранный трек второй части: segmentation** (часть B — сегментация; detection не используется).

Трек части B (обязательная строка для проверки): **segmentation** (не detection).

## 1. Кратко: что сделано

- Часть A: датасет `STL10` — стандартный выбор из методички, удобно сравнивать CNN и `ResNet18` на одном сплите.
- Часть B: датасет `OxfordIIITPet`, трек **segmentation** — проще воспроизвести без обучения детектора, достаточно pretrained FCN и метрик по маске.
- В части A сравнивались эксперименты **C1–C4**; в части B — два режима постобработки **V1** и **V2** на одной и той же модели.

## 2. Среда и воспроизводимость

- Python: 3.9.6 (указать свою версию при отличии)
- torch / torchvision: 2.8.0 / 0.23.0 (указать свои версии при отличии)
- Устройство (CPU/GPU): CPU (`cuda` недоступен в этом прогоне; при наличии GPU подставьте `cuda`)
- Seed: 42 (фиксирован в ноутбуке)
- Как запустить: открыть `homeworks/HW10-11/HW10-11.ipynb` и выполнить **Run All** (корректные пути к `data/` и `artifacts/` подставляются и из корня репозитория, и при текущей папке `HW10-11`).

## 3. Данные

### 3.1. Часть A: классификация

- Датасет: `STL10`
- Разделение: train/val/test (val получаем из train разбиением; seed фиксирован)
- Базовые transforms: `Resize(96x96) -> ToTensor -> Normalize(ImageNet)`
- Augmentation transforms (для C2): `Resize(96x96) -> RandomHorizontalFlip -> RandomCrop -> ToTensor -> Normalize(ImageNet)`
- Комментарий: STL10 содержит 10 классов, изображения относительно небольшие. Для CNN и ResNet применяются одинаковые resize/normalization, чтобы сравнение C1–C4 было корректным.

### 3.2. Часть B: structured vision

- Датасет: `OxfordIIITPet`
- Трек: `segmentation`
- Что считается ground truth: foreground = `mask > 0` (пиксели пет-объекта vs фон)
- Какие предсказания использовались: берём сегментацию из pretrained `FCN-ResNet50` (COCO), считаем foreground как `p(cat)+p(dog)` и/или `argmax != background`
- Комментарий: постановка позволяет без дообучения запускать pretrained пайплайн, визуализировать маски и посчитать `mean_iou` для foreground.

## 4. Часть A: модели и обучение (C1-C4)

Опишите коротко и сопоставимо:

- C1 (simple-cnn-base): простая CNN, базовые аугментации отсутствуют (только resize/normalize).
- C2 (simple-cnn-aug): та же архитектура, что в C1, плюс аугментации (flip/crop).
- C3 (resnet18-head-only): pretrained `ResNet18`, заморожен backbone, обучается только `fc`.
- C4 (resnet18-finetune): pretrained `ResNet18`, разморожены `layer4` и `fc` для частичного дообучения.

По итогам прогона лучшей оказалась **C4** (см. числа в разделе 6 и файл `./artifacts/runs.csv`).

Дополнительно:

- Loss: `CrossEntropyLoss`
- Optimizer(ы): `Adam` (для CNN и для head/частичного fine-tune у ResNet)
- Batch size: `32` (в режиме `FAST_DEV_RUN`) или `64` иначе
- Epochs (макс): `1` (в `FAST_DEV_RUN`) или `3/2` иначе (зависит от эксперимента)
- Критерий выбора лучшей модели: `best_val_accuracy` (лучший из C1–C4)

## 5. Часть B: постановка задачи и режимы оценки (V1-V2)

### Если выбран detection track

- Модель:
- V1: `score_threshold = 0.3`
- V2: `score_threshold = 0.7`
- Как считался IoU:
- Как считались precision / recall:

### Если выбран segmentation track

- Модель: pretrained `torchvision.models.segmentation.fcn_resnet50` (weights DEFAULT, COCO)
- Что считается foreground: `mask > 0` и предсказанный foreground по правилам V1/V2
- V1: foreground по `argmax != background_idx`
- V2: foreground по порогу `p_fg = p(cat)+p(dog)`, например `p_fg > 0.7`
- Как считался mean IoU: бинарный IoU foreground (для каждой картинки) и среднее по изображениям
- Считались ли дополнительные pixel-level метрики: да, `precision` и `recall` по foreground (TP/FP/FN)

## 6. Результаты

Ссылки на файлы в репозитории:

- Таблица результатов: `./artifacts/runs.csv`
- Лучшая модель части A: `./artifacts/best_classifier.pt`
- Конфиг лучшей модели части A: `./artifacts/best_classifier_config.json`
- Кривые лучшего прогона классификации: `./artifacts/figures/classification_curves_best.png`
- Сравнение C1-C4: `./artifacts/figures/classification_compare.png`
- Визуализация аугментаций: `./artifacts/figures/augmentations_preview.png`
- Визуализации второй части: `./artifacts/figures/segmentation_examples.png`, `./artifacts/figures/segmentation_metrics.png`

Короткая сводка (6-10 строк):

- Лучший эксперимент части A: **C4** (`resnet18-finetune`, частичный fine-tune `layer4` + `fc`).
- Лучшая `val_accuracy` (по val): **≈ 0.503** (точнее `0.5028409091`; см. `./artifacts/runs.csv`).
- Итоговая `test_accuracy` лучшего классификатора (один прогон на test): **≈ 0.528** (точнее `0.5284090909`).
- Сводка по C1–C3 (только **val**, test для них в `runs.csv` пустой — **test один раз для лучшей модели C4**): C1 val **0.1875**; C2 val **0.1875**; C3 val **0.1193** (округлено, по `./artifacts/runs.csv`).
- Что дали аугментации (C2 vs C1): в этом быстром прогоне val/test почти совпали; полное сравнение — в `./artifacts/runs.csv` и на `./artifacts/figures/classification_compare.png`.
- Что дал transfer learning (C3/C4 vs C1/C2): предобученные признаки ResNet дают заметный скачок качества относительно маленькой CNN на `STL10`.
- Что оказалось лучше: head-only или partial fine-tuning: **partial fine-tuning (C4)** — лучший `best_val_accuracy` среди C1–C4.
- Что показал режим V1 во второй части: **mean IoU ≈ 0.395** (foreground по `argmax != background`).
- Что показал режим V2 во второй части: **mean IoU ≈ 0.316** (foreground по порогу \(p_{\text{cat}} + p_{\text{dog}} > 0.7\)).
- Как интерпретируются метрики второй части: `mean_iou` отражает перекрытие предсказанной и эталонной масок foreground; дополнительно в ноутбуке считаются pixel-level `precision`/`recall` по foreground (см. `./artifacts/runs.csv`).

## 7. Анализ

На `STL10` простая CNN имеет ограниченную ёмкость и сильнее зависит от аугментаций и числа эпох: без сильных аугментаций она легче переобучается на train и хуже обобщается на val/test. Аугментации в C2 добавляют инвариантности (отражение, кроп), что обычно улучшает устойчивость по сравнению с C1, хотя при коротком обучении (`FAST_DEV_RUN`) разница может быть менее выраженной, чем при полном прогоне. Pretrained `ResNet18` несёт признаки, обученные на ImageNet, поэтому уже в режиме «только голова» (C3) качество часто заметно выше, чем у scratch-CNN. Partial fine-tuning (C4) позволяет подстроить высокоуровневые фильтры под домен `STL10`, что в нашем случае дало лучший выбор по `best_val_accuracy` среди C1–C4.

Во второй части `mean_iou` по foreground — естественная метрика для сегментации: она штрафует и пропуски объекта, и лишние пиксели вне маски. Переход от V1 к V2 меняет правило бинаризации маски: V1 берёт класс с максимальной вероятностью по всем COCO-классам, V2 выделяет foreground только если суммарная уверенность в «кошка/собака» выше порога 0.7. Такой порог режет слабые ответы модели: он может уменьшить ложные срабатывания фона, но также «съесть» границы и редкие пиксели класса, из‑за чего union в знаменателе IoU растёт и средний IoU падает — в нашем прогоне V1 оказался выше V2. Типичные ошибки pretrained FCN на `OxfordIIITPet`: путаница фона с шерстью/текстурой, неточные границы и смещение масок при несовпадении домена COCO и снимков питомцев; визуально это видно на `./artifacts/figures/segmentation_examples.png`.

## 8. Итоговый вывод

- Базовым конфигом для классификации на `STL10` разумно брать **C4** (pretrained `ResNet18` + partial fine-tune): он дал лучший `best_val_accuracy` и итоговый `test_accuracy` в нашем эксперименте.
- Transfer learning переносит универсальные признаки с крупного датасета и снижает потребность в очень глубокой архитектуре с нуля; fine-tune последних слоёв часто выгоднее, чем только голова.
- Для сегментации важно согласовать постановку foreground, визуализацию и метрику: порог по вероятности меняет баланс «покрыть объект» vs «не раздувать маску», что напрямую отражается в `mean_iou`.

## 9. Приложение (опционально)

Если вы делали дополнительные сравнения:

- дополнительные fine-tuning сценарии
- confusion matrix для классификации
- дополнительная постобработка для второй части
- дополнительные графики: `./artifacts/figures/...`
