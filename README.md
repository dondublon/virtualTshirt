# vurtualTshirt

Виртуальная камера для Zoom, которая закрывает голый торс "футболкой"
через трекинг позы (MediaPipe) — без реальной одежды.

Пайплайн: реальная камера → OpenCV + MediaPipe (поза + сегментация
человека) → рисуем футболку поверх торса → кадр уходит в виртуальную
камеру `v4l2loopback` → её выбираешь в Zoom вместо настоящей камеры.

## Установка на чистой машине (Xubuntu / Ubuntu 22.04)

### 1. Системные пакеты

```bash
sudo apt update
sudo apt install -y v4l-utils dkms build-essential git
```

### 2. Модуль виртуальной камеры v4l2loopback

Пакет `v4l2loopback-dkms` из репозиториев Ubuntu 22.04 (0.12.7) **не
работает** на свежих ядрах 6.8+ — модуль грузится, но не объявляет
capture/output capabilities, и Zoom/ffmpeg не видят устройство как
рабочее. Поэтому собираем актуальную версию из исходников через DKMS:

```bash
git clone --depth 1 https://github.com/umlaeute/v4l2loopback.git /tmp/v4l2loopback-src
V4L2LB_VER=$(grep PACKAGE_VERSION /tmp/v4l2loopback-src/dkms.conf | cut -d'"' -f2)

# если пакетная версия уже стоит — снести
sudo apt remove -y v4l2loopback-dkms 2>/dev/null
sudo modprobe -r v4l2loopback 2>/dev/null

sudo rm -rf "/usr/src/v4l2loopback-$V4L2LB_VER"
sudo cp -r /tmp/v4l2loopback-src "/usr/src/v4l2loopback-$V4L2LB_VER"
sudo dkms add -m v4l2loopback -v "$V4L2LB_VER"
sudo dkms build -m v4l2loopback -v "$V4L2LB_VER"
sudo dkms install -m v4l2loopback -v "$V4L2LB_VER"
```

Проверить, что модуль собрался под текущее ядро:

```bash
dkms status
```

Должна быть строка вида `v4l2loopback/0.15.4, <версия ядра>: installed`.

#### Автозагрузка модуля при старте системы (по желанию)

Чтобы виртуальная камера появлялась сразу после включения компьютера,
не набирая `modprobe` руками каждый раз:

```bash
echo "v4l2loopback" | sudo tee /etc/modules-load.d/v4l2loopback.conf
echo 'options v4l2loopback video_nr=10 card_label="FakeCam" exclusive_caps=1' \
  | sudo tee /etc/modprobe.d/v4l2loopback.conf
sudo update-initramfs -u
```

После перезагрузки `/dev/video10` будет создаваться автоматически —
шаг "Активировать виртуальную камеру" из раздела запуска можно
пропускать.

### 3. Python-окружение (uv)

```bash
cd ~/storage/projects/vurtualTshirt
uv sync
```

`uv sync` поставит зависимости из `pyproject.toml`/`uv.lock`
(opencv-python, mediapipe, pyvirtualcam, numpy).

### 4. Модель позы MediaPipe

```bash
mkdir -p models
curl -sSL -o models/pose_landmarker_lite.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
```

### 5. Проверка своей камеры

```bash
v4l2-ctl --list-devices
```

Запомни индекс реальной камеры (например, `/dev/video0`) — он
понадобится как `--camera` при запуске.

## Запуск (когда нужно пользоваться)

Если автозагрузку модуля (шаг 2, опционально) не настраивал — сначала
подними виртуальную камеру:

```bash
sudo modprobe v4l2loopback video_nr=10 card_label="FakeCam" exclusive_caps=1
```

Затем запусти сам скрипт:

```bash
cd ~/storage/projects/vurtualTshirt
uv run main.py --camera 0 --output /dev/video10
```

В Zoom: Настройки → Видео → выбрать камеру **FakeCam**.

Остановить — `Ctrl+C` в терминале.

### Полезные флаги

- `--camera N` — индекс реальной камеры (по умолчанию `0`)
- `--output /dev/videoN` — устройство виртуальной камеры
- `--width`, `--height`, `--fps` — разрешение и частота кадров
- `--debug` — открыть окно предпросмотра на экране (для отладки без Zoom)

Пример с окном предпросмотра, без запуска Zoom:

```bash
uv run main.py --camera 0 --output /dev/video10 --debug
```

## Диагностика

- `v4l2-ctl --list-devices` — какие камеры видит система и под каким
  индексом.
- `v4l2-ctl -d /dev/video10 --all` — в блоке `Device Caps` должны быть
  `Video Capture` и `Video Output` (если их нет — см. раздел про сборку
  модуля выше).
- Если Zoom не видит FakeCam — перезапусти Zoom после того, как
  `uv run main.py` уже запущен (некоторые версии Zoom сканируют список
  камер только при старте).
