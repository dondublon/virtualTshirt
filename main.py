"""
Virtual T-shirt: закрывает голый торс на веб-камере и отдаёт результат
в виртуальную камеру (v4l2loopback) для использования в Zoom.

Использование:
    uv run main.py --camera 0 --output /dev/video10

Подбери --camera под свою реальную камеру (см. `v4l2-ctl --list-devices`).
"""

import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pyvirtualcam
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as mp_base_options

MODEL_PATH = Path(__file__).parent / "models" / "pose_landmarker_lite.task"

SHIRT_COLOR = (60, 60, 190)  # BGR — тёмно-красная футболка

# Индексы точек позы (PoseLandmarker), см. официальную схему MediaPipe Pose.
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12

# Где проводить линию горловины между носом и плечами: 0 = на уровне носа
# (закроет и шею), 1 = на уровне плеч (может оставить голую шею/грудь).
NECKLINE_RATIO = 0.7


def build_shirt_mask(frame_shape, landmarks, person_mask):
    """Всё тело человека ниже линии горловины — по реальному силуэту,
    а не по нарисованному контуру, чтобы футболка точно закрывала торс
    целиком (руки, бока, живот) без пробелов."""
    h, w = frame_shape[:2]

    nose_y = landmarks[NOSE].y * h
    shoulder_y = (landmarks[LEFT_SHOULDER].y + landmarks[RIGHT_SHOULDER].y) / 2 * h

    neckline_y = int(round(nose_y + (shoulder_y - nose_y) * NECKLINE_RATIO))
    neckline_y = max(0, min(h, neckline_y))

    band = np.zeros((h, w), dtype=np.uint8)
    band[neckline_y:, :] = 255
    return cv2.bitwise_and(band, person_mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="Индекс реальной камеры")
    parser.add_argument("--output", default="/dev/video10", help="Виртуальное устройство")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--debug", action="store_true", help="Показать окно предпросмотра")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit(f"Не найдена модель позы: {MODEL_PATH}")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        raise SystemExit(f"Не удалось открыть камеру {args.camera}")

    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Камера открылась, но кадр прочитать не удалось")
    h, w = frame.shape[:2]

    options = vision.PoseLandmarkerOptions(
        base_options=mp_base_options.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=True,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    shirt_mask_smooth = None
    smoothing = 0.5  # 0 = без сглаживания, ближе к 1 = сильнее сглаживание
    start_time = time.monotonic()

    with pyvirtualcam.Camera(
        width=w, height=h, fps=args.fps, device=args.output, fmt=pyvirtualcam.PixelFormat.BGR
    ) as cam:
        print(f"Виртуальная камера запущена: {cam.device} ({w}x{h}@{args.fps})")
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - start_time) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            out = frame

            if result.pose_landmarks and result.segmentation_masks:
                landmarks = result.pose_landmarks[0]
                person_mask = (
                    result.segmentation_masks[0].numpy_view() > 0.4
                ).astype(np.uint8) * 255
                shirt_mask = build_shirt_mask(frame.shape, landmarks, person_mask)
                shirt_mask = cv2.GaussianBlur(shirt_mask, (15, 15), 0)

                if shirt_mask_smooth is None:
                    shirt_mask_smooth = shirt_mask.astype(np.float32)
                else:
                    shirt_mask_smooth = (
                        smoothing * shirt_mask_smooth
                        + (1 - smoothing) * shirt_mask.astype(np.float32)
                    )

                alpha = (shirt_mask_smooth / 255.0)[..., None]
                shirt_layer = np.full_like(frame, SHIRT_COLOR, dtype=np.uint8)
                out = (
                    frame.astype(np.float32) * (1 - alpha)
                    + shirt_layer.astype(np.float32) * alpha
                ).astype(np.uint8)
            else:
                shirt_mask_smooth = None

            if args.debug:
                cv2.imshow("preview", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            cam.send(out)
            cam.sleep_until_next_frame()

    landmarker.close()
    cap.release()
    if args.debug:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
