"""
Virtual T-shirt: закрывает голый торс на веб-камере и отдаёт результат
в виртуальную камеру (v4l2loopback) для использования в Zoom.

Использование:
    uv run main.py --camera 0 --output /dev/video10

Подбери --camera под свою реальную камеру (см. `v4l2-ctl --list-devices`).
"""

import argparse

import cv2
import mediapipe as mp
import numpy as np
import pyvirtualcam

mp_pose = mp.solutions.pose
mp_selfie = mp.solutions.selfie_segmentation

SHIRT_COLOR = (60, 60, 190)  # BGR — тёмно-красная футболка


def build_torso_mask(frame_shape, landmarks, pad_ratio=0.18):
    """Полигон футболки: от плеч вниз до низа кадра, с запасом по бокам."""
    h, w = frame_shape[:2]

    def pt(name):
        lm = landmarks[mp_pose.PoseLandmark[name].value]
        return np.array([lm.x * w, lm.y * h])

    l_sh, r_sh = pt("LEFT_SHOULDER"), pt("RIGHT_SHOULDER")
    l_hip, r_hip = pt("LEFT_HIP"), pt("RIGHT_HIP")
    l_ear, r_ear = pt("LEFT_EAR"), pt("RIGHT_EAR")

    shoulder_w = np.linalg.norm(l_sh - r_sh)
    pad = shoulder_w * pad_ratio

    # Воротник поднимаем чуть выше линии плеч к шее, рукава расширяем по бокам.
    neck_l = l_sh + (l_ear - l_sh) * 0.35
    neck_r = r_sh + (r_ear - r_sh) * 0.35

    l_sleeve = l_sh + np.array([pad * 1.6, -pad * 0.2])
    r_sleeve = r_sh + np.array([-pad * 1.6, -pad * 0.2])

    bottom_y = min(h - 1, max(l_hip[1], r_hip[1]) + shoulder_w * 0.9)
    l_bottom = np.array([l_hip[0] - pad, bottom_y])
    r_bottom = np.array([r_hip[0] + pad, bottom_y])

    polygon = np.array(
        [neck_l, l_sleeve, l_sh, l_bottom, r_bottom, r_sh, r_sleeve, neck_r],
        dtype=np.int32,
    )

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="Индекс реальной камеры")
    parser.add_argument("--output", default="/dev/video10", help="Виртуальное устройство")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--debug", action="store_true", help="Показать окно предпросмотра")
    args = parser.parse_args()

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

    pose = mp_pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    segmenter = mp_selfie.SelfieSegmentation(model_selection=1)

    shirt_mask_smooth = None
    smoothing = 0.5  # 0 = без сглаживания, ближе к 1 = сильнее сглаживание

    with pyvirtualcam.Camera(
        width=w, height=h, fps=args.fps, device=args.output, fmt=pyvirtualcam.PixelFormat.BGR
    ) as cam:
        print(f"Виртуальная камера запущена: {cam.device} ({w}x{h}@{args.fps})")
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False

            pose_result = pose.process(rgb)
            seg_result = segmenter.process(rgb)

            out = frame

            if pose_result.pose_landmarks is not None:
                torso = build_torso_mask(frame.shape, pose_result.pose_landmarks.landmark)

                # Ограничиваем футболку силуэтом человека (segmentation mask),
                # чтобы она не рисовалась поверх фона за пределами тела.
                person_mask = (seg_result.segmentation_mask > 0.4).astype(np.uint8) * 255
                shirt_mask = cv2.bitwise_and(torso, person_mask)
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
                out = (frame.astype(np.float32) * (1 - alpha) + shirt_layer.astype(np.float32) * alpha).astype(
                    np.uint8
                )

            if args.debug:
                cv2.imshow("preview", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            cam.send(out)
            cam.sleep_until_next_frame()

    cap.release()
    if args.debug:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
