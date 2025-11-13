# src/utils/visualize_predictions.py

import argparse
import os
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import matplotlib.pyplot as plt

from src.infer.inference import PantsMeasurementPredictor
from src.utils.heatmap import decode_heatmaps_batch, resize_coords

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def _make_colors(num_points: int) -> np.ndarray:
    """각 keypoint마다 다른 RGB 색을 할당."""
    # 12개 정도의 고정 팔레트 (필요 시 반복해서 사용)
    base_colors = np.array([
        [1.0, 0.0, 0.0],   # red
        [0.0, 1.0, 0.0],   # green
        [0.0, 0.0, 1.0],   # blue
        [1.0, 1.0, 0.0],   # yellow
        [1.0, 0.5, 0.0],   # orange
        [1.0, 0.0, 1.0],   # magenta
        [0.0, 1.0, 1.0],   # cyan
        [0.6, 0.3, 0.2],   # brown
        [0.5, 0.0, 1.0],   # violet
        [0.0, 0.5, 0.5],   # teal
        [0.5, 1.0, 0.0],   # lime
        [1.0, 0.75, 0.8],  # pink
    ], dtype=np.float32)

    if num_points <= len(base_colors):
        return base_colors[:num_points]
    else:
        reps = int(np.ceil(num_points / len(base_colors)))
        colors = np.tile(base_colors, (reps, 1))
        return colors[:num_points]


def draw_keypoints_on_image(
    image: Image.Image,
    keypoints: np.ndarray,
    names: list,
    save_path: Path
):
    """
    원본 이미지 위에 각 measurement의 start/end 점과 선을 그려서 저장.

    Args:
        image: PIL Image (원본)
        keypoints: (K, 2) np.ndarray, 원본 이미지 좌표계 (x, y)
        names: measurement 이름 리스트
        save_path: 저장 경로 (png)
    """
    img_np = np.array(image)
    H, W = img_np.shape[:2]
    K = keypoints.shape[0]

    colors = _make_colors(K)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_np)
    ax.axis('off')

    for i, name in enumerate(names):
        start_idx = 2 * i
        end_idx = 2 * i + 1

        if start_idx >= K or end_idx >= K:
            continue

        x1, y1 = keypoints[start_idx]
        x2, y2 = keypoints[end_idx]

        c = colors[start_idx]  # start point 기준 색

        # 점 두 개
        ax.scatter(x1, y1, s=30, c=[c], marker='o')
        ax.scatter(x2, y2, s=30, c=[c], marker='x')

        # 두 점을 잇는 선
        ax.plot([x1, x2], [y1, y2], color=c, linewidth=2, label=name if i == 0 else None)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def create_heatmap_overlay(
    image: Image.Image,
    heatmaps: np.ndarray,  # (K, H_hm, W_hm)
    save_path: Path
):
    """
    모든 keypoint heatmap을 색깔 다르게 해서 원본 위에 overlay한 이미지를 저장.

    Args:
        image: PIL Image (원본)
        heatmaps: (K, H_hm, W_hm) numpy array
        save_path: 저장 경로
    """
    img_np = np.array(image).astype(np.float32) / 255.0
    H, W = img_np.shape[:2]
    K, H_hm, W_hm = heatmaps.shape

    colors = _make_colors(K)
    heat_rgb = np.zeros((H_hm, W_hm, 3), dtype=np.float32)

    # 각 keypoint heatmap에 색 입히고 합산
    for k in range(K):
        hm = heatmaps[k]
        if hm.max() > 0:
            hm = hm / hm.max()
        c = colors[k]
        for ch in range(3):
            heat_rgb[..., ch] += hm * c[ch]

    heat_rgb = np.clip(heat_rgb, 0.0, 1.0)

    # 원본 크기로 업샘플
    heat_img = Image.fromarray((heat_rgb * 255).astype(np.uint8))
    heat_img = heat_img.resize((W, H), resample=Image.BILINEAR)
    heat_np_up = np.array(heat_img).astype(np.float32) / 255.0

    # overlay (원본 0.6, heatmap 0.4 비율)
    overlay = np.clip(img_np * 0.6 + heat_np_up * 0.4, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(overlay)
    ax.axis('off')

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def run_inference_and_visualize(
    config_path: str,
    checkpoint_path: str,
    output_dir: str,
    highlight_uris: list,
    max_highlight: int = None
):
    """
    test.csv 전체에 대해 예측 → JSON 저장,
    highlight_uris 리스트에 있는 이미지들에 대해 시각화 PNG 저장.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Predictor 로드
    predictor = PantsMeasurementPredictor(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device
    )

    # config에서 test_csv 경로 가져오기
    test_csv = predictor.config.get('test_csv', None)
    if test_csv is None:
        raise ValueError("config.yaml에 'test_csv' 항목이 없습니다.")
    test_csv_path = Path(test_csv)
    if not test_csv_path.exists():
        raise FileNotFoundError(f"test_csv not found: {test_csv_path}")

    df = pd.read_csv(test_csv_path)
    logger.info(f"Loaded {len(df)} samples from {test_csv_path}")

    # output dir 설정: results/<checkpoint_name_without_ext>/
    model_name = Path(checkpoint_path).stem
    base_out_dir = Path(output_dir) / model_name
    base_out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results to: {base_out_dir}")

    # highlight용 URI set
    highlight_set = set(highlight_uris or [])
    if max_highlight is not None and len(highlight_set) > max_highlight:
        # 너무 많이 넣었으면 앞에서부터 자름
        highlight_list = list(highlight_set)[:max_highlight]
        highlight_set = set(highlight_list)

    all_results = {}
    highlight_count = 0

    for idx, row in df.iterrows():
        image_uri = row['image_uri']

        # 이미지 로드 및 전처리
        try:
            image = predictor.load_image(image_uri)
        except Exception as e:
            logger.error(f"[{idx}] Failed to load image {image_uri}: {e}")
            continue

        original_size = image.size  # (W, H)
        image_np = np.array(image)
        transformed = predictor.transform(image=image_np)
        image_tensor = transformed['image'].unsqueeze(0).to(device)

        # 추론 (heatmaps)
        with torch.no_grad():
            heatmaps = predictor.model(image_tensor)  # (1, K, H_hm, W_hm)

        heatmaps_np = heatmaps.cpu().numpy()[0]  # (K, H_hm, W_hm)

        # heatmap → 좌표 (heatmap 좌표계)
        keypoints_heatmap = decode_heatmaps_batch(heatmaps.cpu())[0]  # (K, 2)

        # 좌표를 원본 이미지 크기로 리사이즈
        keypoints_original = resize_coords(
            keypoints_heatmap,
            from_size=(predictor.heatmap_size[1], predictor.heatmap_size[0]),
            to_size=original_size  # (W, H)
        )  # (K, 2)

        # 결과 dict 생성 (기존 PantsMeasurementPredictor의 결과 포맷과 동일)
        result_per_img = {}
        for i, name in enumerate(predictor.names):
            start_idx = 2 * i
            end_idx = 2 * i + 1
            if end_idx >= keypoints_original.shape[0]:
                continue
            start_point = keypoints_original[start_idx].tolist()
            end_point = keypoints_original[end_idx].tolist()
            result_per_img[name] = {
                'start': start_point,
                'end': end_point
            }

        all_results[image_uri] = result_per_img

        # highlight 대상이면 시각화 PNG 저장
        if image_uri in highlight_set:
            highlight_count += 1
            # 파일 이름: index_XXXXX.png 형태로
            base_name = f"{idx:05d}"

            keypoint_png = base_out_dir / f"{base_name}_keypoints.png"
            heatmap_png = base_out_dir / f"{base_name}_heatmap.png"

            logger.info(f"[{idx}] Visualizing {image_uri}")
            draw_keypoints_on_image(
                image=image,
                keypoints=keypoints_original,
                names=predictor.names,
                save_path=keypoint_png
            )
            create_heatmap_overlay(
                image=image,
                heatmaps=heatmaps_np,
                save_path=heatmap_png
            )

    # 전체 결과 JSON으로 저장
    json_path = base_out_dir / "predictions.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved predictions JSON to: {json_path}")
    logger.info(f"Highlighted images (with PNGs): {highlight_count}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference on test.csv and visualize selected images."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best.pt",
        help="Path to model checkpoint (.pt)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Base output directory for results"
    )
    parser.add_argument(
        "--images",
        type=str,
        nargs="*",
        default=None,
        help="List of image_uri to visualize (하드코딩된 URI 리스트를 넘겨줄 때 사용)"
    )
    parser.add_argument(
        "--max_highlight",
        type=int,
        default=None,
        help="(옵션) highlight할 이미지 최대 개수 제한"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_inference_and_visualize(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        highlight_uris=args.images,
        max_highlight=args.max_highlight
    )


if __name__ == "__main__":
    main()
