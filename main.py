# -*- coding: utf-8 -*-
"""
主控腳本 - 資料前處理 / 模型訓練 / 比賽驗證 (AP@0.5)

使用方式:
python main.py preprocess
python main.py train
python main.py eval
python main.py eval-threshold <model_path>
python main.py tune
"""

import os
import shutil
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from ultralytics import YOLO
from datetime import datetime
import argparse
import random
import locale

# 設置系統編碼為 UTF-8
try:
    locale.setlocale(locale.LC_ALL, 'C.UTF-8')
except locale.Error:
    pass


# =========================
# 浮動參數設定 (針對 YOLOv12n.pt)
# =========================
TRAIN_CSV = "train.csv"
VAL_CSV = "val.csv"
BASE_IMAGE_PATH = "42_training_image/training_image"
DATASET_PATH = "dataset"
CLASS_NAMES = ["target"]
NC = len(CLASS_NAMES)

nowtime = datetime.now().strftime("%Y%m%d_%H%M%S")
# MODEL_SAVE_NAME = f"exp_v12n_{nowtime}"
MODEL_SAVE_NAME = "1118_12n"  # 固定名稱，方便後續辨識

RESUME = False  # ✅ 可從中斷點續訓
MODEL_INIT = "yolo12n.pt"  # 預訓練模型或先前訓練的權重檔
DATA_YAML = "data.yaml"
EPOCHS = 180
BATCH = 32
IMG_SIZE = 640
DEVICE = 0
LOSS_BOX = 9.0
LOSS_CLS = 1.5
LOSS_DFL = 1.5
AUG = {
    "translate": 0.05,
    "scale": 0.8,
    "fliplr": 0.5,
    "flipud": 0.05,
    "mosaic": 0.3,
    "mixup": 0.05,
    "erasing": 0.1,
    "auto_augment": None,
}
LR0 = 0.01
LRF = 0.001
COS_LR = True
WARMUP_EPOCHS = 5
PATIENCE = 20
FREEZE = 0
KEEP_EMPTY_RATIO = 0.5


# =========================
# 資料前處理
# =========================
def preprocess():
    """
    互動式切分影像與標註，建立 YOLO 訓練結構
    (images/train, images/val, labels/train, labels/val)
    """
    print("📂 建立資料集結構 (互動式切分) ...")
    os.makedirs(os.path.join(DATASET_PATH, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_PATH, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_PATH, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_PATH, "labels", "val"), exist_ok=True)

    # 🧩 設定影像與標註根目錄
    IMAGES_ROOT = BASE_IMAGE_PATH
    LABELS_ROOT = r"C:\Users\laiyu\OneDrive\桌面\AICUP\training_label"  # ← 這裡可以改你的標註資料夾

    # 🔢 使用者輸入切分比例
    while True:
        try:
            val_ratio = float(input("請輸入驗證集比例 (0~1，例如 0.2 表示 20% 驗證)："))
            if 0 < val_ratio < 1:
                break
            print("❌ 請輸入介於 0~1 的數值")
        except ValueError:
            print("❌ 請輸入數字")

    # 🔍 掃描所有影像
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    all_images = []
    for root, _, files in os.walk(IMAGES_ROOT):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                all_images.append(os.path.join(root, f))

    print(f"📦 找到影像數量：{len(all_images)}")

    if len(all_images) == 0:
        print("❌ 沒有影像檔可切分，請確認路徑。")
        return

    import random
    random.shuffle(all_images)
    val_count = int(len(all_images) * val_ratio)
    val_images = all_images[:val_count]
    train_images = all_images[val_count:]

    print(f"✅ 訓練影像數：{len(train_images)}")
    print(f"✅ 驗證影像數：{len(val_images)}")

    def copy_image_and_label(img_paths, split_name):
        for img_path in img_paths:
            img_name = os.path.basename(img_path)
            dst_img = os.path.join(DATASET_PATH, "images", split_name, img_name)
            os.makedirs(os.path.dirname(dst_img), exist_ok=True)
            shutil.copy2(img_path, dst_img)

            # 嘗試找標註檔
            patient_folder = os.path.basename(os.path.dirname(img_path))
            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_path = os.path.join(LABELS_ROOT, patient_folder, label_name)
            dst_label = os.path.join(DATASET_PATH, "labels", split_name, label_name)
            if os.path.exists(label_path):
                shutil.copy2(label_path, dst_label)

    print("⏳ 複製訓練集影像與標註中 ...")
    copy_image_and_label(train_images, "train")
    print("⏳ 複製驗證集影像與標註中 ...")
    copy_image_and_label(val_images, "val")

    # ✏️ 寫出 YAML
    yaml_content = (
        f"train: {os.path.join(DATASET_PATH, 'images/train')}\n"
        f"val: {os.path.join(DATASET_PATH, 'images/val')}\n"
        f"nc: {NC}\n"
        f"names: {CLASS_NAMES}\n"
    )
    with open(DATA_YAML, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print("\n✅ 已完成切分並建立結構！")
    print(f"📁 YAML 已建立於：{DATA_YAML}")


# =========================
# 模型訓練（含 AP50 即時另存 best_ap50.pt）
# =========================
def train():
    run_name = MODEL_SAVE_NAME
    print(f"🚀 開始訓練: {run_name} (模型: {MODEL_INIT})")

    try:
        model = YOLO(MODEL_INIT)
    except Exception as e:
        print(f"❌ 無法載入模型 {MODEL_INIT}: {e}")
        return

    # 依 KEEP_EMPTY_RATIO 篩選無標註影像
    image_dir = os.path.join(DATASET_PATH, "images", "train")
    label_dir = os.path.join(DATASET_PATH, "labels", "train")

    all_images = os.listdir(image_dir) if os.path.exists(image_dir) else []
    kept_images = []
    for img_name in all_images:
        label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")
        has_label = os.path.exists(label_path) and os.path.getsize(label_path) > 0
        if has_label or random.random() < KEEP_EMPTY_RATIO:
            kept_images.append(os.path.join(image_dir, img_name))

    train_list_path = os.path.join(DATASET_PATH, "train_list.txt")
    os.makedirs(DATASET_PATH, exist_ok=True)
    with open(train_list_path, "w", encoding="utf-8") as f:
        f.write('\n'.join(kept_images))

    print(f"📊 篩選後保留 {len(kept_images)} 張圖片 (含 {KEEP_EMPTY_RATIO*100:.0f}% 無標註比例)")

    yaml_content = (
        f"train: {train_list_path}\n"
        f"val: {DATASET_PATH}/images/val\n"
        f"nc: {NC}\n"
        f"names: {CLASS_NAMES}\n"
    )
    with open(DATA_YAML, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    # === 以 AP50 即時另存 best_ap50.pt 的 callback ===
    best_ap50 = {"score": -1.0}

    def _get_ap50_from_metrics(metrics):
        """
        嘗試從 metrics 物件或 dict 取出 AP50，支援新版 DetMetrics。
        """
        if metrics is None:
            return None

        # 🧩 新版 YOLOv8/v12：metrics 是 DetMetrics 物件
        if hasattr(metrics, "results_dict"):
            d = metrics.results_dict
            if "metrics/box/map50" in d:
                return float(d["metrics/box/map50"])
            elif "metrics/mAP50(B)" in d:
                return float(d["metrics/mAP50(B)"])
            elif "metrics/map50" in d:
                return float(d["metrics/map50"])
            elif "map50" in d:
                return float(d["map50"])
            else:
                return None

        # 🧩 舊版：metrics 是 dict
        if isinstance(metrics, dict):
            for k in ("metrics/box/map50", "metrics/mAP50(B)", "metrics/map50", "map50"):
                if k in metrics:
                    try:
                        return float(metrics[k])
                    except Exception:
                        pass
        return None


    def _on_val_end(trainer):
        cur = _get_ap50_from_metrics(getattr(trainer, "metrics", {}) or {})
        if cur is None:
            return
        if cur > best_ap50["score"]:
            best_ap50["score"] = cur
            weights_dir = os.path.join(trainer.save_dir, "weights")
            os.makedirs(weights_dir, exist_ok=True)
            src = os.path.join(weights_dir, "last.pt")  # 每個 epoch 都會覆寫
            dst = os.path.join(weights_dir, "best_ap50.pt")
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"[AP50 callback] 🎯 New best AP50={cur:.4f} → saved: {dst}")

    # 掛上 callback（新版 API）
    try:
        model.add_callback("on_val_end", _on_val_end)
    except Exception:
        pass  # 若 add_callback 不可用，可改用 train(..., callbacks={"on_val_end": _on_val_end})

    try:
        model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            batch=BATCH,
            resume=RESUME,
            imgsz=IMG_SIZE,
            workers=8,
            device=DEVICE,
            amp=True,
            project="runs/detect",
            name=run_name,
            box=LOSS_BOX,
            cls=LOSS_CLS,
            dfl=LOSS_DFL,
            lr0=LR0,
            lrf=LRF,
            cos_lr=COS_LR,
            warmup_epochs=WARMUP_EPOCHS,
            patience=PATIENCE,
            freeze=FREEZE,
            # translate=AUG["translate"],
            # scale=AUG["scale"],
            # fliplr=AUG["fliplr"],
            # flipud=AUG["flipud"],
            # mosaic=AUG["mosaic"],
            # mixup=AUG["mixup"],
            save_period=-1,               # 不另外存 epochX.pt，避免佔空間
            # callbacks={"on_val_end": _on_val_end},  # 若上面 add_callback 失敗，改用這行（擇一）
        )
        print(f"✅ 訓練完成，模型存於 runs/detect/{run_name}/weights/best.pt")
        print(f"📌 若以 AP50 擇優，請使用 runs/detect/{run_name}/weights/best_ap50.pt")
    except Exception as e:
        print(f"❌ 訓練過程中發生錯誤: {e}")


# =========================
# 評分公式 AP@0.5
# =========================
def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return 0.0
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0


def calculate_ap(predictions, gts, iou_thr=0.5):
    predictions = sorted(predictions, key=lambda x: x[2], reverse=True)
    gt_dict = defaultdict(list)
    matched = defaultdict(list)

    for gt in gts:
        gt_dict[gt[0]].append(gt[2:6])
    for k, v in gt_dict.items():
        matched[k] = [False] * len(v)

    tp, fp = [], []
    for img, cls, conf, x1, y1, x2, y2 in predictions:
        pred_box = [x1, y1, x2, y2]
        best_iou, best_idx = 0, -1
        if img in gt_dict:
            for i, gt_box in enumerate(gt_dict[img]):
                if not matched[img][i]:
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou, best_idx = iou, i
        if best_iou >= iou_thr and best_idx != -1:
            tp.append(1)
            fp.append(0)
            matched[img][best_idx] = True
        else:
            tp.append(0)
            fp.append(1)

    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
    total_gt = sum(len(v) for v in gt_dict.values())
    precision = tp_cum / (tp_cum + fp_cum + 1e-10)
    recall = tp_cum / (total_gt + 1e-10)
    recall_points = np.concatenate([[0], recall])
    precision_points = np.concatenate([[precision[0] if len(precision)>0 else 0], precision])
    ap = np.sum((recall_points[1:] - recall_points[:-1]) * precision_points[1:])
    return ap, precision, recall


# =========================
# 比賽驗證（多模型 PR 曲線 + AP@0.5）
# =========================
def compare_models(model_paths, data_yaml, device="0", labels=None, use_tta=False):
    results = []
    plt.figure(figsize=(8, 6))
    val_dir = os.path.join(DATASET_PATH, "images", "val")
    label_dir = os.path.join(DATASET_PATH, "labels", "val")

    for i, mp in enumerate(model_paths):
        label = labels[i] if labels else f"Model{i+1}"
        print(f"\n🔍 驗證模型: {label} ({mp})")
        print(f"🚀 TTA 狀態: {'啟用' if use_tta else '禁用'}")
        model = YOLO(mp)

        preds, gts = [], []
        for img_file in os.listdir(val_dir):
            img_path = os.path.join(val_dir, img_file)
            results_yolo = model.predict(img_path, conf=0.001, device=device, verbose=False, augment=use_tta)
            for r in results_yolo:
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                for box, conf in zip(boxes, confs):
                    x1, y1, x2, y2 = box.tolist()
                    preds.append([img_file, 0, float(conf), x1, y1, x2, y2])

            label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + ".txt")
            if os.path.exists(label_path):
                with open(label_path, "r", encoding="utf-8") as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cls = int(float(parts[0]))
                        xc, yc, w, h = map(float, parts[1:])
                        img_cv = cv2.imread(img_path)
                        h_img, w_img = img_cv.shape[:2]
                        x1 = (xc - w/2) * w_img
                        y1 = (yc - h/2) * h_img
                        x2 = (xc + w/2) * w_img
                        y2 = (yc + h/2) * h_img
                        gts.append([img_file, cls, x1, y1, x2, y2])

        ap, prec, rec = calculate_ap(preds, gts)
        print(f"  AP@0.5={ap:.3f} (基於 {len(preds)} 預測 / {len(gts)} 標註)")
        results.append((label, ap))
        plt.plot(rec, prec, label=f"{label} (AP@0.5={ap:.3f}, TTA={'Y' if use_tta else 'N'})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Comparison (AP@0.5)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"competition_eval_{ts}.png", dpi=300, bbox_inches="tight")
    print(f"✅ 圖表已存: competition_eval_{ts}.png")


def eval_interactive():
    try:
        num_models = int(input("請輸入要比較的模型數量: ").strip())
        tta_input = input("是否啟用 TTA? (y/n, 預設 n): ").strip().lower()
        use_tta = tta_input == 'y'
    except ValueError:
        print("❌ 請輸入正整數")
        return

    model_paths, labels = [], []
    for i in range(num_models):
        path = input(f"請輸入第 {i+1} 個模型的 .pt 路徑: ").strip()
        label = input(f"請輸入第 {i+1} 個模型的名稱 (預設: Model{i+1}): ").strip()
        if not label:
            label = f"Model{i+1}"
        if os.path.exists(path):
            model_paths.append(path)
            labels.append(label)
        else:
            print(f"⚠️ 找不到檔案: {path}")

    if not model_paths:
        print("❌ 沒有有效的模型路徑。")
        return

    compare_models(model_paths, DATA_YAML, DEVICE, labels, use_tta=use_tta)


# =========================
# 尋找最佳信心度閾值（F1@IoU0.5）
# =========================
def calculate_f1_score(predictions, gts, iou_thr=0.5):
    """計算給定預測和標註的 TP, FP, FN"""
    gt_dict = defaultdict(list)
    for gt in gts:
        gt_dict[gt[0]].append([gt[2:6], False])

    tp = 0
    fp = 0
    predictions = sorted(predictions, key=lambda x: x[2], reverse=True)

    for img, cls, conf, x1, y1, x2, y2 in predictions:
        pred_box = [x1, y1, x2, y2]
        best_iou, best_idx = 0, -1

        if img in gt_dict:
            for i, (gt_box, is_matched) in enumerate(gt_dict[img]):
                if not is_matched:
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou, best_idx = iou, i

        if best_iou >= iou_thr and best_idx != -1:
            if not gt_dict[img][best_idx][1]:
                tp += 1
                gt_dict[img][best_idx][1] = True
            else:
                fp += 1
        else:
            fp += 1

    total_gt = sum(len(v) for v in gt_dict.values())
    fn = total_gt - tp

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    return f1, precision, recall


def find_best_threshold(model_path, device="0"):
    print(f"\n🔍 正在為模型 {model_path} 搜尋最佳信心度閾值...")
    model = YOLO(model_path)
    val_dir = os.path.join(DATASET_PATH, "images", "val")
    label_dir = os.path.join(DATASET_PATH, "labels", "val")

    # 1. 載入所有 GT 標註
    gts = []
    for img_file in os.listdir(val_dir):
        label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + ".txt")
        if os.path.exists(label_path):
            img_path = os.path.join(val_dir, img_file)
            img_cv = cv2.imread(img_path)
            h_img, w_img = img_cv.shape[:2]
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls, xc, yc, w, h = map(float, parts)
                    x1 = (xc - w/2) * w_img
                    y1 = (yc - h/2) * h_img
                    x2 = (xc + w/2) * w_img
                    y2 = (yc + h/2) * h_img
                    gts.append([img_file, int(cls), x1, y1, x2, y2])

    # 2. 用極低的信心度預測一次，獲取所有可能的預測框
    print("⏳ 正在對驗證集進行預測 (conf=0.0003)...")
    all_preds = []
    for img_file in os.listdir(val_dir):
        img_path = os.path.join(val_dir, img_file)
        results_yolo = model.predict(img_path, conf=0.0003, device=device, verbose=False, stream=True)
        for r in results_yolo:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = box.tolist()
                all_preds.append([img_file, 0, float(conf), x1, y1, x2, y2])

    print(f"✅ 預測完成，共得到 {len(all_preds)} 個候選框。")

    # 3. 遍歷閾值，尋找最佳 F1-Score
    best_f1 = -1
    best_threshold = -1
    best_p = -1
    best_r = -1

    thresholds = np.arange(0.05, 0.95, 0.01)
    f1_scores = []

    print("🧠 正在搜尋最佳閾值...")
    for conf_thr in thresholds:
        filtered_preds = [p for p in all_preds if p[2] >= conf_thr]
        f1, precision, recall = calculate_f1_score(filtered_preds, gts)
        f1_scores.append(f1)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = conf_thr
            best_p = precision
            best_r = recall

    print("\n" + "="*40)
    print("🎉 搜尋完成！ 🎉")
    print(f"🎯 最佳信心度閾值: {best_threshold:.2f}")
    print(f"📊 在此閾值下的表現:")
    print(f"   - 最高 F1-Score: {best_f1:.4f}")
    print(f"   - Precision: {best_p:.4f}")
    print(f"   - Recall:    {best_r:.4f}")
    print("="*40)
    print("💡 建議：用這個閾值跑比賽測試集產生提交檔。")

    # 繪製 F1-Score vs. Threshold 圖
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, marker='.', label='F1-Score')
    plt.axvline(best_threshold, linestyle='--', label=f'Best = {best_threshold:.2f}')
    plt.title('F1-Score vs. Confidence Threshold')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('F1-Score')
    plt.grid(True, alpha=0.5)
    plt.legend()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"f1_score_analysis_{ts}.png"
    plt.savefig(save_path, dpi=300)
    print(f"📈 分析圖表已儲存至: {save_path}")


# =========================
# 主程式入口
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO 實驗主控腳本")
    subparsers = parser.add_subparsers(dest="mode", help="選擇要執行的模式", required=True)

    subparsers.add_parser("preprocess", help="執行資料前處理")
    subparsers.add_parser("train", help="執行模型訓練")
    subparsers.add_parser("eval", help="互動式比較多個模型的 AP@0.5")

    parser_threshold = subparsers.add_parser("eval-threshold", help="為單一模型尋找最佳信心度閾值")
    parser_threshold.add_argument("model_path", type=str, help="要分析的 .pt 模型路徑")

    subparsers.add_parser("tune", help="執行超參數自動搜尋")

    args = parser.parse_args()

    if args.mode == "preprocess":
        preprocess()
    elif args.mode == "train":
        train()
    elif args.mode == "eval":
        eval_interactive()
    elif args.mode == "eval-threshold":
        if os.path.exists(args.model_path):
            find_best_threshold(args.model_path, device=str(DEVICE))
        else:
            print(f"❌ 錯誤: 找不到模型檔案 {args.model_path}")
    elif args.mode == "tune":
        print("🚀 開始自動超參數搜尋 (Tuning)...")
        try:
            model = YOLO(MODEL_INIT)
            # 確保 preprocess/train 已建立最新的 data.yaml & train_list.txt
            model.tune(
                data=DATA_YAML,
                epochs=50,         # Tuning 不需太長
                iterations=300,    # 嘗試 300 組
                optimizer='AdamW',
                plots=True,        # ✅ 參數名為 plots
                save=True,
                val=True,
                project="runs/tune",
                name=f"tune_{nowtime}"
            )
            print("✅ 搜尋完成！最佳參數已儲存。")
        except Exception as e:
            print(f"❌ 執行 tune 時發生錯誤: {e}")
            print("💡 請檢查 ultralytics 是否為最新版本： pip install -U ultralytics")
