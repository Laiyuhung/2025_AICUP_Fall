import os
import glob
import argparse
import numpy as np
from collections import defaultdict
import re
from ultralytics import YOLO

# ================= 可編輯的參數（請直接在此區修改，註解為中文） =================
# 模型權重檔或模型資料夾（可填單一 .pt 檔或一個資料夾路徑，資料夾會視為模型群組）
MODEL_PATH = "1118_12n_best.pt"
# 測試圖片資料夾（會遞迴尋找 .png/.jpg）
TEST_FOLDER = "./42_testing_image/testing_image"
# Ground truth 檔案（與 best_comp.py 格式相同）
GT_FILE = "./competition_outputs/exp_v12n_20251015_154909_172epoch/predictions/exp_v12n_20251015_154909_172epoch_0p0088.txt"
# 輸出資料夾（所有 prediction / eval / summary 會寫在此處）
OUT_FOLDER = "./competition_outputs"
# confidence 範圍（min, max, step）
CONF_MIN = 0.0003
CONF_MAX = 0.07
CONF_STEP = 0.0005
# IoU 評估閾值（預設 0.5）
IOU_THRESH = 0.5
# ============================================================================


def iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    box1_area = max(0, (box1[2] - box1[0] + 1)) * max(0, (box1[3] - box1[1] + 1))
    box2_area = max(0, (box2[2] - box2[0] + 1)) * max(0, (box2[3] - box2[1] + 1))
    union = box1_area + box2_area - inter
    return inter / union if union > 0 else 0


def read_txt(file_path):
    data = defaultdict(list)
    if not os.path.exists(file_path):
        print(f"⚠️ 讀取失敗，找不到檔案: {file_path}")
        return data
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip().replace("\ufeff", "")
            if not line:
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < 7:
                continue
            img_name = parts[0]
            try:
                cls = int(parts[1])
                conf = float(parts[2])
                x1, y1, x2, y2 = map(float, parts[3:7])
            except Exception:
                continue
            data[img_name].append({"conf": conf, "box": [x1, y1, x2, y2]})
    return data


def calculate_ap(preds, gts, iou_thres=0.5):
    all_preds = []
    for img, boxes in preds.items():
        for b in boxes:
            all_preds.append((img, b["conf"], b["box"]))
    all_preds.sort(key=lambda x: x[1], reverse=True)

    if len(all_preds) == 0:
        return 0.0, 0.0, 0.0

    TP, FP = [], []
    matched = defaultdict(set)
    total_gt = sum(len(v) for v in gts.values())

    if total_gt == 0:
        return 0.0, 0.0, 0.0

    for img, conf, pbox in all_preds:
        best_iou, best_gt = 0, None
        for i, g in enumerate(gts.get(img, [])):
            iou_val = iou(pbox, g["box"])
            if iou_val > best_iou:
                best_iou, best_gt = iou_val, i
        if best_iou >= iou_thres and best_gt not in matched[img]:
            TP.append(1)
            FP.append(0)
            matched[img].add(best_gt)
        else:
            TP.append(0)
            FP.append(1)

    TP_cum = np.cumsum(TP)
    FP_cum = np.cumsum(FP)
    recall = TP_cum / total_gt
    precision = TP_cum / (TP_cum + FP_cum + 1e-6)
    AP = np.trapz(precision, recall) if len(precision) > 0 else 0.0

    return AP, (precision[-1] if len(precision) else 0), (recall[-1] if len(recall) else 0)


def run_inference_and_eval(weight_path, img_paths, gt_file, output_dir, conf_values, iou_thresh=0.5):
    model_name = os.path.splitext(os.path.basename(weight_path))[0]
    model_out_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_out_dir, exist_ok=True)
    preds_dir = os.path.join(model_out_dir, "predictions")
    eval_dir = os.path.join(model_out_dir, "eval")
    os.makedirs(preds_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    print(f"➡️  載入模型 {weight_path} ...")
    model = YOLO(weight_path)

    # load ground truth once
    gts = read_txt(gt_file)

    summary = []
    
    # 只用最低的 confidence threshold 跑一次模型推論
    min_conf = min(conf_values)
    min_conf_str = f"{min_conf:.4f}".replace('.', 'p')
    min_pred_file = os.path.join(preds_dir, f"{model_name}_{min_conf_str}.txt")
    
    all_results = []
    
    # 只跑一次最低 confidence 的推論
    print(f"   • 推論 conf={min_conf:.4f} ({len(img_paths)} 張圖) ...")
    batch_size = 64  # Process 64 images at a time
    results_all = []
    
    try:
        for i in range(0, len(img_paths), batch_size):
            batch_paths = img_paths[i:i+batch_size]
            batch_results = model(batch_paths, conf=min_conf, iou=iou_thresh, verbose=False)
            results_all.extend(batch_results)
            # Print progress
            if (i + batch_size) % 256 == 0 or i + batch_size >= len(img_paths):
                print(f"      處理進度: {min(i + batch_size, len(img_paths))}/{len(img_paths)} 張圖")
    except Exception as e:
        # fall back to per-image inference
        print(f"   ⚠️ 批次推論失敗: {e}，改為逐張推論")
        results_all = []
        for p in img_paths:
            results_all.append(model(p, conf=min_conf, iou=iou_thresh, verbose=False)[0])

    # 將所有結果存成一個清單，包含置信度，以便之後篩選
    for img_path, res in zip(img_paths, results_all):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        if getattr(res, 'boxes', None) is None:
            continue
        try:
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
        except Exception:
            # no boxes
            continue
        for box, c in zip(boxes, confs):
            x1, y1, x2, y2 = box
            all_results.append({
                "img_name": img_name,
                "conf": float(c),
                "box": [int(x1), int(y1), int(x2), int(y2)]
            })
    
    # 對每個 confidence 值，篩選結果並評估
    for conf in conf_values:
        conf_str = f"{conf:.4f}".replace('.', 'p')
        pred_file = os.path.join(preds_dir, f"{model_name}_{conf_str}.txt")
        eval_file = os.path.join(eval_dir, f"{model_name}_{conf_str}_eval.txt")
        
        # 篩選出達到當前閾值的結果
        filtered_results = [r for r in all_results if r["conf"] >= conf]
        
        # 轉換成需要的輸出格式
        results_list = []
        for r in filtered_results:
            line = f"{r['img_name']} 0 {r['conf']:.4f} {r['box'][0]} {r['box'][1]} {r['box'][2]} {r['box'][3]}"
            results_list.append(line)
        
        print(f"   • 篩選 conf>={conf:.4f} 產生 {len(results_list)} 個預測結果")
        
        # write prediction file
        with open(pred_file, 'w', encoding='utf-8', newline='\n') as f:
            f.write('\n'.join(results_list))

        # evaluate
        preds = read_txt(pred_file)
        ap, prec, rec = calculate_ap(preds, gts, iou_thres=iou_thresh)

        # write eval file
        with open(eval_file, 'w', encoding='utf-8', newline='\n') as f:
            f.write(f"AP@{iou_thresh:.2f}  : {ap:.6f}\n")
            f.write(f"Precision: {prec:.6f}\n")
            f.write(f"Recall   : {rec:.6f}\n")
            f.write(f"Predictions: {len(results_list)}\n")

        summary.append({
            'model': model_name,
            'weight': weight_path,
            'conf': conf,
            'AP': ap,
            'precision': prec,
            'recall': rec,
            'pred_file': pred_file,
            'eval_file': eval_file,
        })

        print(f"   ✅ conf={conf:.4f} -> AP={ap:.6f} P={prec:.6f} R={rec:.6f}  saved to {eval_file}")

    return summary


def collect_img_paths(test_path):
    img_paths = glob.glob(os.path.join(test_path, "**/*.png"), recursive=True)
    img_paths += glob.glob(os.path.join(test_path, "**/*.jpg"), recursive=True)
    img_paths = sorted(list(set(img_paths)))
    return img_paths


def parse_conf_range(min_conf, max_conf, step):
    if step <= 0:
        raise ValueError("step 必須為正")
    vals = np.arange(min_conf, max_conf + step/2, step)
    vals = np.round(vals.astype(float), 6)
    return vals


def main():
    parser = argparse.ArgumentParser(description="Batch inference+evaluation for multiple model weights and confidence ranges")
    parser.add_argument('--model', '-m', required=False,
                        help='path to a model weight (.pt) or a directory containing weights (this folder is treated as model name)',
                        default=MODEL_PATH)
    parser.add_argument('--test', '-t', default=TEST_FOLDER, help='test images folder')
    parser.add_argument('--gt', '-g', required=False, help='ground truth txt file to evaluate against', default=GT_FILE)
    parser.add_argument('--out', '-o', default=OUT_FOLDER, help='output base folder')
    parser.add_argument('--conf-min', type=float, default=CONF_MIN, help='minimum confidence')
    parser.add_argument('--conf-max', type=float, default=CONF_MAX, help='maximum confidence')
    parser.add_argument('--conf-step', type=float, default=CONF_STEP, help='confidence step')
    parser.add_argument('--iou', type=float, default=IOU_THRESH, help='IoU threshold for evaluation')
    args = parser.parse_args()

    # determine weight files
    weights = []
    if os.path.isfile(args.model):
        weights = [os.path.abspath(args.model)]
    elif os.path.isdir(args.model):
        # find .pt files inside
        pt_files = glob.glob(os.path.join(args.model, '*.pt'))
        weights = [os.path.abspath(p) for p in sorted(pt_files)]
    else:
        # try to interpret model as name of folder in cwd
        alt = os.path.join(os.getcwd(), args.model)
        if os.path.isdir(alt):
            pt_files = glob.glob(os.path.join(alt, '*.pt'))
            weights = [os.path.abspath(p) for p in sorted(pt_files)]
        else:
            raise FileNotFoundError(f"找不到模型或資料夾: {args.model}")

    if len(weights) == 0:
        raise FileNotFoundError(f"在指定的模型路徑未找到任何 .pt 權重: {args.model}")

    img_paths = collect_img_paths(args.test)
    if len(img_paths) == 0:
        raise FileNotFoundError(f"找不到測試圖片於: {args.test}")

    conf_values = parse_conf_range(args.conf_min, args.conf_max, args.conf_step)

    os.makedirs(args.out, exist_ok=True)

    all_summary = []
    for w in weights:
        s = run_inference_and_eval(w, img_paths, args.gt, args.out, conf_values, iou_thresh=args.iou)
        all_summary.extend(s)

    # write global summary CSV
    summary_file = os.path.join(args.out, 'summary.csv')
    with open(summary_file, 'w', encoding='utf-8', newline='\n') as f:
        f.write('model,weight,conf,AP,precision,recall,pred_file,eval_file\n')
        for r in all_summary:
            f.write(f"{r['model']},{os.path.basename(r['weight'])},{r['conf']:.6f},{r['AP']:.6f},{r['precision']:.6f},{r['recall']:.6f},{r['pred_file']},{r['eval_file']}\n")

    print(f"\n🎉 完成。所有結果已存到 {args.out}，總共 {len(all_summary)} 個 evaluation 結果。總表: {summary_file}")


if __name__ == '__main__':
    main()
