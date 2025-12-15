# AI CUP 2025秋季賽－電腦斷層心臟肌肉影像分割競賽 II－主動脈瓣物件偵測

國立臺北科技大學 電資學士班 賴昱宏、陳羿錦、廖翊宏

YOLOv12n / Colab 前處理 / 模型訓練 / 多模型評估 / F1 閾值搜尋 / 批次推論

本專案整合了 AI CUP 模型訓練與評估所需的完整流程：

* Colab 前處理（ai_cup_preprocess.ipynb）
* YOLO 訓練
* 多模型 AP@0.5 驗證
* Confidence threshold（F1）搜尋
* 批次推論（multi-confidence）
* 提交檔產生

前處理完全使用 `ai_cup_preprocess.ipynb`，不需使用 main.py 的 preprocess。


# 專案結構

```
.
├── ai_cup_preprocess.ipynb  # Colab 專用資料前處理（主要使用）
├── main.py                  # 訓練、驗證、threshold 功能
├── competition_batch.py     # 批次推論＋AP 評估
├── dataset/                 # YOLO dataset（由前處理 notebook 產生）
├── 42_training_image/
├── 42_testing_image/
└── runs/                    # YOLO 訓練輸出
```



# 1. 安裝環境

```
pip install ultralytics opencv-python matplotlib numpy pandas
```



# 2. 資料前處理（請使用 Colab Notebook）

前處理統一使用這份 notebook：

**`ai_cup_preprocess.ipynb`（必須在 Colab 執行）**

Notebook 功能：

* 掃描原始 training_image
* 匹配標註 txt
* 設定 train/val split
* 產生 YOLO 格式 dataset：

```
dataset/
  images/train
  images/val
  labels/train
  labels/val
```

* 自動產生 `data.yaml`

## 注意：Colab 路徑 vs. 本機路徑

在 Notebook 內應使用：

```
/content/42_training_image
/content/training_label
/content/dataset
```

若在本機操作 `main.py`，請確保 dataset 路徑一致：

```
./dataset/images/train
./dataset/images/val
```



# 3. 模型訓練（使用 main.py）

main.py 的訓練程式碼：

執行：

```bash
python main.py train
```

輸出：

```
runs/detect/1118_12n/weights/
    best.pt
    best_ap50.pt   ← 建議比賽使用
```

其中：

* `best.pt` 是 YOLO 官方 early stop
* `best_ap50.pt` 是本專案根據「AP50 callback」額外儲存的最佳模型



# 4. 多模型評估（AP@0.5）

```bash
python main.py eval
```

會詢問：

* 模型數量
* 模型路徑
* 模型名稱
* 是否開啟 TTA

輸出：

* PR 曲線：`competition_eval_XXXX.png`
* 各模型 AP@0.5 成績



# 5. 尋找最佳 Confidence Threshold（F1）

讓你知道**最適合提交的 confidence 閾值**。

```bash
python main.py eval-threshold your_model.pt
```

程式會：

1. 用 conf=0.0003 取得所有候選框
2. 從 conf=0.05~0.95 掃描
3. 計算 F1 / Precision / Recall
4. 找出最佳 threshold
5. 輸出圖檔：
   `f1_score_analysis_XXXX.png`

輸出結果中會顯示：

```
最佳信心度閾值: 0.xx
Precision: 0.xxx
Recall:    0.xxx
```

比賽提交建議用此 threshold。



# 6. 批次推論與多閾值 AP 評估（competition_batch.py）

此工具用來：

* 針對測試資料一次推論
* 在多個 confidence 下產生對應 submit txt
* 計算 AP50、Precision、Recall
* 產生 summary.csv

執行方式：

```bash
python competition_batch.py 
```

主要程式碼：

### 功能摘要

| 機制                       | 說明                          |
| ------------------------ | --------------------------- |
| 單次最低 conf 推論             | 減少重複推論時間                    |
| 支援 conf range            | conf-min、conf-max、conf-step |
| 自動生成 predictions/*.txt   | 提交檔可直接使用                    |
| 自動計算 AP/Precision/Recall | 依 IoU=0.5                   |
| summary.csv              | 全部結果彙整                      |



# 最推薦的使用流程（最省時間）

以下是比賽最省時、最穩定的 pipeline：

### Step 1 — Colab 執行前處理

產生 dataset、labels、yaml

### Step 2 — 本機訓練

```bash
python main.py train
```

### Step 3 — 選出最佳模型

```bash
python main.py eval
```

### Step 4 — 找出最佳 confidence threshold

```bash
python main.py eval-threshold best_ap50.pt
```

### Step 5 — 批次跑全測試集，生成不同 conf submit

```bash
python competition_batch.py
```

### Step 6 — 查看 summary.csv，選出最適提交 txt



