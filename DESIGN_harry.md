# Smart Drawer Design Document

## 1. 目的

本系統以 IMX93 為邊緣運算平台，搭配俯角 RGB-D 相機，記錄抽屜內「本次被放入或取出」的物品及其所在層位。使用者之後可透過語音或畫面查詢物品位置。

本設計的核心不是在每次打開抽屜時盤點整層物品，而是以兩次穩定的深度圖找出操作造成的變更區域，只對該區域執行辨識。

## 2. 範圍與非目標

### 系統要做的事

- 在抽屜打開並靜止後，判定目前是哪一層抽屜。
- 在使用者放入或取出物品後，偵測深度圖的局部變化。
- 只裁切變更區域進行物件偵測／分類。
- 判斷該區域的操作為放入或取出，並更新本機資料庫。
- 查詢物品名稱、同義詞或語音指令時，回覆物品所在的抽屜層位。

### 系統不做的事

- 不在每次開抽屜時對整層物品做 object detection。
- 不維護每一幀的完整抽屜 inventory。
- 不將全層影像送往雲端進行辨識；正常流程應完全在 edge 端完成。

## 3. 關鍵概念

### 兩個不同的深度事件

1. **開啟事件**：抽屜被拉出時，深度畫面先產生變化。待抽屜停止後，取得操作前的第一組穩定 RGB-D 影像 `S_before`。此影像用於判定抽屜層位，並作為本次操作前的 baseline。
2. **物品操作事件**：使用者放入或取出物品時，深度畫面再次變化。待畫面重新穩定後，取得操作後的第二組穩定 RGB-D 影像 `S_after`。

系統比較的是 `S_before.depth` 與 `S_after.depth`，而非把兩張影像中的所有物品各自重新辨識一次。

## 4. 使用流程與狀態機

```text
IDLE
  │ 偵測到抽屜開啟
  ▼
WAIT_OPEN_STABLE
  │ 深度連續穩定
  ▼
CAPTURE_BEFORE ──→ 判定 drawer_level，保存 S_before
  │
  │ 偵測到人手／物品操作造成的深度變化
  ▼
WAIT_CHANGE_STABLE
  │ 深度再次連續穩定
  ▼
CAPTURE_AFTER ──→ 保存 S_after
  │
  ▼
DEPTH_DIFF → 變更區域 crop → 偵測／分類 → 更新資料庫
  │
  └──────────────────────────────────────────→ IDLE
```

### 4.1 開啟後的穩定判定

只在固定的「抽屜幾何 ROI」上計算相鄰深度幀的變化，避免抽屜外的人員移動干擾。當下列條件持續 `N_stable` 幀時，視為穩定：

```text
changed_pixel_ratio(frame[t], frame[t-1]) < T_stable_ratio
median_abs_depth_delta(frame[t], frame[t-1]) < T_stable_depth
```

`N_stable`、`T_stable_ratio` 與 `T_stable_depth` 必須依相機雜訊、安裝高度與抽屜尺寸校正，不應硬編碼在程式中。

### 4.2 抽屜層位判定

安裝／初始化時，讓每一層抽屜依序完全拉開，為每層建立 layer profile。profile 應取不易被收納物遮擋的結構特徵，例如抽屜前緣、側板、軌道或固定的幾何 ROI，而不是抽屜內物品。

開啟後從 `S_before.depth` 擷取相同特徵，和各 layer profile 比對，取得：

```text
drawer_level = argmin(profile_distance(current_profile, layer_profile[i]))
```

若最佳結果低於可信度門檻，系統不寫入資料庫，並要求重新開啟或人工確認層位。

## 5. 變更偵測與裁切

### 5.1 深度圖前處理

- 對齊 RGB 與 depth 座標系；兩次 capture 使用相同抽屜 ROI。
- 排除無效深度值、鏡面反射區與抽屜結構的固定遮罩。
- 對深度圖做時間中值／空間中值濾波，減少感測器雜訊。
- 如有微小相機或抽屜位移，先以抽屜固定結構做 registration，再進行差分。

### 5.2 變更遮罩

在對齊後的抽屜 ROI 內計算：

```text
ΔD(x, y) = D_after(x, y) - D_before(x, y)
change_mask(x, y) = abs(ΔD(x, y)) > T_change_depth
```

對 `change_mask` 做去雜訊、形態學閉運算與連通元件分析。面積小於 `T_min_area` 的元件視為雜訊；每個其餘元件形成一個候選變更區域。將 bounding box 加上 context margin 後，投影到 RGB 影像裁切。

### 5.3 只對變更區域辨識

- **放入**：以 `S_after.rgb` 的 crop 做偵測／分類，因物品仍在畫面中。
- **取出**：以 `S_before.rgb` 的 crop 做偵測／分類，因物品可能已不在 `S_after.rgb` 中。
- 若該位置已有可信的資料庫物品紀錄，可用位置關聯協助辨識取出的物品；但不應將其視為唯一依據。

這項設計尤其重要：若只拍攝操作後影像，取出的物品已經離開畫面，無法可靠辨識其類別。

### 5.4 放入／取出方向判定

相機座標校正後，在候選區域計算深度差的 robust median 或 percentile：

- 新物品使表面更靠近相機（量測深度變小）時，判定為 **放入**。
- 原物品消失、露出較遠的背景／抽屜底部（量測深度變大）時，判定為 **取出**。

實際正負號要以該深度感測器的定義與安裝姿態驗證。若差異符號混雜、候選區太大、或分類信心不足，事件標記為 `REVIEW_REQUIRED`，不自動更新 inventory。

## 6. 邊緣端推論架構

```text
RGB-D camera
  → 深度穩定判定與抽屜層位辨識
  → 兩張穩定 RGB-D capture
  → depth registration 與差分
  → 變更區域 crop
  → 輕量化 object detector / classifier
  → SQLite inventory 與 event log
  → 本機 UI / 語音查詢
```

為符合 IMX93 資源限制：

- 深度差分、ROI、mask 與連通元件分析優先使用傳統影像處理。
- 偵測模型只接受候選 crop，並採用可在目標 NPU／CPU 上執行的量化模型。
- 模型輸出使用 confidence threshold；低信心結果保留影像 crop 與事件供人工確認。
- 全層影像不需要反覆推論，也不應成為常態資料庫內容。

## 7. 資料模型

### drawers

| 欄位 | 說明 |
| --- | --- |
| `drawer_id` | 抽屜櫃識別碼 |
| `level_id` | 抽屜層位 |
| `layer_profile` | 初始化時建立的結構深度特徵 |
| `calibration_version` | 相機／層位校正版本 |

### inventory_items

| 欄位 | 說明 |
| --- | --- |
| `item_instance_id` | 物品實例識別碼 |
| `canonical_item_id` | 標準物品類別或向量資料庫 key |
| `drawer_id`, `level_id` | 目前位置 |
| `last_bbox` | 最近一次變更時的抽屜座標位置 |
| `confidence` | 分類與位置可信度 |
| `status` | `PRESENT`、`REMOVED` 或 `REVIEW_REQUIRED` |

### change_events

| 欄位 | 說明 |
| --- | --- |
| `event_id`, `timestamp` | 事件識別與時間 |
| `drawer_id`, `level_id` | 發生位置 |
| `operation` | `INSERT`、`REMOVE`、`UNKNOWN` |
| `change_bbox` | 深度差分產生的區域 |
| `item_instance_id` | 關聯的物品實例（可為空） |
| `before_depth_ref`, `after_depth_ref` | 除錯用的深度快照參考 |
| `model_confidence` | 推論可信度 |

## 8. 查詢與同義詞

每個 `canonical_item_id` 可維護名稱、別名與文字 embedding。例如「膠帶」、「透明膠帶」與口語名稱可映射到同一 canonical item。語音輸入經本機關鍵字／語音辨識後，先做名稱與 embedding matching，再查詢 `inventory_items` 中狀態為 `PRESENT` 的紀錄，回傳抽屜層位。

## 9. 異常與保護機制

- 抽屜未完全靜止：不建立 baseline，也不產生變更事件。
- 一次出現多個變更區：各 crop 獨立辨識；若區域互相重疊或無法分離，標記人工確認。
- 人手仍在 ROI：延後 capture，直到人手／大範圍運動消失。
- 深度失效或反光：事件記為失敗，不用 RGB 單獨猜測 inventory 變更。
- 層位比對不確定：不寫入任何層位資料。
- 資料庫更新採 transaction：事件與 inventory 更新需同時成功，避免只留下半筆紀錄。

## 10. 驗收標準

1. 抽屜打開且穩定後，系統能在校正過的櫃體上正確辨識層位。
2. 未發生物品操作時，系統不對整層抽屜執行物件偵測，也不新增 inventory event。
3. 放入單一物品後，系統只對深度差異 crop 推論並新增一筆 `INSERT` event。
4. 取出單一物品後，系統能由操作前 crop 或既有位置關聯辨識／更新一筆 `REMOVE` event。
5. 深度變化低於設定門檻或畫面未穩定時，不更新資料庫。
6. 低信心或無法判定方向的事件，不自動宣稱物品已放入或取出。

## 11. 實作前需確認的參數

- 相機型號、深度精度、視角、安裝高度與俯角。
- 抽屜尺寸、層數、可拉出距離，以及每層是否有可穩定辨識的結構特徵。
- `T_stable_depth`、`T_stable_ratio`、`N_stable`、`T_change_depth`、`T_min_area` 的實測校正方式。
- 目標物品類別數、物品最小尺寸、堆疊／遮擋容忍度。
- IMX93 上實際可用的 NPU runtime、模型格式與量化限制。
- 初始 inventory 的建立策略：手動登錄、逐次操作累積，或一次性人工盤點。系統不應把「完整自動盤點」誤當成此事件式設計的必要條件。

## 12. 未來擴充

- 為多台抽屜櫃加上 `device_id`，同步到統一的向量資料庫或服務。
- 將低信心事件提供 UI 審核，再回饋分類器／別名表。
- 支援多物件同時放入或取出的 instance association。
- 依隱私需求設定深度快照與 RGB crop 的保存期限。
