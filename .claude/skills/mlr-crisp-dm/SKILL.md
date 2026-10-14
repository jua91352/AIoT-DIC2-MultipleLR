---
name: mlr-crisp-dm
description: 對任意 CSV 資料集（網址或本機路徑）執行 CRISP-DM 多元線性回歸，並比較 5 種特徵選擇方法（Pearson、SelectKBest、RFE、Forward SFS、Lasso）在 k = 1 ~ K 的 Test RMSE / R²，產生 Markdown 報告與 1×2 雙軌跡折線圖：型態判斷、One-hot（drop_first）、train_test_split、OLS 係數檢定、VIF、LassoCV。當使用者要對新資料集「做多元線性回歸 / multiple linear regression / 特徵選擇比較 / 套用 DIC2 流程」時使用。
argument-hint: <CSV 網址或路徑> [目標欄位]
---

# 多元線性回歸 × 5 種特徵選擇方法（CRISP-DM）Skill

這個 skill 把 Problem 1（50 Startups）的分析流程封裝成可重複使用的步驟：給一個 CSV 和目標欄位，就能產生完整的 CRISP-DM 報告。
Problem 2（Boston Housing）就是直接套用這個 skill 完成的。

## 步驟

1. **確認輸入**：資料來源（`$ARGUMENTS` 的第一個參數，CSV 網址或本機路徑）與目標欄位 y（第二個參數）。
   - 沒給目標欄位時，先讀前幾列，預設用「最後一個數值欄位」，並向使用者說明選了哪一欄。
   - GitHub 檔案頁網址（`github.com/.../blob/...`）要先改成 `raw.githubusercontent.com/...`。
2. **寫一句商業問題**（Business Understanding）：這份資料要預測什麼、給誰用。
3. **執行 pipeline**（在專案根目錄）：

   ```bash
   python .claude/skills/mlr-crisp-dm/scripts/mlr_pipeline.py \
     --url "<CSV>" --target "<目標欄位>" \
     --name "<報告標題>" --slug <輸出資料夾名稱> \
     --business "<商業問題>"
   ```

   - 預設：`train_test_split(test_size=0.2, random_state=42)`、`KFold(5, shuffle=True, random_state=42)`，和網頁版（`index.html`）完全相同，數字可以互相對照。
   - 不想放進模型的欄位用 `--exclude 欄位A 欄位B`；分號或 Tab 分隔的檔案用 `--sep ";"`。
   - 需要套件：numpy、pandas、scipy、scikit-learn、matplotlib。
4. **讀結果**：`reports/<slug>/report.md` 與 `results.json`，向使用者摘要：
   - 5 種方法各自的特徵排序，以及在哪個 k 開始分歧
   - Test RMSE 最低的 (方法, k) 與特徵組合，和全特徵模型比較
   - LassoCV 原生收縮模型（★）的 α、保留的特徵與 Test RMSE
   - 5-fold CV 的最佳 k（測試集很小時特別重要）與 VIF > 10 的共線性警告
5. **（選用）加進網頁下拉選單**：在 `assets/js/app.js` 的 `PRESETS` 新增一筆（`label`、`title`、`url`、`target`、`business`、`notes`），重新整理網頁即可切換。

## 5 種方法（每個方法對 k = 1 ~ K 各選 k 個特徵，再用 OLS 在訓練集重新配適、於測試集評估）

| # | 方法 | 類型 | scikit-learn 實作 |
|---|---|---|---|
| 1 | Pearson | Filter | 訓練集 \|r(x, y)\| 由大到小 |
| 2 | SelectKBest | Filter | `SelectKBest(f_regression)`；F 是 \|r\| 的單調函數，排序與 Pearson 相同（兩線重疊是正確的） |
| 3 | RFE | Wrapper | `RFE(LinearRegression)`，標準化特徵、每輪移除 \|係數\| 最小者 |
| 4 | Forward SFS | Wrapper | `SequentialFeatureSelector(forward, cv=KFold(5, shuffle=True))` |
| 5 | Lasso | Embedded | `LassoCV(cv=KFold(5, shuffle=True))` 收縮後的 \|係數\| 由大到小；★ 為 LassoCV 原生收縮模型 |

## 輸出

| 檔案 | 內容 |
|---|---|
| `reports/<slug>/report.md` | CRISP-DM 六步驟報告（商業理解 → 部署） |
| `reports/<slug>/results.json` | 5 種方法在每個 k 的特徵與指標、LassoCV、全特徵 OLS 檢定 |
| `reports/<slug>/figures/feature_selection_5_schemes.png` | 1×2 雙軌跡折線圖（RMSE / R² vs k，5 種顏色 + 線型 + 標記、水平微偏移、★） |
| `reports/<slug>/figures/*.png` | 相關係數熱圖、最佳模型的實際 vs 預測與殘差圖 |

## 注意

- LassoCV 使用 `tol=1e-10` 讓座標下降收斂到精確解；預設 `tol=1e-4` 在相鄰 α 的 CV 誤差幾乎同分時，可能選到隔壁的 α。
- 依作業範例以測試集比較 k；測試集很小時，報告會一併列出 5-fold CV 的結果供參考。
- 若資料含有敏感或有爭議的變數（例如 Boston Housing 的 `b`），在商業理解段落註明。
