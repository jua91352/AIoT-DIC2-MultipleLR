---
name: mlr-crisp-dm
description: 對任意 CSV 資料集（網址或本機路徑）執行 CRISP-DM 多元線性回歸分析，產生 Markdown 報告與圖表：型態判斷、One-hot、80/20 切分、OLS 係數檢定（t / p / 95% CI）、VIF、逆向淘汰、六種特徵選擇方法依 k 比較（逆向 p 值、Forward、RFE、最佳子集、Lasso、ElasticNet）。當使用者要對新資料集「做多元線性回歸 / multiple linear regression / 套用 DIC2 流程」時使用。
argument-hint: <CSV 網址或路徑> [目標欄位]
---

# 多元線性回歸（CRISP-DM）Skill

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

   - 預設：測試集比例 0.2、seed 42、逆向淘汰顯著水準 SL 0.05，和網頁版（`index.html`）完全相同，數字可以互相對照。
   - 不想放進模型的欄位用 `--exclude 欄位A 欄位B`；分號或 Tab 分隔的檔案用 `--sep ";"`。
   - 需要套件：numpy、pandas、scipy、scikit-learn、matplotlib。
4. **讀結果**：`reports/<slug>/report.md` 與 `results.json`，向使用者摘要：
   - 全特徵模型的測試集 R²、RMSE 與顯著特徵
   - 逆向淘汰保留了哪些特徵
   - 特徵選擇比較的最佳 k 與特徵組合
   - VIF > 10 的共線性警告
5. **（選用）加進網頁下拉選單**：在 `assets/js/app.js` 的 `PRESETS` 新增一筆（`label`、`url`、`target`、`business`、`notes`），重新整理網頁即可切換。

## 輸出

| 檔案 | 內容 |
|---|---|
| `reports/<slug>/report.md` | CRISP-DM 六步驟報告（商業理解 → 部署） |
| `reports/<slug>/results.json` | 係數、檢定、各模型指標、特徵選擇結果 |
| `reports/<slug>/figures/*.png` | 相關係數熱圖、實際 vs 預測、殘差圖、標準化係數、特徵數 k vs Test RMSE |

## 注意

- 報告裡依作業範例用測試集比較 k；實務上應該用交叉驗證選 k。
- 若資料含有敏感或有爭議的變數（例如 Boston Housing 的 `b`），在商業理解段落註明。
