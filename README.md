# AIoT DIC2：多元線性回歸（50 Startups & Boston Housing）

**線上網頁（GitHub Pages）：<https://jua91352.github.io/AIoT-DIC2-MultipleLR/>**

依 CRISP-DM 流程完成兩個多元線性回歸問題，並做成純前端的互動網頁：用下拉選單切換題目，也可以輸入任意 CSV 資料集網址。

![網頁截圖](docs/screenshot-top.jpg)

## 作業要求對照

- [x] 兩個問題都完成：Problem 1 — 50 Startups、Problem 2 — Boston Housing
- [x] 第二題直接套用 skill：[`.claude/skills/mlr-crisp-dm`](.claude/skills/mlr-crisp-dm/SKILL.md)
- [x] 加分 +1：有下拉選單呈現兩種 problem
- [x] 加分 +2：可任意輸入 Dataset 網址（GitHub 檔案頁網址會自動轉成 raw，也可上傳本機 CSV）
- [x] 部署到 GitHub Pages

## 結果摘要

訓練/測試切分 80/20（seed 42），網頁與 Python 報告使用相同切分，數字一致。

| | Problem 1 — 50 Startups | Problem 2 — Boston Housing |
|---|---|---|
| 目標 y | `Profit`（年度利潤） | `medv`（房價中位數，千美元） |
| 資料 | 50 筆、4 個特徵（`State` 為類別 → One-hot） | 506 筆、13 個特徵 |
| 做法 | 逐步完成 CRISP-DM 分析，再把流程封裝成 skill | 直接套用 skill `mlr-crisp-dm` |
| 全特徵模型 Test R² / RMSE | 0.9421 / 9,668.0 | 0.7435 / 4.0863 |
| 逆向淘汰（SL = 0.05） | 只保留 `R&D Spend`：0.9399 / 9,850.2 | 移除 `indus`、`age`：0.7458 / 4.0674 |
| 特徵選擇最佳 k（6 種方法比較） | k = 2：`R&D Spend` + `Marketing Spend`：0.9464 / 9,304.1 | k = 11（與逆向淘汰相同）：0.7458 / 4.0674 |
| 影響最大的特徵（標準化係數） | `R&D Spend`（+0.917） | `lstat`（−0.413）、`dis`（−0.345）、`rm`（+0.296） |
| 完整報告 | [reports/50_startups/report.md](reports/50_startups/report.md) | [reports/boston_housing/report.md](reports/boston_housing/report.md) |

**Problem 1 結論**：利潤幾乎由研發支出決定（p = 2.0e-17），行政支出與州別都不顯著。逆向淘汰只留下 `R&D Spend`；但依特徵數 k 比較時，加入 `Marketing Spend` 的 2 特徵模型測試誤差最低，是最精簡又準確的組合：

```
Profit = 46,174 + 0.81449 × R&D Spend + 0.024741 × Marketing Spend
```

**Problem 2 結論**：低社經人口比例 `lstat` 越高、房價越低；房間數 `rm` 越多、房價越高。`indus` 與 `age` 在控制其他變數後不顯著，移除後測試誤差反而略降。

## 網頁功能（CRISP-DM 六步驟）

1. **商業理解**：每個題目的問題定義與成功標準
2. **資料理解**：欄位型態、缺失值、資料預覽、敘述統計、類別分布、相關係數熱圖
3. **資料準備**：缺失值處理、One-hot（drop first 避免 dummy variable trap）、自動排除 ID／常數欄位、訓練/測試切分（比例與 seed 可調）
4. **建模**：OLS 係數、標準誤、t、p 值、95% 信賴區間、VIF、標準化係數；逆向淘汰（SL 可調）
5. **評估**：全特徵／逆向淘汰／最佳 k 三種模型比較、實際 vs 預測、殘差圖、特徵影響力、測試集殘差診斷，以及 6 種特徵選擇方法（逆向 p 值、Forward、RFE、最佳子集、Lasso、ElasticNet）依特徵數 k 比較 Test RMSE / R²
6. **部署**：輸入特徵值即時預測（含 95% 預測區間），可複製分享連結

網址參數：`?problem=startups`、`?problem=boston`，或 `?url=<CSV 網址>&target=<目標欄位>`。

## 第二題：直接套用 skill

Problem 1 的分析流程封裝成 Claude Code skill（[`SKILL.md`](.claude/skills/mlr-crisp-dm/SKILL.md) + [`mlr_pipeline.py`](.claude/skills/mlr-crisp-dm/scripts/mlr_pipeline.py)）。Problem 2 只要給資料網址和目標欄位，一行指令就產生完整報告：

```bash
python .claude/skills/mlr-crisp-dm/scripts/mlr_pipeline.py \
  --url https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv \
  --target medv --name "Problem 2 — Boston Housing" --slug boston_housing
```

在 Claude Code 裡也可以直接呼叫：`/mlr-crisp-dm <CSV 網址> <目標欄位>`。

## 專案結構

```
├── index.html                      # GitHub Pages 網頁
├── assets/
│   ├── css/style.css
│   └── js/
│       ├── mlr.js                  # 回歸計算引擎（OLS、檢定、VIF、特徵選擇），無外部套件
│       └── app.js                  # 介面：下拉選單、資料集網址、圖表（Chart.js）
├── .claude/skills/mlr-crisp-dm/    # 第二題套用的 skill
│   ├── SKILL.md
│   └── scripts/mlr_pipeline.py     # Python 版流程（numpy / scipy / scikit-learn / matplotlib）
├── reports/
│   ├── 50_startups/                # Problem 1 報告、results.json、圖表
│   └── boston_housing/             # Problem 2 報告、results.json、圖表
└── docs/                           # README 截圖
```

## 本機執行

```bash
python -m http.server 8000
```

開啟 <http://localhost:8000>。重新產生報告需要 `numpy pandas scipy scikit-learn matplotlib`。

## 驗證

網頁（JavaScript）與 Python（numpy / scikit-learn）是兩份獨立實作，使用相同的切分方法（mulberry32 + Fisher–Yates）。兩個資料集的係數、標準誤、p 值、R²、RMSE、VIF、逆向淘汰步驟與 6 種特徵選擇在每個 k 的結果全部一致（相對誤差 < 1e-12）；OLS 係數另外和 scikit-learn `LinearRegression` 交叉比對。

## 資料來源與備註

- 50 Startups：[Avik-Jain/100-Days-Of-ML-Code](https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/datasets/50_Startups.csv)
- Boston Housing：[selva86/datasets](https://github.com/selva86/datasets/blob/master/BostonHousing.csv)
- Boston Housing 的變數 `b` 由黑人人口比例計算而來，有倫理爭議（scikit-learn 1.2 起已移除此資料集）。這裡沿用作業指定的經典資料集；網頁上可以取消勾選 `b` 比較結果。
- 特徵數 k 依作業範例用測試集比較；實務上建議用交叉驗證選 k。

本作業使用 Claude Code 協作完成。
