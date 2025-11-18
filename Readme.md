# 基于R语言的双均线交易策略回测

### Dual Moving Average Crossover Strategy Backtest

## 📖 项目简介

本项目是 R 语言课程大作业，旨在利用 R 语言实现经典的**双均线交易策略** (Dual SMA Crossover)。

项目选取了 **苹果公司 (AAPL)** 作为研究对象，基于真实的日度历史数据，通过计算短期均线 (20日) 和长期均线 (60日) 的交叉情况来生成买卖信号，并回测了该策略在最近五年 (2020-2025) 的表现。

## 📂 文件结构

- **`stock_strategy.R`**: 核心代码文件。包含数据读取、数据清洗、策略构建、回测逻辑及可视化绘图的所有代码。
- **`aapl_us_2025.csv`**: 数据文件。包含 AAPL 的历史交易数据（Open, High, Low, Close, Volume）。
- **`实验报告.md`**: (可选) 项目的分析报告及结论。

## 🛠️ 环境依赖

运行本项目需要安装以下 R 包。如果你尚未安装，请在 R 控制台运行以下命令：

```
install.packages(c("quantmod", "TTR", "ggplot2", "PerformanceAnalytics", "scales"))
```

- `quantmod`: 金融数据处理框架
- `TTR`: 技术指标计算 (SMA均线)
- `ggplot2`: 高级绘图
- `PerformanceAnalytics`: 策略绩效评估 (年化收益、夏普比率等)
- `scales`: 图表坐标轴格式化

## 🚀 如何运行

1. **准备文件**：确保 `stock_strategy.R` 和 `aapl_us_2025.csv` 位于电脑上的**同一个文件夹**内。
2. **打开代码**：使用 RStudio 打开 `stock_strategy.R`。
3. **运行脚本**：点击 RStudio 右上角的 `Source` 按钮，或者全选代码 (`Ctrl+A`) 后点击 `Run`。
4. **查看结果**：
   - **控制台 (Console)**：将输出策略的年化收益率、最大回撤等绩效指标。
   - **绘图区 (Plots)**：将生成两张图表（交易信号图、累计收益对比图）。

## 📈 策略逻辑简述

- **短期均线 (SMA20)**: 反应价格短期趋势。
- **长期均线 (SMA60)**: 反应价格长期趋势。
- **买入信号 (金叉)**: 当 SMA20 **上穿** SMA60 时，买入持有。
- **卖出信号 (死叉)**: 当 SMA20 **下穿** SMA60 时，卖出空仓。

*Created for R Language Course Project*