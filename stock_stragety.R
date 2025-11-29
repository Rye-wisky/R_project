# ==============================================================================
# 课题名称：基于参数网格搜索与多因子过滤的稳健双均线策略 (修复版 V3)
# 修复说明：
# 1. 修复 Sharpe Ratio 提取失败的问题
# 2. 增加对“无交易/无结果”情况的防御性检查
# 3. 动态设定时间窗口：自动获取最近 5 年的数据进行分析
# 4. [新增] 自动保存热力图为 'grid.png'
# ==============================================================================

# 1. 环境准备
# ------------------------------------------------------------------------------
# 辅助函数：自动安装并加载包
ensure_package <- function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    if (!require(pkg, character.only = TRUE)) stop(paste("无法安装包:", pkg))
  }
}

ensure_package("quantmod")
ensure_package("TTR")
ensure_package("ggplot2")
ensure_package("PerformanceAnalytics")
ensure_package("scales")
ensure_package("reshape2") 

library(quantmod)
library(TTR)
library(ggplot2)
library(PerformanceAnalytics)
library(scales)
library(reshape2)

# ==============================================================================
# 2. 数据获取与预处理 (动态 5 年)
# ==============================================================================
stock_code <- "AAPL"
csv_file   <- "aapl_us_2025.csv"

# 动态计算开始时间：当前日期减去 5 年 (365 * 5)
end_date   <- Sys.Date()
start_date <- end_date - (365 * 5)

cat(sprintf(">>> 分析时间段: %s 至 %s (最近5年)\n", start_date, end_date))

# 尝试读取数据，如果不存在则下载
if (file.exists(csv_file)) {
  raw_data <- read.csv(csv_file)
  raw_data$Date <- as.Date(raw_data$Date)
  stock_data <- xts(raw_data[, c("Open", "High", "Low", "Close", "Volume")], order.by = raw_data$Date)
} else {
  message("本地文件不存在，正在下载 AAPL 数据...")
  tryCatch({
    getSymbols("AAPL", from = start_date, to = end_date)
    stock_data <- AAPL
    names(stock_data) <- c("Open", "High", "Low", "Close", "Volume", "Adjusted")
  }, error = function(e) {
    stop("无法下载数据。请检查网络或确认 'aapl_us_2025.csv' 文件是否存在。")
  })
}

# 确保数据只包含最近 5 年 (针对本地文件可能包含更久数据的情况)
stock_data <- stock_data[paste0(start_date, "/")]

# 统一使用收盘价
price <- Cl(stock_data)
# 计算基准收益（买入持有）
buy_hold_ret <- ROC(price, type = "discrete")
buy_hold_ret[is.na(buy_hold_ret)] <- 0

# ==============================================================================
# 3. 核心策略函数 (封装化，便于循环调用)
# ==============================================================================
run_strategy <- function(price_data, short_p, long_p, use_rsi_filter = FALSE) {
  
  # 必须保证长周期大于短周期，且参数不为 NA
  if (is.na(short_p) || is.na(long_p) || short_p >= long_p) return(NULL)
  
  # 1. 计算技术指标
  sma_short <- SMA(price_data, n = short_p)
  sma_long  <- SMA(price_data, n = long_p)
  rsi       <- RSI(price_data, n = 14)
  
  # 2. 生成基础信号 (1=持有, 0=空仓)
  signal <- ifelse(sma_short > sma_long, 1, 0)
  
  # 3. 应用 RSI 过滤器 (进阶逻辑)
  if (use_rsi_filter) {
    filter_condition <- ifelse(rsi < 75, 1, 0) # RSI < 75 才持有，防止高位接盘
    signal <- signal * filter_condition
  }
  
  # 处理 NA
  signal[is.na(signal)] <- 0
  
  # 4. 计算策略收益
  daily_ret <- ROC(price_data, type = "discrete")
  daily_ret[is.na(daily_ret)] <- 0
  
  strategy_ret <- Lag(signal, k = 1) * daily_ret
  strategy_ret[is.na(strategy_ret)] <- 0
  
  return(strategy_ret)
}

# ==============================================================================
# 4. 深度模块：参数网格搜索 (Grid Search)
# ==============================================================================
cat(">>> 开始执行参数网格搜索 (寻找最优均线组合)...\n")
cat(">>> 这可能需要一点时间，请耐心等待...\n")

short_range <- seq(5, 40, by = 5)   
long_range  <- seq(30, 100, by = 10) 

results_matrix <- matrix(NA, nrow = length(short_range), ncol = length(long_range))
rownames(results_matrix) <- short_range
colnames(results_matrix) <- long_range

for (i in 1:length(short_range)) {
  for (j in 1:length(long_range)) {
    s_p <- short_range[i]
    l_p <- long_range[j]
    
    if (s_p < l_p) {
      strat_ret <- run_strategy(price, s_p, l_p, use_rsi_filter = TRUE)
      
      if (!is.null(strat_ret) && any(strat_ret != 0)) {
        perf <- table.AnnualizedReturns(strat_ret)
        # [Fix] 健壮地查找 Sharpe Ratio 行
        sharpe_row_idx <- grep("Sharpe", rownames(perf))
        if (length(sharpe_row_idx) > 0) {
           results_matrix[i, j] <- perf[sharpe_row_idx[1], 1]
        } else {
           results_matrix[i, j] <- perf[3, 1]
        }
      }
    }
  }
}

# ==============================================================================
# 5. 结果可视化：参数性能热力图 (自动保存)
# ==============================================================================
melted_cormat <- melt(results_matrix, na.rm = TRUE)

if (nrow(melted_cormat) == 0) {
  stop("错误：网格搜索未产生任何有效结果。可能是数据问题导致所有策略均无交易或夏普比率计算失败。")
}

colnames(melted_cormat) <- c("Short_Period", "Long_Period", "Sharpe_Ratio")

p_heatmap <- ggplot(data = melted_cormat, aes(x = as.factor(Long_Period), y = as.factor(Short_Period), fill = Sharpe_Ratio)) +
  geom_tile() +
  scale_fill_gradient(low = "#f7fbff", high = "#08306b", name = "夏普比率") +
  theme_minimal() +
  labs(title = "策略参数热力图 (近5年)：寻找最优均线组合",
       subtitle = "颜色越深代表夏普比率越高",
       x = "长期均线周期",
       y = "短期均线周期")

# 1. 在屏幕上显示
print(p_heatmap)

# 2. 保存到本地文件 (新增功能)
ggsave("grid.png", plot = p_heatmap, width = 8, height = 6, dpi = 300)
cat(">>> [成功] 热力图已保存为 'grid.png'。\n")
cat(">>> 热力图已生成。请查看 Plot 面板或当前目录下的图片文件。\n")

# ==============================================================================
# 6. 最优策略回测与深度对比
# ==============================================================================

if (all(is.na(results_matrix))) {
  stop("错误：无法找到最优参数。所有组合的夏普比率均为 NA。")
}

best_idx <- which(results_matrix == max(results_matrix, na.rm = TRUE), arr.ind = TRUE)

best_short <- short_range[best_idx[1, 1]]
best_long  <- long_range[best_idx[1, 2]]

cat(sprintf("\n>>> 搜索完成！最优参数组合：短期=%d, 长期=%d\n", best_short, best_long))

# 运行对比
ret_default <- run_strategy(price, 20, 60, use_rsi_filter = FALSE)
ret_opt     <- run_strategy(price, best_short, best_long, use_rsi_filter = FALSE)
ret_filter  <- run_strategy(price, best_short, best_long, use_rsi_filter = TRUE)

compare_all <- merge(ret_default, ret_opt, ret_filter, buy_hold_ret)
colnames(compare_all) <- c("Default(20/60)", "Optimized_MA", "Optimized+RSI", "Benchmark")
compare_all <- na.omit(compare_all)

# ==============================================================================
# 7. 生成专业绩效报告
# ==============================================================================

cat("\n=======================================================\n")
cat("               策略最终绩效报告 (Performance)            \n")
cat("=======================================================\n")

# 显示并保存累积收益图
charts.PerformanceSummary(compare_all, 
                          main = "策略深度对比：默认 vs 优化参数 vs 复合策略",
                          colorset = c("gray", "blue", "red", "green"),
                          lwd = 2)

# 如果想保存绩效图，也可以使用 jpeg/png 函数包裹
# png("performance_summary.png", width=800, height=600)
# charts.PerformanceSummary(...)
# dev.off()

stats_table <- table.AnnualizedReturns(compare_all)
print(stats_table)

cat("\n>>> 分析完成。\n")