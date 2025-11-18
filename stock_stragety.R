# ==============================================================================
# 课题名称：基于R语言的A股/美股双均线交易策略回测分析
# 数据来源：本地文件 "aapl_us_2025.csv" (真实历史数据)
# ==============================================================================

# 1. 加载必要的包
library(quantmod)
library(TTR)
library(ggplot2)
library(PerformanceAnalytics)
library(scales) 

# 2. 设定参数
stock_code   <- "AAPL"
short_period <- 20          # 短期均线 (20天)
long_period  <- 60          # 长期均线 (60天)
csv_file     <- "aapl_us_2025.csv" # 你上传的文件名

# 设置时间范围：最近5年 (2020年 - 2024/2025年)
# 这样可以过滤掉 1984 年那些太久远的数据
start_date   <- "2020-01-01" 

# ==============================================================================
# 第一阶段：读取并清洗数据
# ==============================================================================

# 尝试自动设置工作目录到脚本所在位置
try({
  current_path <- dirname(rstudioapi::getSourceEditorContext()$path)
  setwd(current_path)
  cat(paste("工作目录已设置:", current_path, "\n"))
}, silent = TRUE)

cat(paste(">>> 正在读取文件:", csv_file, "...\n"))

if (!file.exists(csv_file)) {
  stop(paste("错误：找不到文件", csv_file, "\n请确保该 CSV 文件和此 R 脚本在同一个文件夹内。"))
}

# 读取 CSV
raw_data <- read.csv(csv_file)

# 1. 转换日期格式
raw_data$Date <- as.Date(raw_data$Date)

# 2. 转换为 xts 时间序列对象
# 这里的 CSV 只有 Close 列，没有 Adj Close，直接用 Close
stock_data <- xts(raw_data[, c("Open", "High", "Low", "Close", "Volume")], order.by = raw_data$Date)

# 3. 截取最近 5 年的数据
# xts 支持直接用字符串范围筛选，例如 "2020-01-01/" 表示从这一天直到最后
stock_data <- stock_data[paste0(start_date, "/")]

cat(paste(">>> 数据加载成功！\n"))
cat(paste(">>> 时间范围:", start(stock_data), "至", end(stock_data), "\n"))
cat(paste(">>> 总交易日数:", nrow(stock_data), "\n"))

# 提取收盘价用于策略
price <- stock_data$Close
names(price) <- "Close"

# ==============================================================================
# 第二阶段：策略构建
# ==============================================================================

# 1. 计算均线
sma_short <- SMA(price, n = short_period)
sma_long  <- SMA(price, n = long_period)

# 合并数据
strategy_df <- merge(price, sma_short, sma_long)
colnames(strategy_df) <- c("Close", "SMA_Short", "SMA_Long")
strategy_df <- na.omit(strategy_df) # 去除计算初期的 NA

# 2. 生成信号
# 仓位逻辑: 短线 > 长线 = 持有(1)，否则空仓(0)
strategy_df$Position <- ifelse(strategy_df$SMA_Short > strategy_df$SMA_Long, 1, 0)

# 交易信号: 仓位变化 (1=买入, -1=卖出)
strategy_df$Signal <- diff(strategy_df$Position)
strategy_df$Signal[1] <- 0

# ==============================================================================
# 第三阶段：回测与评估
# ==============================================================================

# 1. 计算基准收益 (Buy & Hold)
daily_ret <- ROC(strategy_df$Close, type = "discrete")
daily_ret[1] <- 0 

# 2. 计算策略收益 (T+1 交易机制)
# Lag(Position, 1) 模拟“次日开盘操作”，即今天的收益取决于昨天的信号
strategy_ret <- Lag(strategy_df$Position, k = 1) * daily_ret
strategy_ret[is.na(strategy_ret)] <- 0 

# 合并对比
ret_comp <- merge(daily_ret, strategy_ret)
colnames(ret_comp) <- c("Benchmark", "Strategy")

# 3. 输出绩效表
cat("\n=== 策略绩效评估 (最近5年) ===\n")
print(table.AnnualizedReturns(ret_comp))
cat("\n=== 最大回撤 ===\n")
print(maxDrawdown(ret_comp))

# ==============================================================================
# 第四阶段：可视化
# ==============================================================================

# 准备绘图数据
df_plot <- data.frame(Date = index(strategy_df), coredata(strategy_df))

# 图1: 交易信号图
p1 <- ggplot(df_plot, aes(x = Date)) +
  geom_line(aes(y = Close, color = "Price"), linewidth = 0.5, alpha = 0.6) +
  geom_line(aes(y = SMA_Short, color = "SMA20"), linewidth = 0.8) +
  geom_line(aes(y = SMA_Long, color = "SMA60"), linewidth = 0.8) +
  # 买入点 (红色正三角)
  geom_point(data = subset(df_plot, Signal == 1),
             aes(y = SMA_Short), shape = 17, color = "red", size = 3) +
  # 卖出点 (绿色倒三角)
  geom_point(data = subset(df_plot, Signal == -1),
             aes(y = SMA_Short), shape = 25, color = "green", size = 3, fill="green") +
  scale_color_manual(values = c("Price" = "gray50", "SMA20" = "orange", "SMA60" = "blue")) +
  labs(title = paste("双均线策略交易信号图 -", stock_code),
       subtitle = "红色三角=买入(金叉), 绿色倒三角=卖出(死叉)",
       y = "价格", x = "日期") +
  theme_minimal() +
  theme(legend.title = element_blank(), legend.position = "top")

print(p1)

# 图2: 累计收益对比图
# 计算累计收益
cum_ret_benchmark <- cumprod(1 + ret_comp$Benchmark) - 1
cum_ret_strategy  <- cumprod(1 + ret_comp$Strategy) - 1

df_ret_plot <- data.frame(Date = index(cum_ret_benchmark), 
                          Benchmark = coredata(cum_ret_benchmark),
                          Strategy = coredata(cum_ret_strategy))

p2 <- ggplot(df_ret_plot, aes(x = Date)) +
  geom_line(aes(y = Benchmark, color = "基准 (买入持有)"), linewidth = 1) +
  geom_line(aes(y = Strategy, color = "双均线策略"), linewidth = 1) +
  geom_area(aes(y = Strategy), fill = "blue", alpha = 0.1) + # 蓝色阴影填充
  scale_y_continuous(labels = percent) +
  scale_color_manual(values = c("基准 (买入持有)" = "gray", "双均线策略" = "red")) +
  labs(title = paste("策略收益 vs 基准收益 -", stock_code),
       subtitle = "累计收益率对比 (最近5年)",
       y = "累计收益率", x = "日期") +
  theme_minimal() +
  theme(legend.title = element_blank(), legend.position = "top")

print(p2)

cat("\nDone\n")