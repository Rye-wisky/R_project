import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # 用于绘制热力图
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import os

# ==============================================================================
# 1. 全局配置与数据读取 (本地模式)
# ==============================================================================
STOCK_CODE = "AAPL"
CSV_FILE = "aapl_us_2025.csv"
START_DATE = "2020-01-01"  # 与 R 脚本保持一致

# AI 参数
SEQ_LENGTH = 60    # 回顾窗口
HIDDEN_DIM = 64
EPOCHS = 50
LR = 0.001

# 设置随机种子以保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

print(f">>> [1/6] 正在读取本地文件 {CSV_FILE}...")

try:
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"找不到文件: {CSV_FILE}")

    # 读取 CSV
    df = pd.read_csv(CSV_FILE)
    
    # 清洗与格式化
    if 'Date' not in df.columns:
        raise ValueError("CSV 文件缺少 'Date' 列")
        
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    # 筛选时间段 (从 2020-01-01 开始)
    df = df[df.index >= pd.Timestamp(START_DATE)]
    
    # 检查必要列
    req_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in req_cols):
        raise ValueError(f"CSV 文件缺少必要列: {req_cols}")
        
    df = df[req_cols].dropna()
    print(f"    成功加载数据: {len(df)} 行 ({df.index[0].date()} 至 {df.index[-1].date()})")

except Exception as e:
    print(f"数据读取失败: {e}")
    print(">>> 警告:正在生成模拟数据以演示代码逻辑...")
    dates = pd.date_range(start=START_DATE, end=pd.Timestamp.today())
    df = pd.DataFrame(index=dates)
    # 生成带趋势的随机漫步数据
    df['Close'] = 100 + np.cumsum(np.random.randn(len(dates)) + 0.05) 
    df['Open'] = df['High'] = df['Low'] = df['Close']
    df['Volume'] = 1000000

# ==============================================================================
# 2. 特征工程 & 数据集划分
# ==============================================================================
def calculate_indicators(data):
    df = data.copy()
    df['Returns'] = df['Close'].pct_change()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    
    # 波动率
    df['Volatility'] = df['Returns'].rolling(20).std()
    
    df.dropna(inplace=True)
    return df

df_full = calculate_indicators(df)

# 划分训练集 (80%) 和 测试集 (20%)
# 关键原则：网格搜索和AI训练只能在训练集上做，测试集必须是完全未知的
split_idx = int(len(df_full) * 0.8)
train_data = df_full.iloc[:split_idx].copy()
test_data = df_full.iloc[split_idx:].copy()

print(f">>> [2/6] 数据划分完成 | 训练集: {len(train_data)} 天 | 测试集: {len(test_data)} 天")

# ==============================================================================
# 3. 传统策略优化模块 (Grid Search)
# ==============================================================================
print(">>> [3/6] 正在执行网格搜索 (寻找最优均线组合)...")

def run_ma_strategy(data, short_w, long_w, rsi_filter=False):
    """计算策略的夏普比率和每日收益"""
    if short_w >= long_w: return -np.inf, None
    
    price = data['Close']
    sma_s = price.rolling(window=short_w).mean()
    sma_l = price.rolling(window=long_w).mean()
    
    # 基础信号
    signal = np.where(sma_s > sma_l, 1, 0)
    
    # RSI 过滤器
    if rsi_filter:
        rsi = data['RSI']
        # 只有当 RSI < 75 (非严重超买) 时才持仓
        signal = np.where((signal == 1) & (rsi < 75), 1, 0)
        
    # 计算收益 (shift(1) 代表昨天信号决定今天持仓)
    signal = pd.Series(signal, index=data.index).shift(1).fillna(0)
    strategy_ret = signal * data['Returns']
    
    # 计算年化夏普比率
    if strategy_ret.std() == 0: return 0, strategy_ret
    sharpe = (strategy_ret.mean() / strategy_ret.std()) * np.sqrt(252)
    return sharpe, strategy_ret

# 网格搜索范围
short_range = range(5, 45, 5)
long_range = range(30, 105, 10)
results = np.zeros((len(short_range), len(long_range)))
best_sharpe = -np.inf
best_params = (20, 60)

for i, s in enumerate(short_range):
    for j, l in enumerate(long_range):
        # 注意：只在训练集上搜索！
        sharpe, _ = run_ma_strategy(train_data, s, l)
        results[i, j] = sharpe
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = (s, l)

print(f"    最优参数找到: 短期={best_params[0]}, 长期={best_params[1]} (训练集Sharpe: {best_sharpe:.2f})")

# 绘制热力图
plt.figure(figsize=(10, 6))
sns.heatmap(results, xticklabels=long_range, yticklabels=short_range, cmap="coolwarm", annot=True, fmt=".2f")
plt.title(f"Grid Search Sharpe Ratio (Training Data)")
plt.xlabel("Long Window")
plt.ylabel("Short Window")
plt.show()

# ==============================================================================
# 4. AI 策略模块 (GRU)
# ==============================================================================
print(">>> [4/6] 正在训练 GRU 模型...")

class StockGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(StockGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

# 数据准备
feature_cols = ['Returns', 'RSI', 'MACD', 'Volatility']
scaler = StandardScaler()

# 仅在训练集上拟合 scaler，防止未来信息泄露
X_train_scaled = scaler.fit_transform(train_data[feature_cols])
X_test_scaled = scaler.transform(test_data[feature_cols])

# 制作标签: 预测明天涨跌 (1=涨, 0=跌)
y_train = (train_data['Close'].shift(-1) > train_data['Close']).astype(int).dropna().values
y_test_raw = (test_data['Close'].shift(-1) > test_data['Close']).astype(int) # Series for alignment

# 截断 X 以匹配 y (因为 shift(-1) 会产生 NaN)
X_train_scaled = X_train_scaled[:-1]
X_test_scaled = X_test_scaled[:-1]

def create_seq(X, y, seq_len):
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(xs), np.array(ys)

# 生成序列
X_train_seq, y_train_seq = create_seq(X_train_scaled, y_train, SEQ_LENGTH)
X_test_seq, _ = create_seq(X_test_scaled, np.zeros(len(X_test_scaled)), SEQ_LENGTH) # y for test mostly placeholder here

# PyTorch 转换
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_t = torch.tensor(X_train_seq, dtype=torch.float32).to(device)
y_t = torch.tensor(y_train_seq, dtype=torch.long).to(device)

model = StockGRU(len(feature_cols), HIDDEN_DIM, 2).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# 训练循环
model.train()
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    out = model(X_t)
    loss = criterion(out, y_t)
    loss.backward()
    optimizer.step()

# ==============================================================================
# 5. 综合回测 (Test Set Evaluation)
# ==============================================================================
print(">>> [5/6] 正在执行多策略对比回测...")

# 统一回测时间段 (扣除序列长度，因为 AI 需要预热)
# 这里的 test_df 是我们最终画图的数据基底
test_df = test_data.iloc[SEQ_LENGTH:-1].copy()

# 1. 基准 (Buy & Hold)
test_df['Ret_BH'] = test_df['Returns']

# 2. 默认双均线 (20/60)
_, ret_def = run_ma_strategy(test_data, 20, 60)
test_df['Ret_Default'] = ret_def.loc[test_df.index]

# 3. 优化参数双均线 (Best Params)
_, ret_opt = run_ma_strategy(test_data, best_params[0], best_params[1])
test_df['Ret_Optimized'] = ret_opt.loc[test_df.index]

# 4. 优化参数 + RSI 过滤
_, ret_flt = run_ma_strategy(test_data, best_params[0], best_params[1], rsi_filter=True)
test_df['Ret_Opt_RSI'] = ret_flt.loc[test_df.index]

# 5. AI 策略预测
model.eval()
with torch.no_grad():
    X_test_t = torch.tensor(X_test_seq, dtype=torch.float32).to(device)
    preds = model(X_test_t)
    signals = torch.argmax(preds, dim=1).cpu().numpy()

# 对齐 AI 信号到 test_df
# 注意: X_test_seq 生成的预测对应的是序列末端的"下一天"
# 我们的 test_df 已经截去了前 SEQ_LENGTH 天，所以长度应该大致匹配
if len(signals) > len(test_df):
    signals = signals[:len(test_df)]
elif len(signals) < len(test_df):
    test_df = test_df.iloc[:len(signals)]

test_df['Pos_AI'] = signals
# 昨天的预测信号 * 今天的收益
test_df['Ret_AI'] = test_df['Pos_AI'].shift(1).fillna(0) * test_df['Returns']

# ==============================================================================
# 6. 结果可视化
# ==============================================================================
print(">>> [6/6] 生成最终对比图...")

# 计算累计收益
cum_ret = (1 + test_df[['Ret_BH', 'Ret_Default', 'Ret_Optimized', 'Ret_Opt_RSI', 'Ret_AI']]).cumprod()

plt.figure(figsize=(12, 7))
plt.plot(cum_ret.index, cum_ret['Ret_BH'], label='Benchmark (Buy&Hold)', color='grey', linestyle='--', alpha=0.5)
plt.plot(cum_ret.index, cum_ret['Ret_Default'], label='MA Default (20/60)', color='blue', alpha=0.6)
plt.plot(cum_ret.index, cum_ret['Ret_Optimized'], label=f'MA Optimized {best_params}', color='orange', alpha=0.8)
plt.plot(cum_ret.index, cum_ret['Ret_Opt_RSI'], label='MA Opt + RSI Filter', color='green', linewidth=1.5)
plt.plot(cum_ret.index, cum_ret['Ret_AI'], label='AI (GRU) Strategy', color='red', linewidth=2.5)

plt.title(f'Comprehensive Strategy Comparison: Traditional vs AI ({STOCK_CODE})')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True, alpha=0.3)

# 打印最终统计
final_ret = cum_ret.iloc[-1] - 1
print("\n" + "="*50)
print(f"最终绩效统计 (测试集: {test_df.index[0].date()} ~ {test_df.index[-1].date()})")
print("="*50)
print(f"1. 买入持有 (Benchmark) : {final_ret['Ret_BH']:.2%}")
print(f"2. 默认双均线 (20/60)   : {final_ret['Ret_Default']:.2%}")
print(f"3. 优化双均线 {best_params}: {final_ret['Ret_Optimized']:.2%}")
print(f"4. 优化 + RSI 过滤      : {final_ret['Ret_Opt_RSI']:.2%}")
print(f"5. AI (GRU) 深度学习    : {final_ret['Ret_AI']:.2%}")
print("="*50)

plt.show()