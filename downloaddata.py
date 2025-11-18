import yfinance as yf
import pandas as pd
import os

# ================= 配置参数 =================
stock_code = "600519.SS"       # 股票代码
start_date = "2020-01-01" # 开始日期
end_date   = "2025-11-17" # 结束日期
filename   = "stock_data.csv"

print(f"正在启动 Python 下载器...")
print(f"目标: {stock_code} | 时间: {start_date} 至 {end_date}")

try:
    # 1. 使用 yfinance 下载数据
    # auto_adjust=False 确保我们得到原始的 'Close' 和 'Adj Close'，虽然我们主要用 Close
    df = yf.download(stock_code, start=start_date, end=end_date, progress=True, auto_adjust=False)

    if df.empty:
        print("❌ 下载失败：返回数据为空。请检查网络或股票代码。")
    else:
        # 2. 数据清洗
        # yfinance 下载的数据索引是日期，我们需要把它变成一列，方便 CSV 读取
        df.reset_index(inplace=True)
        
        # 确保日期格式统一 (去掉时分秒)
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

        # 处理多级索引问题 (yfinance 新版特性)
        # 如果列名是元组格式 (Price, Ticker)，简化为 Price
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 3. 保存为 CSV
        # index=False 表示不保存默认的数字索引 0,1,2...
        current_dir = os.getcwd()
        file_path = os.path.join(current_dir, filename)
        
        df.to_csv(file_path, index=False)
        
        print(f"✅ 下载成功！")
        print(f"📊 数据行数: {len(df)}")
        print(f"📂 文件保存在: {file_path}")
        print("👉 现在你可以直接运行 R 脚本了！")

except Exception as e:
    print(f"❌ 发生错误: {e}")