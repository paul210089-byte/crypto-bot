import os
import ccxt
import yfinance as yf
import pandas as pd
from google import genai
import feedparser
import http.server
import socketserver

# 自定義計算 EMA (不依賴 pandas_ta)
def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

# 自定義計算 RSI (不依賴 pandas_ta)
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_macro_and_stock_data():
    print("[1/4] 正在抓取美股與總經市場指標...")
    tickers = {
        "S&P 500": "^GSPC",
        "Nasdaq": "^IXIC",
        "VIX 恐慌指數": "^VIX",
        "美元指數 (DXY)": "DX-Y.NYB",
        "美國 10 年期公債殖利率": "^TNX"
    }
    
    macro_summary = ""
    try:
        data = yf.download(list(tickers.values()), period="5d", interval="1d", progress=False)['Close']
        for name, ticker in tickers.items():
            if ticker in data.columns:
                series = data[ticker].dropna()
                if not series.empty:
                    latest_val = series.iloc[-1]
                    prev_val = series.iloc[-2] if len(series) > 1 else latest_val
                    change_pct = ((latest_val - prev_val) / prev_val) * 100
                    macro_summary += f"- {name}: {latest_val:.2f} (近期漲跌幅: {change_pct:+.2f}%)\n"
    except Exception as e:
        macro_summary += f"抓取美股/總經數據時發生錯誤: {e}\n"
        
    return macro_summary

def get_targeted_news():
    print("[2/4] 正在抓取總經、CPI、戰爭、川普與 FED 相關即時新聞...")
    rss_urls = [
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://feeds.feedburner.com/reuters/businessNews"
    ]
    
    target_keywords = ['cpi', 'inflation', 'fed', 'powell', 'trump', 'war', 'conflict', 'geopolitical', 'interest rate', 'rate cut', 'rate hike']
    collected_news = []
    
    try:
        for url in rss_urls:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                title = entry.title
                summary = getattr(entry, 'summary', '')
                content_to_check = (title + " " + summary).lower()
                
                if any(keyword in content_to_check for keyword in target_keywords):
                    if title not in collected_news:
                        collected_news.append(title)
    except Exception as e:
        collected_news.append(f"抓取新聞發生異常: {e}")
        
    news_summary = ""
    if collected_news:
        for title in collected_news[:10]:
            news_summary += f"- {title}\n"
    else:
        news_summary = "- 近期 RSS 未過濾到特定關鍵字即時頭條，請以市場既有通膨與地緣政治預期為主。\n"
        
    return news_summary

def analyze_strategy_with_gemini(product, direction, target_price, stop_loss, take_profit):
    print(f"[3/4] 正在連線 OKX 交易所抓取 {product} 多週期技術數據...")
    exchange = ccxt.okx()
    symbol = f"{product}/USDT:USDT"
    timeframes = ['15m', '1h', '4h']
    market_data_summary = ""
    
    for tf in timeframes:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms') + pd.Timedelta(hours=8)
        
        df['EMA_20'] = calculate_ema(df['close'], 20)
        df['EMA_50'] = calculate_ema(df['close'], 50)
        df['RSI'] = calculate_rsi(df['close'], 14)
        
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        
        market_data_summary += f"""
--- 【{tf} 週期技術面】 ---
即時 K 線時間: {latest['timestamp']} | 即時價格: {latest['close']}
前一根收盤 K 線時間: {previous['timestamp']} | 收盤價: {previous['close']}
成交量: {previous['volume']:.4f}
RSI (14) [已收盤]: {previous['RSI']:.2f}
EMA 20 [當下]: {latest['EMA_20']:.2f}
EMA 50 [當下]: {latest['EMA_50']:.2f}
------------------------
"""

    macro_data = get_macro_and_stock_data()
    news_data = get_targeted_news()

    print("[4/4] 數據準備完畢，正在交由 Gemini 3.6 Flash 進行全方位宏觀與技術面深度沙盤推演...\n")

    prompt = f"""
你是一個頂尖的華爾街宏觀經濟學家、地緣政治分析師兼加密貨幣量化操盤手。請結合以下四大維度進行深度沙盤推演與驗證。

【監控策略】
- 產品：{product}
- 方向：{direction}
- 進場點位：{target_price}
- 止損點位：{stop_loss}
- 止盈點位：{take_profit}

【1. OKX 實際市場技術數據】
{market_data_summary}

【2. 美股、美元與債市總經指標】
{macro_data}

【3. 即時總經焦點新聞】
{news_data}

請依照格式回覆技術總結、總經與FED政策連動、地緣政治、川普政策影響、風報比檢視以及最終操盤建議。用繁體中文回覆。
"""

    client = genai.Client()
    response = client.models.generate_content(
        model='gemini-3.6-flash',
        contents=prompt,
    )
    
    print("=" * 60)
    print("🤖 【Gemini 3.6 Flash 宏觀暨地緣政治智能操盤驗證報告】")
    print("=" * 60)
    print(response.text)
    print("=" * 60)

# 為了配合 Render Web Service 的要求，建立一個極簡伺服器讓它保持在線
def run_web_server():
    PORT = int(os.environ.get("PORT", 10000))
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Web server is running on port {PORT}")
        httpd.serve_forever()

if __name__ == "__main__":
    print("================ 🤖 旗艦級總經與技術策略驗證機器人 (Web Service 運行版) ================")
    
    product = "ETH"
    direction = "LONG"
    target_price = 3200.0
    stop_loss = 3100.0
    take_profit = 3500.0

    try:
        analyze_strategy_with_gemini(product, direction, target_price, stop_loss, take_profit)
    except Exception as e:
        print(f"執行分析時發生錯誤: {e}")

    # 執行完分析後，啟動網頁伺服器讓 Web Service 保持綠燈
    run_web_server()
