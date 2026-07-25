import os
import ccxt
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from google import genai
import feedparser

def get_macro_and_stock_data():
    """抓取美股與總經基礎數據 (S&P 500, Nasdaq, VIX, 美元指數, 美債殖利率)"""
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
    """抓取包含 CPI、戰爭、川普、FED 等關鍵字的即時財經與新聞動態"""
    print("[2/4] 正在抓取總經、CPI、戰爭、川普與 FED 相關即時新聞...")
    
    # 結合多個主流財經與綜合新聞 RSS (包含路透、CNBC、CoinDesk等)
    rss_urls = [
        "https://www.cnbc.com/id/100003114/device/rss/rss.html", # CNBC 財經/總經
        "https://www.coindesk.com/arc/outboundfeeds/rss/",         # 幣圈即時
        "https://feeds.feedburner.com/reuters/businessNews"        # 路透商業
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
                
                # 篩選出包含重要總經、戰爭、川普、FED 關鍵字的新聞
                if any(keyword in content_to_check for keyword in target_keywords):
                    if title not in collected_news:
                        collected_news.append(title)
    except Exception as e:
        collected_news.append(f"抓取新聞發生異常: {e}")
        
    # 如果篩選後標題不夠，補充幾則最新頭條
    news_summary = ""
    if collected_news:
        for title in collected_news[:10]: # 取前 10 則高度相關的焦點新聞
            news_summary += f"- {title}\n"
    else:
        news_summary = "- 近期 RSS 未過濾到特定關鍵字即時頭條，請以市場既有通膨與地緣政治預期為主。\n"
        
    return news_summary

def analyze_strategy_with_gemini(product, direction, target_price, stop_loss, take_profit):
    print("[3/4] 正在連線 OKX 交易所抓取 ETH 多週期技術數據...")
    exchange = ccxt.okx()
    symbol = f"{product}/USDT:USDT"
    timeframes = ['15m', '1h', '4h']
    market_data_summary = ""
    
    for tf in timeframes:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms') + pd.Timedelta(hours=8)
        
        df['EMA_20'] = ta.ema(df['close'], length=20)
        df['EMA_50'] = ta.ema(df['close'], length=50)
        df['RSI'] = ta.rsi(df['close'], length=14)
        
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

    # 取得總經、美債、殖利率與過濾後的總經焦點新聞（含 CPI、戰爭、川普、FED）
    macro_data = get_macro_and_stock_data()
    news_data = get_targeted_news()

    print("[4/4] 數據準備完畢，正在交由 Gemini 3.6 Flash 進行全方位宏觀與技術面深度沙盤推演...\n")

    prompt = f"""
你是一個頂尖的華爾街宏觀經濟學家、地緣政治分析師兼加密貨幣量化操盤手。現在使用者輸入了交易策略，請結合以下四大維度（**技術面數據**、**美股與總經/債市指標**、**包含 CPI、戰爭、川普、FED 談話等焦點新聞**），為使用者進行最高規格的深度沙盤推演與驗證。

【使用者輸入策略】
- 產品：{product}
- 方向：{direction}
- 進場點位：{target_price}
- 止損點位：{stop_loss}
- 止盈點位：{take_profit}

【1. OKX 實際市場技術數據】
{market_data_summary}

【2. 美股、美元與債市總經指標】
{macro_data}

【3. 即時總經焦點新聞（涵蓋 CPI、通膨、FED、川普動態、地緣政治與戰爭）】
{news_data}

請依照以下格式回覆：
1. **技術面數據總結表**：條列 15m、1h、4h 的價格與技術指標。
2. **總經、通膨 (CPI) 與 FED 政策連動**：評估當前美國通膨數據、降息預期、FED 官員/鮑爾談話對風險資產（幣圈）造成的流動性影響。
3. **地緣政治與戰爭風險評估**：結合最新的國際衝突或戰爭消息，評估市場避險情緒是否會衝擊加密貨幣。
4. **川普政策與言論影響**：若新聞或近期動態涉及川普相關言論或政策預期，分析其對市場方向的隱含衝擊。
5. **點位、滑價與風報比 (RRR) 檢視**：評估群組給的進場點、止損止盈是否在當前宏觀波動下安全。
6. **最終總結與操盤建議**：給出霸氣且一針見血的綜合執行結論。

請用繁體中文回覆，條理清晰、專業且具備前瞻性。
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

if __name__ == "__main__":
    print("================ 🤖 旗艦級總經與技術策略驗證機器人 ================")
    
    product = input("產品: ").strip()
    direction = input("方向: ").strip()
    
    try:
        target_price = float(input("進場點位: ").strip())
        stop_loss = float(input("止損點位: ").strip())
        take_profit = float(input("止盈點位: ").strip())
        
        analyze_strategy_with_gemini(product, direction, target_price, stop_loss, take_profit)
        
    except ValueError:
        print("❌ 點位與價格必須為純數字，請重新執行！")
