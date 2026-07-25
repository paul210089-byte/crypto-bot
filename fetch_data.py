import os
import logging
import asyncio
import http.server
import socketserver
import threading
import ccxt
import yfinance as yf
import pandas as pd
from google import genai
import feedparser
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

# 啟用 Logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# 自定義計算 EMA
def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

# 使用標準 Wilder 平滑法計算 RSI (與 TradingView / OKX 完全同步)
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # 採用 Wilder's Smoothing (RMA)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_macro_and_stock_data():
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
                    macro_summary += f"- {name}: {latest_val:.2f} (漲跌幅: {change_pct:+.2f}%)\n"
    except Exception as e:
        macro_summary += f"抓取總經數據發生錯誤: {e}\n"
    return macro_summary

def get_targeted_news():
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
        for title in collected_news[:5]:
            news_summary += f"- {title}\n"
    else:
        news_summary = "- 近期 RSS 未過濾到特定關鍵字即時頭條。\n"
    return news_summary

def analyze_strategy_with_gemini(product, direction, target_price, stop_loss, take_profit):
    try:
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
--- 【{tf} 週期】 ---
價格: {latest['close']} | RSI: {previous['RSI']:.2f}
EMA20: {latest['EMA_20']:.2f} | EMA50: {latest['EMA_50']:.2f}
"""

        macro_data = get_macro_and_stock_data()
        news_data = get_targeted_news()

        prompt = f"""
你是一個頂尖的華爾街宏觀經濟學家、地緣政治分析師兼加密貨幣量化操盤手。請為以下帶單策略進行深度沙盤推演：
- 產品：{product}
- 方向：{direction}
- 進場點位：{target_price}
- 止損點位：{stop_loss}
- 止盈點位：{take_profit}

【市場數據】
{market_data_summary}

【總經指標】
{macro_data}

【即時焦點新聞】
{news_data}

請給出：1.技術總結 2.總經與FED連動 3.地緣政治風險 4.點位與風報比檢視 5.最終操盤結論。繁體中文回覆。
"""
        client = genai.Client()
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"分析過程發生錯誤: {e}"

# Telegram 指令: /check 產品 方向 進場 止損 止盈
async def check_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if len(args) < 5:
        await update.message.reply_text(
            "⚠️ 格式錯誤！\n請依照以下格式輸入：\n`/check ETH LONG 1860 1845 1880`\n(參數分別為: 幣種 方向 進場價 止損價 止盈價)",
            parse_mode="Markdown"
        )
        return

    product = args[0].upper()
    direction = args[1].upper()
    try:
        target_price = float(args[2])
        stop_loss = float(args[3])
        take_profit = float(args[4])
    except ValueError:
        await update.message.reply_text("❌ 點位與價格必須為純數字！")
        return

    await update.message.reply_text(f"🔍 收到指令！正在背景調用 OKX、總經指標與 Gemini 3.6 Flash 分析 {product}，請稍候...")

    # 執行 AI 分析
    report = analyze_strategy_with_gemini(product, direction, target_price, stop_loss, take_profit)

    # 如果報告太長，Telegram 會限制 4000 字，分段發送
    if len(report) > 4000:
        for i in range(0, len(report), 4000):
            await update.message.reply_text(report[i:i+4000])
    else:
        await update.message.reply_text(report)

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 歡迎使用旗艦級總經與技術策略驗證機器人！\n\n"
        "請輸入指令開始驗證帶單：\n"
        "`/check ETH LONG 1860 1845 1880`",
        parse_mode="Markdown"
    )

# 啟動極簡 Web 伺服器應付 Render
def run_web_server():
    PORT = int(os.environ.get("PORT", 10000))
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Web server running on port {PORT}")
        httpd.serve_forever()

if __name__ == "__main__":
    # 在背景執行 Web 伺服器
    server_thread = threading.Thread(target=run_web_server, daemon=True)
    server_thread.start()

    # 取得 Telegram Bot Token
    TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not TOKEN:
        print("❌ 錯誤：未設定 TELEGRAM_BOT_TOKEN 環境變數！")
        exit(1)

    # 啟動 Telegram 機器人
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("check", check_command))

    print("🤖 Telegram 驗證機器人已在雲端成功啟動並監聽中...")
    app.run_polling()
