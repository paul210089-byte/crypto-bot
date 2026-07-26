import os
import logging
import http.server
import socketserver
import threading
import re
import time
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

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
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

        # 升級為「客觀多空預測與沙盤推演」的提示詞
        prompt = f"""
你是一個頂尖的加密貨幣量化預測專家與多空市場分析師。請針對以下策略及當前市場結構進行【客觀的多空趨勢預測與沙盤推演】：
- 產品：{product}
- 原始設定方向：{direction}
- 進場點位：{target_price}
- 止損點位：{stop_loss}
- 止盈點位：{take_profit}

【市場數據】
{market_data_summary}

【總經指標】
{macro_data}

【即時焦點新聞】
{news_data}

請依據盤面結構獨立判斷多空，並給出：
1. **多空結構與動能預測**（當前技術指標、均線排列與量能偏向多方還是空方）
2. **多方情境沙盤**（若延續漲勢，支撐、壓力與目標價位為何）
3. **空方情境沙盤**（若出現反轉、假突破或疲態，回測與下跌防守點位為何）
4. **風控檢視與點位評估**（檢視原始點位的風險報酬比是否合理）
5. **最終操盤結論**（明確給出多空偏向與操作建議）。
繁體中文回覆。
"""
        client = genai.Client()
        
        response = None
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model='gemini-3.6-flash',
                    contents=prompt,
                )
                break
            except Exception as api_err:
                if ("429" in str(api_err) or "503" in str(api_err)) and attempt < 2:
                    time.sleep(3)
                    continue
                raise api_err

        return response.text
    except Exception as e:
        return f"分析過程發生錯誤: {e}"

def parse_signal_text(text):
    product, direction, target_price, stop_loss, take_profit = None, None, None, None, None
    
    prod_match = re.search(r'(?:产品|幣種|币种|标的)[:：]?\s*([A-Za-z]+)', text)
    if prod_match:
        product = prod_match.group(1).upper()
    else:
        for coin in ['ETH', 'BTC', 'SOL', 'XRP', 'DOGE']:
            if coin in text.upper():
                product = coin
                break

    if '做多' in text or 'LONG' in text.upper():
        direction = 'LONG'
    elif '做空' in text or 'SHORT' in text.upper():
        direction = 'SHORT'

    numbers = re.findall(r'\d+(?:\.\d+)?', text)
    nums = [float(n) for n in numbers]

    match_tp = re.search(r'(?:进场点位|進場點位|进场|進場)[:：]?\s*([^\n]+)', text)
    match_sl = re.search(r'(?:止損點位|止损点位|止损|止損)[:：]?\s*([^\n]+)', text)
    match_profit = re.search(r'(?:止盈點位|止盈点位|止盈)[:：]?\s*([^\n]+)', text)

    try:
        if match_tp:
            target_price = float(re.findall(r'\d+(?:\.\d+)?', match_tp.group(1))[0])
        if match_sl:
            stop_loss = float(re.findall(r'\d+(?:\.\d+)?', match_sl.group(1))[0])
        if match_profit:
            take_profit = float(re.findall(r'\d+(?:\.\d+)?', match_profit.group(1))[0])
    except:
        pass

    if not target_price and len(nums) >= 1: target_price = nums[0]
    if not stop_loss and len(nums) >= 2: stop_loss = nums[1]
    if not take_profit and len(nums) >= 3: take_profit = nums[2]

    return product, direction, target_price, stop_loss, take_profit

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text or update.message.caption
    if not text:
        return

    await update.message.reply_text("🤖 偵測到喊單！正在進行多空獨立預測與沙盤推演...")
    product, direction, target_price, stop_loss, take_profit = parse_signal_text(text)

    if not all([product, direction, target_price, stop_loss, take_profit]):
        await update.message.reply_text("❌ 無法完整解析此則喊單的點位，請手動使用 `/check` 指令輸入！", parse_mode="Markdown")
        return

    report = analyze_strategy_with_gemini(product, direction, target_price, stop_loss, take_profit)
    
    if len(report) > 4000:
        for i in range(0, len(report), 4000):
            await update.message.reply_text(report[i:i+4000])
    else:
        await update.message.reply_text(report)

async def check_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if len(args) < 5:
        await update.message.reply_text("⚠️ 格式錯誤！請輸入 `/check ETH LONG 1857 1845 1900`", parse_mode="Markdown")
        return
    
    product, direction, target_price, stop_loss, take_profit = args[0].upper(), args[1].upper(), float(args[2]), float(args[3]), float(args[4])
    await update.message.reply_text(f"🔍 收到指令！正在獨立預測 {product} 的多空結構...")
    
    report = analyze_strategy_with_gemini(product, direction, target_price, stop_loss, take_profit)
    await update.message.reply_text(report)

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 智能多空預測機器人已就緒！\n\n"
        "✨ 直接將群組喊單訊息**「轉發 (Forward)」**給本機器人，即可自動產出多空雙向預測報告！",
        parse_mode="Markdown"
    )

def run_web_server():
    PORT = int(os.environ.get("PORT", 10000))
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Web server running on port {PORT}")
        httpd.serve_forever()

if __name__ == "__main__":
    server_thread = threading.Thread(target=run_web_server, daemon=True)
    server_thread.start()

    TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not TOKEN:
        print("❌ 錯誤：未設定 TELEGRAM_BOT_TOKEN 環境變數！")
        exit(1)

    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("check", check_command))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    print("🤖 機器人已成功啟動並監聽中...")
    app.run_polling()
