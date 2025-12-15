import yfinance as yf
import pandas as pd
import numpy as np
from arch import arch_model
import json
import time

# --- GENİŞLETİLMİŞ COIN LİSTESİ ---
COINS = [
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 
    'ADA-USD', 'AVAX-USD', 'DOGE-USD', 'DOT-USD', 'LINK-USD',
    'MATIC-USD', 'LTC-USD', 'UNI-USD', 'ATOM-USD'
]

def analyze_market():
    results = {}
    print("Mega Analiz Başlıyor...")

    for symbol in COINS:
        try:
            print(f"{symbol} işleniyor...")
            ticker = yf.Ticker(symbol)
            
            # Grafik için son 1.5 ayın verisini çek
            data = ticker.history(period="3mo")
            
            if data.empty or len(data) < 30:
                print(f"ATLANDI: {symbol} (Veri yok)")
                continue

            current_price = data['Close'].iloc[-1]
            prev_price = data['Close'].iloc[-2]
            
            # --- 1. GÜNLÜK DEĞİŞİM ---
            daily_change = ((current_price - prev_price) / prev_price) * 100
            
            # --- 2. GARCH VOLATİLİTE ---
            returns = 100 * np.log(data['Close'] / data['Close'].shift(1)).dropna()
            # Son 1 yıllık veriyi model için kullanalım
            model_data = returns[-365:] if len(returns) > 365 else returns
            
            model = arch_model(model_data, vol='Garch', p=1, q=1, mean='Zero', dist='Normal')
            res = model.fit(disp='off')
            forecast_var = res.forecast(horizon=1).variance.iloc[-1, 0]
            daily_vol_percent = np.sqrt(forecast_var)
            
            # --- 3. TAHMİN ARALIĞI ---
            move = current_price * (daily_vol_percent / 100)
            
            # --- 4. DESTEK & DİRENÇ (CLASSIC PIVOT) ---
            last_high = data['High'].iloc[-2]
            last_low = data['Low'].iloc[-2]
            last_close = data['Close'].iloc[-2]
            
            pivot = (last_high + last_low + last_close) / 3
            support_1 = (2 * pivot) - last_high
            resistance_1 = (2 * pivot) - last_low
            
            # --- 5. RSI & TREND ---
            sma_50 = data['Close'].rolling(50).mean().iloc[-1] if len(data) > 50 else current_price
            
            trend = "YATAY"
            if current_price > sma_50: trend = "YÜKSELİŞ (Boğa)"
            elif current_price < sma_50: trend = "DÜŞÜŞ (Ayı)"
            
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi_val = 100 - (100 / (1 + rs)).iloc[-1]
            if np.isnan(rsi_val): rsi_val = 50

            # --- 6. AKILLI SİNYAL ÜRETİCİ ---
            signal = "NÖTR / İZLE"
            signal_color = "gray"
            
            risk_score = min((daily_vol_percent * np.sqrt(365)), 100)
            
            if trend == "YÜKSELİŞ (Boğa)" and rsi_val < 70:
                signal = "GÜÇLÜ ALIM BÖLGESİ"
                signal_color = "#27ae60" # Yeşil
            elif trend == "DÜŞÜŞ (Ayı)" and rsi_val > 30:
                signal = "SATIŞ BASKISI / BEKLE"
                signal_color = "#c0392b" # Kırmızı
            elif rsi_val < 30:
                signal = "DİP FIRSATI OLABİLİR"
                signal_color = "#f39c12" # Turuncu
            elif rsi_val > 75:
                signal = "AŞIRI ŞİŞTİ / DÜZELTME RİSKİ"
                signal_color = "#e74c3c"
                
            # --- 7. GRAFİK VERİSİ (Son 30 Gün) ---
            # JSON boyutunu şişirmemek için sadece son 30 günü alıyoruz
            history_prices = data['Close'].tail(30).tolist()
            history_dates = data.index.tail(30).strftime('%d %b').tolist()

            clean_name = symbol.replace('-USD','')
            results[clean_name] = {
                "price": f"${current_price:,.2f}",
                "change_pct": daily_change,
                "risk_score": int(risk_score),
                "trend": trend,
                "signal": signal,
                "signal_color": signal_color,
                "rsi": int(rsi_val),
                "support": f"${support_1:,.2f}",
                "resistance": f"${resistance_1:,.2f}",
                "low_forecast": f"${(current_price - move):,.2f}",
                "high_forecast": f"${(current_price + move):,.2f}",
                "history_prices": history_prices, # Grafik verisi
                "history_dates": history_dates    # Grafik tarihleri
            }
            print(f"{clean_name} Eklendi.")
            time.sleep(0.5)

        except Exception as e:
            print(f"HATA {symbol}: {e}")

    with open('risk_data.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Veritabanı güncellendi.")

if __name__ == "__main__":
    analyze_market()
