import yfinance as yf
import numpy as np
import pandas as pd
from arch import arch_model
import json

# TAKİP LİSTESİ
COINS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'AVAX-USD', 'DOGE-USD', 'XRP-USD']

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

results = {}

print("Analiz Başlıyor...")

for symbol in COINS:
    try:
        # Son 1.5 yıllık veriyi çek (200 günlük ortalama için gerekli)
        data = yf.download(symbol, period="2y", interval="1d", progress=False)
        if len(data) < 200: continue
        
        current_price = data['Close'].iloc[-1]
        
        # --- 1. GARCH VOLATİLİTE HESABI ---
        returns = 100 * np.log(data['Close'] / data['Close'].shift(1)).dropna()
        model = arch_model(returns, vol='Garch', p=1, q=1, mean='Zero', dist='Normal')
        res = model.fit(disp='off')
        
        # Gelecek varyans tahmini
        forecast_var = res.forecast(horizon=1).variance.iloc[-1, 0]
        daily_vol_percent = np.sqrt(forecast_var) # Örn: 3.5 (Yüzde olarak)
        
        # --- 2. FİYAT TAHMİN ARALIĞI (GARCH Temelli) ---
        # %95 Güven aralığı için 1.96 standart sapma kullanılır ama kripto için 1.5 daha gerçekçidir.
        move_amount = current_price * (daily_vol_percent / 100)
        low_target = current_price - move_amount
        high_target = current_price + move_amount
        
        # --- 3. TREND VE DÖNGÜ ANALİZİ ---
        sma_50 = data['Close'].rolling(window=50).mean().iloc[-1]
        sma_200 = data['Close'].rolling(window=200).mean().iloc[-1]
        
        trend = "YATAY"
        if current_price > sma_50 and sma_50 > sma_200:
            trend = "BOĞA (Yükseliş)"
        elif current_price < sma_50 and sma_50 < sma_200:
            trend = "AYI (Düşüş)"
        elif current_price < sma_50 and sma_50 > sma_200:
            trend = "DÜZELTME"
            
        # --- 4. RSI (MOMENTUM) ---
        rsi_series = calculate_rsi(data)
        rsi_val = rsi_series.iloc[-1]
        
        rsi_status = "Nötr"
        if rsi_val > 70: rsi_status = "Aşırı Isınma (Satış Riski)"
        elif rsi_val < 30: rsi_status = "Dip Bölge (Alım Fırsatı)"
        
        # --- 5. RİSK SKORU ---
        annual_vol = daily_vol_percent * np.sqrt(365)
        risk_score = min((annual_vol), 100) # Basit skorlama
        
        # JSON ÇIKTISI
        clean_name = symbol.replace('-USD','')
        results[clean_name] = {
            "price": f"${current_price:,.2f}",
            "risk_score": int(risk_score),
            "trend": trend,
            "rsi": int(rsi_val),
            "rsi_msg": rsi_status,
            "low_forecast": f"${low_target:,.2f}",
            "high_forecast": f"${high_target:,.2f}",
            "volatility": f"%{daily_vol_percent:.2f}"
        }
        print(f"{clean_name} tamamlandı.")
        
    except Exception as e:
        print(f"Hata {symbol}: {e}")

# Kaydet
with open('risk_data.json', 'w') as f:
    json.dump(results, f)
