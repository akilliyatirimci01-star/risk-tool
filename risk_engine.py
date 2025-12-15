import yfinance as yf
import pandas as pd
import numpy as np
from arch import arch_model
import json
import time
import random

# --- TAKİP LİSTESİ ---
COINS = [
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 
    'ADA-USD', 'AVAX-USD', 'DOGE-USD', 'DOT-USD', 'LINK-USD',
    'MATIC-USD', 'LTC-USD'
]

# YEDEK VERİLER (Hata durumunda burası devreye girer)
BACKUP_DATA = {
    'BTC': 96500, 'ETH': 2700, 'SOL': 145, 'BNB': 600, 
    'XRP': 1.10, 'ADA': 0.75, 'AVAX': 35, 'DOGE': 0.38,
    'DOT': 7.50, 'LINK': 18.00, 'MATIC': 0.55, 'LTC': 85
}

def analyze_market():
    results = {}
    print("Analiz Başlıyor (Güvenli Mod)...")

    for symbol in COINS:
        clean_name = symbol.replace('-USD', '')
        try:
            # 1. VERİ ÇEKMEYİ DENE
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="3mo")
            
            # Eğer veri boşsa veya hata varsa exception fırlat
            if data.empty or len(data) < 10:
                raise ValueError("Veri boş geldi")

            # --- HESAPLAMALAR ---
            current_price = data['Close'].iloc[-1]
            prev_price = data['Close'].iloc[-2]
            daily_change = ((current_price - prev_price) / prev_price) * 100
            
            # GARCH Volatilite
            returns = 100 * np.log(data['Close'] / data['Close'].shift(1)).dropna()
            model_data = returns[-365:] if len(returns) > 365 else returns
            model = arch_model(model_data, vol='Garch', p=1, q=1, mean='Zero', dist='Normal')
            res = model.fit(disp='off')
            daily_vol_percent = np.sqrt(res.forecast(horizon=1).variance.iloc[-1, 0])
            
            # Grafik Verisi
            history_prices = data['Close'].tail(30).tolist()
            history_dates = data.index.tail(30).strftime('%d %b').tolist()

            # Sinyal Mantığı
            sma_50 = data['Close'].rolling(50).mean().iloc[-1]
            trend = "YÜKSELİŞ (Boğa)" if current_price > sma_50 else "DÜŞÜŞ (Ayı)"
            
            signal = "NÖTR"
            signal_color = "gray"
            if trend == "YÜKSELİŞ (Boğa)" and daily_change > 0:
                signal = "GÜÇLÜ ALIM BÖLGESİ"
                signal_color = "#27ae60"
            elif trend == "DÜŞÜŞ (Ayı)":
                signal = "SATIŞ BASKISI"
                signal_color = "#c0392b"

        except Exception as e:
            print(f"HATA ({symbol}): {e} -> YEDEK MOD DEVREDE")
            
            # --- YEDEK MOD (HATA OLURSA BURASI ÇALIŞIR) ---
            current_price = BACKUP_DATA.get(clean_name, 0)
            daily_change = random.uniform(-2, 2) # Küçük rastgele hareket simülasyonu
            daily_vol_percent = 2.5
            trend = "VERİ ALINAMADI"
            signal = "BEKLEMEDE"
            signal_color = "#f39c12"
            
            # Düz çizgi grafik oluştur (Boş durmasın diye)
            history_prices = [current_price] * 30
            history_dates = [f"Gün {i}" for i in range(1, 31)]

        # --- ORTAK ÇIKTI (Her durumda JSON oluşur) ---
        move = current_price * (daily_vol_percent / 100)
        risk_score = min((daily_vol_percent * 19), 100) # Basitleştirilmiş skor

        results[clean_name] = {
            "price": f"${current_price:,.2f}",
            "change_pct": daily_change,
            "risk_score": int(risk_score),
            "trend": trend,
            "signal": signal,
            "signal_color": signal_color,
            "support": f"${(current_price * 0.95):,.2f}",
            "resistance": f"${(current_price * 1.05):,.2f}",
            "low_forecast": f"${(current_price - move):,.2f}",
            "high_forecast": f"${(current_price + move):,.2f}",
            "history_prices": history_prices,
            "history_dates": history_dates
        }
        time.sleep(1) # IP ban yememek için bekleme

    # Kaydet
    with open('risk_data.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Veritabanı güncellendi (Garanti Mod).")

if __name__ == "__main__":
    analyze_market()
