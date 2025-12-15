import yfinance as yf
import pandas as pd
import numpy as np
from arch import arch_model
import json
import time
import random

COINS = [
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 
    'ADA-USD', 'AVAX-USD', 'DOGE-USD', 'DOT-USD', 'LINK-USD',
    'MATIC-USD', 'LTC-USD'
]

BACKUP_DATA = {
    'BTC': 97000, 'ETH': 2750, 'SOL': 150, 'BNB': 610, 
    'XRP': 1.15, 'ADA': 0.78, 'AVAX': 36, 'DOGE': 0.39,
    'DOT': 7.60, 'LINK': 18.50, 'MATIC': 0.58, 'LTC': 88
}

def analyze_market():
    results = {}
    print("Risk Yönetimi Modülü Başlatılıyor...")

    for symbol in COINS:
        clean_name = symbol.replace('-USD', '')
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="6mo") # ATR için biraz daha veri lazım
            
            if data.empty or len(data) < 20:
                raise ValueError("Yetersiz Veri")

            current_price = data['Close'].iloc[-1]
            
            # --- 1. ATR HESAPLAMASI (Volatilite Bazlı Stop-Loss için) ---
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            atr = true_range.rolling(14).mean().iloc[-1]

            # Stop-Loss ve Take-Profit Seviyeleri
            # Genellikle 2 ATR altı stop, 3 ATR üstü hedef makuldür
            stop_loss = current_price - (atr * 2)
            take_profit = current_price + (atr * 3)
            
            # --- 2. GÜNLÜK DEĞİŞİM & RİSK SKORU ---
            prev_price = data['Close'].iloc[-2]
            daily_change = ((current_price - prev_price) / prev_price) * 100
            
            returns = 100 * np.log(data['Close'] / data['Close'].shift(1)).dropna()
            model_data = returns[-365:] if len(returns) > 365 else returns
            model = arch_model(model_data, vol='Garch', p=1, q=1, mean='Zero', dist='Normal')
            res = model.fit(disp='off')
            daily_vol_percent = np.sqrt(res.forecast(horizon=1).variance.iloc[-1, 0])
            
            # Risk Skoru (0-100)
            risk_score = min((daily_vol_percent * 15), 100)

            # --- 3. KASA YÖNETİMİ ÖNERİSİ ---
            # Risk ne kadar yüksekse, önerilen pozisyon o kadar düşük olmalı
            if risk_score > 80:
                position_size = "Çok Riskli (%2)"
            elif risk_score > 60:
                position_size = "Düşük (%5)"
            elif risk_score > 40:
                position_size = "Orta (%10)"
            else:
                position_size = "Yüksek (%15)"

            # Grafik Verisi (Simülasyon yok, gerçek veri)
            history_prices = data['Close'].tail(30).tolist()
            history_dates = data.index.tail(30).strftime('%d %b').tolist()
            
            # Trend Analizi
            sma_50 = data['Close'].rolling(50).mean().iloc[-1]
            trend = "YÜKSELİŞ" if current_price > sma_50 else "DÜŞÜŞ"
            
            signal = "NÖTR"
            signal_color = "#95a5a6" # Gri
            
            # Basit Sinyal Mantığı
            if trend == "YÜKSELİŞ" and daily_change > -2:
                signal = "ALIM FIRSATI"
                signal_color = "#27ae60"
            elif trend == "DÜŞÜŞ":
                signal = "SAT / BEKLE"
                signal_color = "#c0392b"

        except Exception as e:
            # --- YEDEK MOD (HATA DURUMU) ---
            print(f"Hata ({symbol}): {e}, Yedek Veri Kullanılıyor.")
            current_price = BACKUP_DATA.get(clean_name, 100)
            atr = current_price * 0.05 # Tahmini %5 volatilite
            stop_loss = current_price * 0.90
            take_profit = current_price * 1.15
            daily_change = random.uniform(-1.5, 1.5)
            risk_score = 50
            position_size = "Nötr (%5)"
            trend = "BİLİNMİYOR"
            signal = "VERİ YOK"
            signal_color = "#f39c12"
            
            # Yapay Dalgalı Grafik
            history_prices = []
            p = current_price
            for _ in range(30):
                p = p * random.uniform(0.97, 1.03)
                history_prices.append(p)
            history_dates = [f"Gün {i}" for i in range(1, 31)]

        results[clean_name] = {
            "price": f"${current_price:,.2f}",
            "change_pct": daily_change,
            "risk_score": int(risk_score),
            "trend": trend,
            "signal": signal,
            "signal_color": signal_color,
            "stop_loss": f"${stop_loss:,.2f}",     # YENİ
            "take_profit": f"${take_profit:,.2f}", # YENİ
            "position_size": position_size,         # YENİ
            "history_prices": history_prices,
            "history_dates": history_dates
        }
        time.sleep(0.5)

    with open('risk_data.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    analyze_market()
