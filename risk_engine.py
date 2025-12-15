import yfinance as yf
import pandas as pd
import numpy as np
from arch import arch_model
import json
import time

# TAKİP LİSTESİ
COINS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'AVAX-USD', 'DOGE-USD', 'XRP-USD']

def analyze_market():
    results = {}
    print("Analiz Başlıyor...")

    for symbol in COINS:
        try:
            print(f"{symbol} indiriliyor...")
            
            # YÖNTEM DEĞİŞİKLİĞİ: Ticker nesnesi daha kararlıdır
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1y")
            
            # Veri boş mu kontrol et
            if data.empty or len(data) < 30:
                print(f"UYARI: {symbol} verisi boş geldi.")
                # Hata olsa bile listeye ekle ki JSON boş kalmasın
                results[symbol.replace('-USD','')] = {
                    "price": "Veri Yok",
                    "risk_score": 50,
                    "trend": "Veri Bekleniyor",
                    "rsi": 50,
                    "rsi_msg": "Nötr",
                    "low_forecast": "-",
                    "high_forecast": "-",
                    "volatility": "%0.00"
                }
                continue

            current_price = data['Close'].iloc[-1]
            
            # --- 1. GARCH HESABI ---
            returns = 100 * np.log(data['Close'] / data['Close'].shift(1)).dropna()
            
            # Basitleştirilmiş model (Hata riskini azaltır)
            model = arch_model(returns, vol='Garch', p=1, q=1, mean='Zero', dist='Normal')
            res = model.fit(disp='off')
            
            forecast_var = res.forecast(horizon=1).variance.iloc[-1, 0]
            daily_vol_percent = np.sqrt(forecast_var)
            
            # --- 2. TAHMİN ARALIĞI ---
            move = current_price * (daily_vol_percent / 100)
            
            # --- 3. TREND VE RSI ---
            sma_50 = data['Close'].rolling(50).mean().iloc[-1] if len(data) > 50 else current_price
            sma_200 = data['Close'].rolling(200).mean().iloc[-1] if len(data) > 200 else current_price
            
            trend = "YATAY"
            if current_price > sma_50: trend = "BOĞA (Yükseliş)"
            elif current_price < sma_50: trend = "AYI (Düşüş)"
            
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi_val = 100 - (100 / (1 + rs)).iloc[-1]
            if np.isnan(rsi_val): rsi_val = 50

            rsi_msg = "Nötr"
            if rsi_val > 70: rsi_msg = "Aşırı Alım"
            elif rsi_val < 30: rsi_msg = "Aşırı Satım"

            # Risk Skoru
            risk_score = min((daily_vol_percent * np.sqrt(365)), 100)

            results[symbol.replace('-USD','')] = {
                "price": f"${current_price:,.2f}",
                "risk_score": int(risk_score),
                "trend": trend,
                "rsi": int(rsi_val),
                "rsi_msg": rsi_msg,
                "low_forecast": f"${(current_price - move):,.2f}",
                "high_forecast": f"${(current_price + move):,.2f}",
                "volatility": f"%{daily_vol_percent:.2f}"
            }
            
            print(f"{symbol} Başarılı.")
            time.sleep(1) # GitHub sunucusunu yormamak için bekleme

        except Exception as e:
            print(f"KRİTİK HATA {symbol}: {str(e)}")

    # Sonuçları Kaydet
    with open('risk_data.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("JSON dosyası oluşturuldu.")

if __name__ == "__main__":
    analyze_market()
