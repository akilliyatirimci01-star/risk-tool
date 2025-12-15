import requests
import pandas as pd
import numpy as np
import json
import time

# CoinGecko ID Eşleştirmesi (Sembol -> API ID)
COINS = {
    'BTC': 'bitcoin',
    'ETH': 'ethereum',
    'SOL': 'solana',
    'BNB': 'binancecoin',
    'XRP': 'ripple',
    'ADA': 'cardano',
    'AVAX': 'avalanche-2',
    'DOGE': 'dogecoin',
    'DOT': 'polkadot',
    'LINK': 'chainlink',
    'MATIC': 'matic-network', # Polygon yeni adıyla POL olsa da API'de bazen eski kalabilir, matic-network genelde çalışır
    'LTC': 'litecoin'
}

def analyze_market():
    results = {}
    print("CoinGecko API ile Analiz Başlıyor (Profesyonel Mod)...")

    # API'yi yormamak için her turda 4 coin çekip bekleyeceğiz
    counter = 0

    for symbol, coin_id in COINS.items():
        try:
            # 1. FİYAT VE 24SAAT DEĞİŞİM VERİSİ
            url_price = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd&include_24hr_change=true"
            r_price = requests.get(url_price, timeout=10)
            data_price = r_price.json()
            
            if coin_id not in data_price:
                print(f"Veri bulunamadı: {symbol}")
                continue

            current_price = data_price[coin_id]['usd']
            daily_change = data_price[coin_id]['usd_24h_change']

            # 2. GRAFİK VE GEÇMİŞ VERİ (Son 30 Gün)
            # CoinGecko ücretsiz planda bazen geçmiş veriyi kısıtlar, bu yüzden try-except ile koruyalım
            time.sleep(1.5) # İki istek arası bekleme (Rate Limit yememek için)
            
            url_hist = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=30&interval=daily"
            r_hist = requests.get(url_hist, timeout=10)
            data_hist = r_hist.json()
            
            prices_array = [x[1] for x in data_hist['prices']] # Sadece fiyatları al
            
            # --- TEKNİK ANALİZ HESAPLAMALARI ---
            if len(prices_array) < 14:
                prices_array = [current_price] * 30 # Hata olursa düz çizgi

            # Volatilite (Standart Sapma)
            df_prices = pd.Series(prices_array)
            volatility = df_prices.pct_change().std() * 100 # Yüzdelik volatilite
            
            # Risk Skoru (Basitleştirilmiş)
            # Volatilite ne kadar yüksekse risk o kadar artar
            risk_score = min(volatility * 10, 95)
            if risk_score < 10: risk_score = 15 # Minimum taban

            # Trend (SMA 14)
            sma = df_prices.rolling(14).mean().iloc[-1]
            trend = "YÜKSELİŞ" if current_price > sma else "DÜŞÜŞ"

            # Sinyal Mantığı
            signal = "NÖTR"
            signal_color = "#95a5a6"
            
            if trend == "YÜKSELİŞ" and daily_change > -1:
                signal = "GÜÇLÜ AL"
                signal_color = "#27ae60"
            elif trend == "DÜŞÜŞ" and daily_change < -2:
                signal = "DİP ARAYIŞI" # Sat demek yerine
                signal_color = "#e74c3c"
            elif trend == "DÜŞÜŞ":
                signal = "BEKLE"
                signal_color = "#f39c12"

            # ATR Benzeri Basit Stop Loss (Fiyatın %5 altı veya volatiliteye göre)
            stop_margin = current_price * (volatility / 100 * 2) # 2 kat volatilite payı
            if stop_margin == 0: stop_margin = current_price * 0.05
            
            stop_loss = current_price - stop_margin
            take_profit = current_price + (stop_margin * 1.5)

            # Kasa Yönetimi
            if risk_score > 70: position_size = "%2-3 (Riskli)"
            elif risk_score > 40: position_size = "%5 (Orta)"
            else: position_size = "%10 (Güvenli)"

            # Grafik Tarihleri (Basitçe son 30 gün)
            history_dates = [f"{i} Gün" for i in range(1, len(prices_array)+1)]

            results[symbol] = {
                "price": f"${current_price:,.2f}",
                "change_pct": daily_change,
                "risk_score": int(risk_score),
                "trend": trend,
                "signal": signal,
                "signal_color": signal_color,
                "stop_loss": f"${stop_loss:,.2f}",
                "take_profit": f"${take_profit:,.2f}",
                "position_size": position_size,
                "history_prices": prices_array,
                "history_dates": history_dates
            }
            
            print(f"{symbol} OK. Fiyat: {current_price}")
            
        except Exception as e:
            print(f"HATA {symbol}: {e}")
        
        # CoinGecko Free API limiti: Dakikada ~10-30 istek.
        # Biz her coinde 2 istek atıyoruz. O yüzden beklemeliyiz.
        time.sleep(4) 

    with open('risk_data.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Tüm veriler CoinGecko üzerinden başarıyla güncellendi.")

if __name__ == "__main__":
    analyze_market()
