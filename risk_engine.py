import yfinance as yf
import numpy as np
from arch import arch_model
import json

# İSTEDİĞİN COINLERİ BURAYA EKLE
COINS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'AVAX-USD', 'DOGE-USD']

results = {}
for symbol in COINS:
    try:
        data = yf.download(symbol, period="1y", interval="1d", progress=False)
        if len(data) < 30: continue

        # GARCH Hesabı
        returns = 100 * np.log(data['Close'] / data['Close'].shift(1)).dropna()
        model = arch_model(returns, vol='Garch', p=1, q=1, mean='Zero', dist='Normal')
        res = model.fit(disp='off')

        # Sonuçları hazırla
        vol = np.sqrt(res.forecast(horizon=1).variance.iloc[-1, 0])
        risk_score = min((vol * np.sqrt(365)), 100)

        results[symbol.replace('-USD','')] = {
            "risk_score": int(risk_score),
            "volatility": f"%{vol:.2f}",
            "status": "Yüksek" if risk_score > 70 else "Orta" if risk_score > 30 else "Düşük"
        }
    except: continue

with open('risk_data.json', 'w') as f:
    json.dump(results, f)
