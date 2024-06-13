# Cryptobot
Cryptobot

> Entrenamiento modelo:

$ python .\engineBTCLive.py 5m False          -> fullhistory5m.csv

$ python .\runIndicators.py 5m 20 4 0.5       -> fullhistorywithindicators5m_up02pct

$ python .\modeloNeural5m_up02pct.py


> Validaciones modelo:

$ python .\engineBTCLive.py 5m True 2000

$ python .\runIndicatorsLive.py 5m 20 4 0.5

$ python .\predictNeural5m_up02pct_scaled.py	
