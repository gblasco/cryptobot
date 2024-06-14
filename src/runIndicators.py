import sys
from engineIndicators import Indicators

if len(sys.argv) < 5:
    print("ERROR: [ Parametros necesarios: <intervalo> <window_res_sup> <look_ahead_intervals> <pct> ]: Por ejemplo: " + sys.argv[0] + " 5m " + "20 " + "3 " + "0.5")
    sys.exit(1)  # Termina el programa con un error

interval = sys.argv[1]
windowrs = int(sys.argv[2])
look_ahead_intervals = int(sys.argv[3])
pct = float(sys.argv[4])
bot = Indicators()
df=bot.getIndicatorsCalculated(interval, windowrs, look_ahead_intervals, pct)
#print(df)
