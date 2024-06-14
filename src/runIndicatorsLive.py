import sys
from engineIndicatorsLive import IndicatorsLive

if len(sys.argv) < 5:
    print("ERROR: [ Parametros necesarios: <intervalo> <window_res_sup> <look_ahead_intervals> <pct> ]: Por ejemplo: " + sys.argv[0] + " 5m " + "20 " + "3 " + "0.5")
    sys.exit(1)

interval = sys.argv[1]
windowrs = int(sys.argv[2])
look_ahead_intervals = int(sys.argv[3])
pct = float(sys.argv[4])
bot = IndicatorsLive()
df=bot.getIndicatorsCalculated(interval, windowrs, look_ahead_intervals, pct)
