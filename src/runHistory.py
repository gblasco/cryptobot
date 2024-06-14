from engineBTCHistory import GetDataBTC

bot = GetDataBTC()
dfhistorico=bot.getBTCHistory(0, "5m", "10 years", "../data/history")
#dflive=dfhistorico.tail(1)
#dfhistorico.to_csv('datos_entrada_modelo.csv', index=False)
# runmodel = ModelNeural()
# model, accuracy = runmodel.CreateModel('Final2',dfhistorico)
# print(accuracy)
#model.save('modelo_redneuronal_1m')
#dflive.to_csv('to_be_predicted.csv', index=False)