import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as func
from pyspark.sql.window import Window
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA

spark = SparkSession \
        .builder \
        .appName("mlPyspark Project") \
        .getOrCreate()

#Reading the File
df1 = spark.read.csv('C:/data/turbine_power_data.csv', header=True, inferSchema=True)

#Droping null values
df2 = df1.dropna("any")
# Data Transformation
df2 = df2.withColumn('mapCol', \
                    func.create_map(func.lit('turbine1'),df2.power_turbine1,
                                    func.lit('turbine2'),df2.power_turbine2,
                                    func.lit('turbine3'),df2.power_turbine3
                                   )
                  )
df3 = df2.select('time',func.explode(df2.mapCol).alias('turbine','power'))
# df3.show()

# Conversion time to timestamp and Power to Double
df4 = df3.withColumn("Npower", df3["power"].cast("double")).withColumn("Ntime", func.to_timestamp(df3["time"], format = 'MM/dd/yyyy HH:mm:ss')).select("time", "Npower", "Ntime")
#df4.show()

#Resample of data at 10min Granularity and Sum of Power
df5 = df4.groupBy('turbine', Window("Ntime", "10m")).agg(sum("Npower").alias('Sum Power')).select("Ntime", "Sum Power")

#Tranform the Power
#df6 = df5.withColumn("logPower", func.log("Sum Power"))

header = df5.first()

#Remove Header
data = df5.filter(lambda row: row != header)

#def test_stationarity(timeseries):
#    # Perform Dickey-Fuller test:
#    print('Results of Dickey-Fuller Test:')
#    dftest = adfuller(timeseries.iloc[:, 0].values, autolag='AIC')
#    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
#    for key, value in dftest[4].items():
#        dfoutput['Critical Value (%s)' % key] = value
#    # print(dftest)
#    print(dfoutput)

#plot_acf(data)

#plot_pacf(data)

#Split data into train and test
(traindata, testdata) = data.randomSplit([0.7, 0.3])

#fit model

model = ARIMA(data, order=[1, 0, 0])
model_ARIMA = model.fit()
print(model_ARIMA.summary())

power_forecast = model_ARIMA.forecast(steps=6)[0]

print(power_forecast)
