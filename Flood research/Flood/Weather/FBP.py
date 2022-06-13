import pandas as pd
from prophet import Prophet


class FB:
    def __init__(self, path):
        self.path = path

    def get_path(self): return self.path

    def get_week(self):
        df = pd.read_csv('Weather/FBP_Data/'+self.path)
        df.head()

        m = Prophet()
        m.fit(df)

        future = m.make_future_dataframe(periods=7)
        future.tail()

        forecast = m.predict(future)
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

        return forecast['yhat'][-7:]

    def get_month(self):
        df = pd.read_csv('Weather/FBP_Data/'+self.path)
        df.head()

        m = Prophet()
        m.fit(df)

        future = m.make_future_dataframe(periods=30)
        future.tail()

        forecast = m.predict(future)
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

        return forecast['yhat'][-30:]



