from . import _py2java_double_array
from sparkts.models._model import PyModel

from pyspark.mllib.common import _py2java, _java2py
from pyspark.mllib.linalg import Vectors


def fit_model(ts, period, modelType="additive", method="BOBYQA", sc=None):
    '''
    Fit HoltWinter model to a given time series. Holt Winter Model has three parameters
    level, trend and season component of time series.
    We use BOBYQA optimizer which is used to calculate minimum of a function with
    bounded constraints and without using derivatives.
    See http://www.damtp.cam.ac.uk/user/na/NA_papers/NA2009_06.pdf for more details.
   
    @param ts Time Series for which we want to fit HoltWinter Model
    @param period Seasonality of data i.e  period of time before behavior begins to repeat itself
    @param modelType Two variations differ in the nature of the seasonal component.
        Additive method is preferred when seasonal variations are roughly constant through the series,
        Multiplicative method is preferred when the seasonal variations are changing
        proportional to the level of the series.
    @param method: Currently only BOBYQA is supported.
    '''
    assert sc != None, "Missing SparkContext"
    
    jvm = sc._jvm
    jmodel = jvm.com.cloudera.sparkts.models.HoltWinters.fitModel(_py2java(sc, Vectors.dense(ts)), period, modelType, method)
    return HoltWintersModel(jmodel=jmodel, sc=sc)


class HoltWintersModel(PyModel):
    def __init__(self, modelType='additive', period=1, alpha=0.0, beta=0.0, gamma=0.0, jmodel=None, sc=None):
        assert sc != None, "Missing SparkContext"

        self._ctx = sc
        if jmodel == None:
            self._jmodel = self._ctx._jvm.com.cloudera.sparkts.models.HoltWintersModel(modelType, period, alpha, beta, gamma)
        else:
            self._jmodel = jmodel

    def forecast(self, ts, ts1):
        jts = _py2java(self._ctx, Vectors.dense(ts))
        jts1 = _py2java(self._ctx, Vectors.dense(ts1))
        jfore = self._jmodel.forecast(jts, jts1)
        return _java2py(self._ctx, jfore)
