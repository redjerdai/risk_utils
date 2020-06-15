#
import numpy


def CAPM(portfolio, benchmark, model='OLS', check_pvals=False):
    if model == 'OLS':
        from sklearn.linear_model import LinearRegression as OLS
        model = OLS(n_jobs=-1, fit_intercept=True)
        model.fit(X=portfolio.reshape(-1, 1), y=benchmark)
        if check_pvals:
            raise NotImplemented("Not yet!")
        else:
            alpha, beta = model.intercept_, model.coef_[0]

    else:
        raise NotImplemented("Not yet!")

    return alpha, beta


def SemiDeviation(series):
    series_cut = series[series < series.mean()]
    ratio = series_cut.std(ddof=1)

    return ratio


def RatioTreynor(portfolio, benchmark, beta):
    ratio = (portfolio - benchmark).mean() / beta

    return ratio


def RatioSortino(portfolio, benchmark):
    semi_deviation = SemiDeviation(series=benchmark)
    ratio = (portfolio - benchmark).mean() / semi_deviation

    return ratio


def RatioSharpe(portfolio, benchmark):
    ratio = (portfolio - benchmark).mean() / (portfolio - benchmark).std(ddof=1)

    return ratio


def RatioVaR(portfolio, q):
    ratio = numpy.quantile(a=portfolio, q=(1 - q))

    return ratio


def RatioCVaR(portfolio, q):
    VaR = numpy.quantile(a=portfolio, q=(1 - q))
    portfolio_cut = portfolio[portfolio < VaR]
    ratio = portfolio_cut.mean()

    return ratio

