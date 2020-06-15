#
import numpy
from matplotlib import pyplot

from risk_utils.measures import CAPM, RatioTreynor, RatioSortino, RatioSharpe, RatioVaR, RatioCVaR


class RiskMachine:

    def __init__(self):

        self.benchs_raw = []
        self.benchs = []
        self.benchs_names = []
        self.portfolios_raw = []
        self.portfolios = []
        self.portfolios_names = []

        self.N = None
        self.M = None

        self.tt = None

        self.ratioAlpha, self.ratioBeta = None, None
        self.ratioTreynor = None
        self.ratioSortino = None
        self.ratioSharpe = None
        self.ratioVaR99 = None
        self.ratioCVaR99 = None

    def add_benchs(self, benchs):

        for j in range(len(benchs)):
            a = benchs[j]
            b = numpy.roll(a, shift=1)
            c = a / b - 1
            c = c[1:]
            self.benchs.append(c)

        self.M = len(benchs)
        self.benchs_raw = self.benchs_raw + benchs

    def add_portfolios(self, portfolios):

        for i in range(len(portfolios)):
            a = portfolios[i]
            b = numpy.roll(a, shift=1)
            c = a / b - 1
            c = c[1:]
            self.portfolios.append(c)

        self.tt = numpy.array(numpy.arange(portfolios[0].shape[0]))

        self.N = len(portfolios)
        self.portfolios_raw = self.portfolios_raw + portfolios

    def compute_measures(self):

        self.ratioAlpha = numpy.full(shape=(self.N, self.M), fill_value=numpy.nan, dtype=numpy.float64)
        self.ratioBeta = numpy.full(shape=(self.N, self.M), fill_value=numpy.nan, dtype=numpy.float64)
        self.ratioTreynor = numpy.full(shape=(self.N, self.M), fill_value=numpy.nan, dtype=numpy.float64)
        self.ratioSortino = numpy.full(shape=(self.N, self.M), fill_value=numpy.nan, dtype=numpy.float64)
        self.ratioSharpe = numpy.full(shape=(self.N, self.M), fill_value=numpy.nan, dtype=numpy.float64)
        self.ratioVaR99 = numpy.full(shape=(self.N, self.M), fill_value=numpy.nan, dtype=numpy.float64)
        self.ratioCVaR99 = numpy.full(shape=(self.N, self.M), fill_value=numpy.nan, dtype=numpy.float64)

        for i in range(self.N):
            for j in range(self.M):
                self.ratioAlpha[i, j], self.ratioBeta[i, j] = CAPM(portfolio=self.portfolios[i],
                                                                   benchmark=self.benchs[j])
                self.ratioTreynor[i, j] = RatioTreynor(portfolio=self.portfolios[i], benchmark=self.benchs[j],
                                                       beta=self.ratioBeta[i, j])
                self.ratioSortino[i, j] = RatioSortino(portfolio=self.portfolios[i], benchmark=self.benchs[j])
                self.ratioSharpe[i, j] = RatioSharpe(portfolio=self.portfolios[i], benchmark=self.benchs[j])
                self.ratioVaR99[i, j] = RatioVaR(portfolio=self.portfolios[i], q=0.99)
                self.ratioCVaR99[i, j] = RatioCVaR(portfolio=self.portfolios[i], q=0.99)

    def plot(self):

        fig, ax = pyplot.subplots(self.N, self.M, figsize=(10, 10), sharex=True, sharey=True)

        if self.N * self.M > 1:
            for i in range(self.N):
                for j in range(self.M):
                    ax[i, j].plot(self.tt, self.portfolios_raw[i], color='orange', label='Portfolio {0}'.format(i))
                    ax[i, j].plot(self.tt, self.benchs_raw[j], color='navy', label='Benchmark {0}'.format(j))
                    ax[i, j].title.set_text(
                        'A={0:.2f}    B={1:.2f}\nTR={2:.2f}    SO={3:.2f}    SH={4:.2f}\nVaR={5:.2f}    CVaR={6:.2f}'.format(
                            self.ratioAlpha[i, j], self.ratioBeta[i, j], self.ratioTreynor[i, j], self.ratioSortino[i, j],
                            self.ratioSharpe[i, j], self.ratioVaR99[i, j], self.ratioCVaR99[i, j]))
        else:
            ax.plot(self.tt, self.portfolios_raw[0], color='orange', label='Portfolio {0}'.format(0))
            ax.plot(self.tt, self.benchs_raw[0], color='navy', label='Benchmark {0}'.format(0))
            ax.title.set_text(
                'A={0:.2f}    B={1:.2f}\nTR={2:.2f}    SO={3:.2f}    SH={4:.2f}\nVaR={5:.2f}    CVaR={6:.2f}'.format(
                    self.ratioAlpha[0, 0], self.ratioBeta[0, 0], self.ratioTreynor[0, 0], self.ratioSortino[0, 0],
                    self.ratioSharpe[0, 0], self.ratioVaR99[0, 0], self.ratioCVaR99[0, 0]))
        fig.show()

    def summary(self):

        self.compute_measures()
        self.plot()
