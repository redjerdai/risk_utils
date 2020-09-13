#
import numpy
from matplotlib import pyplot
from sklearn.linear_model import LinearRegression

#
from risk_utils.risk_machine import RiskMachine

# Generate data

K = 10_000
N = 10_000
tt = numpy.array(numpy.arange(N))
portfolio = numpy.random.normal(loc=0.10, scale=10.00, size=(N,)).cumsum() + K
benchmark = numpy.random.normal(loc=0.01, scale=01.00, size=(N,)).cumsum() + K

pyplot.plot(tt, portfolio, 'navy', tt, benchmark, 'black')
pyplot.show()

risk_machine = RiskMachine()
risk_machine.add_benchs([benchmark])
risk_machine.add_portfolios([portfolio])
risk_machine.summary()

# Calculate the rates manually

portfolio_lagged = numpy.roll(portfolio, shift=1)
portfolio_yield = portfolio / portfolio_lagged - 1
portfolio_yield = portfolio_yield[1:]
benchmark_lagged = numpy.roll(benchmark, shift=1)
benchmark_yield = benchmark / benchmark_lagged - 1
benchmark_yield = benchmark_yield[1:]

portfolio_yield.reshape(-1, 1)

lm = LinearRegression(fit_intercept=True)
lm.fit(portfolio_yield.reshape(-1, 1), benchmark_yield)

alpha_rate = lm.intercept_
beta_rate = lm.coef_[0]
treynor_ratio = (portfolio_yield - benchmark_yield).mean() / lm.coef_[0]
semi_deviation = (benchmark_yield[benchmark_yield < benchmark_yield.mean()]).std(ddof=1)
sortino_ratio = (portfolio_yield - benchmark_yield).mean() / semi_deviation
sharpe_ratio = (portfolio_yield - benchmark_yield).mean() / (portfolio_yield - benchmark_yield).std(ddof=1)
value_at_risk = numpy.quantile(a=portfolio_yield, q=(1 - 0.99))
conditional_value_at_risk = (portfolio_yield[portfolio_yield < value_at_risk]).mean()
