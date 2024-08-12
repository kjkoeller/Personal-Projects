import requests
from bs4 import BeautifulSoup
import yfinance as yf
import logging
from decimal import Decimal
import csv
# import numpy as np
# import pandas as pd
import os
# from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import json
import aiohttp
import asyncio
# import matplotlib.pyplot as plt
# import unittest

# Logging configuration
logging.basicConfig(filename='robo_advisor.log', level=logging.INFO)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load configuration
with open('config.json') as config_file:
    config = json.load(config_file)

class Portfolio:
    def __init__(self, cash):
        self.cash = Decimal(cash)
        self.stocks = {}
        self.initial_cash = Decimal(cash)  # Track initial cash for performance tracking

    def portfolio_performance(self, stock_prices):
        current_value = self.portfolio_value(stock_prices)
        performance = (current_value - self.initial_cash) / self.initial_cash * Decimal('100')
        logging.info(f"Portfolio Performance: {performance:.2f}%")
        return performance

    def buy_stock(self, symbol, price, quantity):
        cost = price * quantity
        if self.cash >= cost:
            self.cash -= cost
            self.stocks[symbol] = self.stocks.get(symbol, 0) + quantity

            logging.info(f"Bought {quantity} shares of {symbol} at ${price:.2f} each.")
        else:
            logging.error("Insufficient funds to buy.")

    def sell_stock(self, symbol, price, quantity):
        if self.stocks.get(symbol, 0) >= quantity:
            self.cash += price * quantity
            self.stocks[symbol] -= quantity
            logging.info(f"Sold {quantity} shares of {symbol} at ${price:.2f} each.")
        else:
            logging.error("Insufficient shares to sell.")

    def portfolio_value(self, stock_prices):
        total_value = self.cash
        for symbol, quantity in self.stocks.items():
            total_value += stock_prices.get(symbol, Decimal('0')) * quantity
        return total_value

    def import_portfolio_from_csv(self, filename):
        try:
            with open(filename, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    self.stocks[row['Symbol']] = int(row['Quantity'])
            logging.info("Portfolio imported successfully.")
        except Exception as e:
            logging.error(f"Error importing portfolio from CSV: {e}")

    def save_portfolio_to_csv(self, filename, bought_stocks, criteria):
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Symbol', 'Quantity', 'Price Per Stock', 'Total Cost', 'Gross Margin',
                             'Net Operating Margin', 'Operating Leverage', 'Financial Leverage'])
            for symbol, quantity in bought_stocks.items():
                price_per_stock = Decimal(yf.Ticker(symbol).history(period="1d")['Close'].iloc[-1])
                total_cost = price_per_stock * quantity
                writer.writerow([symbol, quantity, float(price_per_stock), float(total_cost),
                                 criteria[symbol]['gross_margin'], criteria[symbol]['net_operating_margin'],
                                 criteria[symbol]['operating_leverage'], criteria[symbol]['financial_leverage']])

class StockDataFetcher:
    @staticmethod
    def get_sp_components(url):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                table = soup.find('table', {'class': 'wikitable sortable'})
                return [row.find_all('td')[0].text.strip() for row in table.find_all('tr')[1:]]
            else:
                logging.error(f"Failed to fetch S&P component stocks from {url}. Status code: {response.status_code}")
                return []
        except Exception as e:
            logging.error(f"Error fetching S&P component stocks from {url}: {e}")
            return []

    @staticmethod
    def get_sp500_components():
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        return StockDataFetcher.get_sp_components(url)

    @staticmethod
    def get_sp400_components():
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_400_companies'
        return StockDataFetcher.get_sp_components(url)

    @staticmethod
    def get_sp600_components():
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_600_companies'
        return StockDataFetcher.get_sp_components(url)
    
    _price_cache = {}
    @staticmethod
    async def fetch_price(symbol, session):
        url = f'https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbol}'
        try:
            async with session.get(url) as response:
                data = await response.json()
                if 'quoteResponse' in data and 'result' in data['quoteResponse']:
                    result = data['quoteResponse']['result'][0]
                    latest_price = Decimal(result['regularMarketPrice'])
                    return (symbol, latest_price)
                else:
                    logging.warning(f"No price data available for {symbol}.")
                    return (symbol, Decimal('0'))
        except Exception as e:
            logging.error(f"Error fetching stock price for {symbol}: {e}")
            return (symbol, Decimal('0'))

    @staticmethod
    async def get_stock_prices(symbols):
        stock_prices = {}
        async with aiohttp.ClientSession() as session:
            tasks = [StockDataFetcher.fetch_price(symbol, session) for symbol in symbols]
            results = await asyncio.gather(*tasks)
            stock_prices.update(results)
        return stock_prices

    @staticmethod
    async def fetch_financials(symbol, session):
        url = f'https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbol}'
        try:
            async with session.get(url) as response:
                data = await response.json()
                if 'quoteResponse' in data and 'result' in data['quoteResponse']:
                    result = data['quoteResponse']['result'][0]
                    # Extract relevant financial data
                    financials = {
                        'cogs': result.get('costOfGoodsSold', None),
                        'gross_profit': result.get('grossProfit', None),
                        'ebit': result.get('ebit', None),
                        'operating_income': result.get('operatingIncome', None),
                        'total_assets': result.get('totalAssets', None),
                        'total_debt': result.get('totalDebt', None),
                        'total_equity': result.get('totalEquity', None),
                        'market_cap': result.get('marketCap', None)
                    }
                    return financials
                else:
                    logging.warning(f"No financial data available for {symbol}.")
                    return None
        except Exception as e:
            logging.error(f"Error fetching financial data for {symbol}: {e}")
            return None

    @staticmethod
    async def calculate_ratios(financials):
        try:
            # Calculate financial ratios
            gross_margin = (Decimal(financials['gross_profit']) / Decimal(financials['cogs'])) if financials['cogs'] else None
            net_operating_margin = (Decimal(financials['operating_income']) / Decimal(financials['total_assets'])) if financials['total_assets'] else None
            operating_leverage = (Decimal(financials['ebit']) / Decimal(financials['operating_income'])) if financials['operating_income'] else None
            financial_leverage = (Decimal(financials['total_assets']) / Decimal(financials['total_equity'])) if financials['total_equity'] else None

            return {
                'gross_margin': gross_margin,
                'net_operating_margin': net_operating_margin,
                'operating_leverage': operating_leverage,
                'financial_leverage': financial_leverage
            }
        except Exception as e:
            logging.error(f"Error calculating financial ratios: {e}")
            return {
                'gross_margin': None,
                'net_operating_margin': None,
                'operating_leverage': None,
                'financial_leverage': None
            }

    @staticmethod
    async def get_stock_criteria():
        criteria = {}
        try:
            SP500 = StockDataFetcher.get_sp500_components()
            SP400 = StockDataFetcher.get_sp400_components()
            SP600 = StockDataFetcher.get_sp600_components()

            symbols = SP500 + SP400 + SP600
            async with aiohttp.ClientSession() as session:
                tasks = [StockDataFetcher.fetch_financials(symbol, session) for symbol in symbols]
                financial_data_list = await asyncio.gather(*tasks)

                for symbol, financial_data in zip(symbols, financial_data_list):
                    if not financial_data:
                        continue

                    market_cap = Decimal(financial_data.get("market_cap", 0))

                    if market_cap > 10e9:
                        info = yf.Ticker(symbol).info
                        pe_ratio = info.get("forwardPE", None)
                        dividend_yield = info.get("dividendYield", None)
                        revenue_growth_rate = info.get("revenueGrowth", None)
                        eps_growth_rate = info.get("earningsGrowth", None)

                        if pe_ratio is not None and dividend_yield is not None and revenue_growth_rate is not None and eps_growth_rate is not None:
                            criteria[symbol] = {
                                'pe_ratio': pe_ratio,
                                'dividend_yield': float(dividend_yield or 0),
                                'revenue_growth_rate': revenue_growth_rate,
                                'earnings_growth_rate': eps_growth_rate,
                                **(await StockDataFetcher.calculate_ratios(financial_data) if financial_data else {})
                            }
                            logging.info(f"Criteria for {symbol}: {criteria[symbol]}")
        except Exception as e:
            logging.error(f"Error fetching stock criteria: {e}")
        return criteria

class RoboAdvisor:
    def __init__(self, portfolio):
        self.portfolio = portfolio
        
    def adjust_allocation_based_on_market(self, market_condition):
        if market_condition == 'bullish':
            return {sector: weight + Decimal('0.05') for sector, weight in config['target_allocation'].items()}
        elif market_condition == 'bearish':
            return {sector: weight - Decimal('0.05') for sector, weight in config['target_allocation'].items()}
        else:
            return config['target_allocation']

    def adjust_portfolio(self, filename, target_allocation, market_condition):
        self.portfolio.import_portfolio_from_csv(filename)
        target_allocation_decimal = {k: Decimal(v) for k, v in target_allocation.items()}
        bought_stocks = self.rebalance_portfolio(target_allocation_decimal, market_condition)
        return bought_stocks

    async def rebalance_portfolio_async(self, target_allocation, market_condition):
        criteria = StockDataFetcher.get_stock_criteria()
        if not criteria:
            logging.error("No stock criteria available.")
            return {}

        picked_stocks = self.pick_stocks(criteria)
        if not picked_stocks:
            logging.error("No stocks picked based on criteria.")
            return {}

        stock_prices = await StockDataFetcher.get_stock_prices_async(picked_stocks)
        if not stock_prices:
            logging.error("Unable to rebalance portfolio.")
            return {}

        stop_loss_threshold = Decimal('0.9') * min(stock_prices.values())
        self.sell_stocks_with_stop_loss(stop_loss_threshold, stock_prices)

        total_value = self.portfolio.portfolio_value(stock_prices)
        bought_stocks = {}
        remaining_cash = self.portfolio.cash

        max_investment_per_stock = total_value * Decimal('0.1')  # No more than 10% of the total portfolio value per stock

        for symbol in picked_stocks:
            if symbol in stock_prices:
                stock_price_decimal = stock_prices[symbol]
                current_value = self.portfolio.stocks.get(symbol, 0) * stock_price_decimal
                category = self.determine_stock_category(criteria[symbol])
                target_percentage = Decimal(str(target_allocation.get(category, 0)))
                target_value = total_value * target_percentage

                if current_value < target_value:
                    additional_quantity = min(int((target_value - current_value) / stock_price_decimal),
                                              int(criteria[symbol]['market_cap'] / stock_price_decimal))
                    actual_quantity = min(additional_quantity, int(max_investment_per_stock / stock_price_decimal))
                    actual_quantity = min(actual_quantity, int(remaining_cash / stock_price_decimal))
                    if actual_quantity > 0:
                        total_cost = actual_quantity * stock_price_decimal
                        if remaining_cash - total_cost >= Decimal('0'):
                            self.portfolio.buy_stock(symbol, stock_prices[symbol], actual_quantity)
                            bought_stocks[symbol] = actual_quantity
                            remaining_cash -= total_cost

        underallocated_stocks = [symbol for symbol in picked_stocks if symbol not in bought_stocks]
        if underallocated_stocks:
            remaining_cash_per_stock = remaining_cash / Decimal(len(underallocated_stocks))
            for symbol in underallocated_stocks:
                stock_price_decimal = stock_prices[symbol]
                additional_quantity = min(int(remaining_cash_per_stock / stock_price_decimal), 5)
                if additional_quantity > 0:
                    total_cost = additional_quantity * stock_price_decimal
                    if remaining_cash_per_stock - total_cost >= Decimal('0'):
                        self.portfolio.buy_stock(symbol, stock_prices[symbol], additional_quantity)
                        bought_stocks[symbol] = additional_quantity
                        remaining_cash -= total_cost

        logging.info("\nFinalized Portfolio:")
        logging.info(f"Cash: {self.portfolio.cash}")
        if bought_stocks:
            logging.info("\nBought stocks:")
            for symbol, quantity in bought_stocks.items():
                logging.info(f"{quantity} shares of {symbol}")
                logging.info(f"Gross Margin: {criteria[symbol]['gross_margin']}")
                logging.info(f"Net Operating Margin: {criteria[symbol]['net_operating_margin']}")
                logging.info(f"Operating Leverage: {criteria[symbol]['operating_leverage']}")
                logging.info(f"Financial Leverage: {criteria[symbol]['financial_leverage']}")

        self.portfolio.save_portfolio_to_csv('portfolio.csv', bought_stocks, criteria)
        return bought_stocks

    def rebalance_portfolio(self, target_allocation, market_condition):
        return asyncio.run(self.rebalance_portfolio_async(target_allocation, market_condition))

    def pick_stocks(self, criteria):
        return list(criteria.keys())

    def sell_stocks_with_stop_loss(self, stop_loss_threshold, stock_prices):
        for symbol, quantity in list(self.portfolio.stocks.items()):
            if symbol in stock_prices:
                current_price = stock_prices[symbol]
                if current_price < stop_loss_threshold:
                    self.portfolio.sell_stock(symbol, current_price, quantity)
                    logging.info(f"Sold {quantity} shares of {symbol} at ${current_price:.2f} due to stop loss.")

    def determine_stock_category(self, criteria):
        pe_ratio_threshold = config['pe_ratio_threshold']
        dividend_yield_threshold = config['dividend_yield_threshold']
        if criteria["pe_ratio"] > pe_ratio_threshold and criteria["dividend_yield"] < dividend_yield_threshold:
            return "growth"
        elif criteria["pe_ratio"] < pe_ratio_threshold and criteria["dividend_yield"] > dividend_yield_threshold:
            return "value"
        else:
            return "other"

if __name__ == "__main__":
    initial_cash = config['initial_cash']
    initial_portfolio = Portfolio(initial_cash)
    robo_advisor = RoboAdvisor(initial_portfolio)
    target_allocation = config['target_allocation']
    market_condition = config['market_condition']
    bought_stocks = robo_advisor.rebalance_portfolio(target_allocation[market_condition], market_condition)

    if bought_stocks:
        logging.info(f"Bought stocks: {bought_stocks}")
    else:
        logging.error("No stocks were bought.")

