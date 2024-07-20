import requests
from bs4 import BeautifulSoup
import yfinance as yf
import logging
from decimal import Decimal
import csv
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import json
import matplotlib.pyplot as plt
import unittest

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
            if self.stocks[symbol] == 0:
                del self.stocks[symbol]
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

    def save_portfolio_to_csv(self, filename, bought_stocks):
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Symbol', 'Quantity', 'Price Per Stock', 'Total Cost'])
            for symbol, quantity in bought_stocks.items():
                price_per_stock = Decimal(yf.Ticker(symbol).history(period="1d")['Close'].iloc[-1])
                total_cost = price_per_stock * quantity
                writer.writerow([symbol, quantity, float(price_per_stock), float(total_cost)])

class StockDataFetcher:
    @staticmethod
    def get_sp500_components():
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                table = soup.find('table', {'class': 'wikitable sortable'})
                return [row.find_all('td')[0].text.strip() for row in table.find_all('tr')[1:]]
            else:
                logging.error(f"Failed to fetch S&P 500 component stocks. Status code: {response.status_code}")
                return []
        except Exception as e:
            logging.error(f"Error fetching S&P 500 component stocks: {e}")
            return []

    @staticmethod
    def fetch_price(symbol):
        try:
            stock = yf.Ticker(symbol)
            hist_data = stock.history(period="1d")
            if not hist_data.empty:
                latest_price = Decimal(hist_data['Close'].iloc[-1])
                return (symbol, latest_price)
            else:
                logging.warning(f"No price data available for {symbol}.")
                return (symbol, Decimal('0'))
        except Exception as e:
            logging.error(f"Error fetching stock price for {symbol}: {e}")
            return (symbol, Decimal('0'))

    @staticmethod
    def get_stock_prices(symbols):
        stock_prices = {}
        with ThreadPoolExecutor() as executor:
            results = executor.map(StockDataFetcher.fetch_price, symbols)
            stock_prices.update(results)
        return stock_prices

    @staticmethod
    def get_stock_criteria():
        criteria = {}
        try:
            symbols = StockDataFetcher.get_sp500_components()
            for symbol in symbols:
                stock = yf.Ticker(symbol)
                info = stock.info
                market_cap = info.get("marketCap", None)

                if market_cap and market_cap > 10e9:
                    pe_ratio = info.get("forwardPE", 0)
                    dividend_yield = info.get("dividendYield", 0)
                    revenue_growth_rate = info.get("revenueGrowth", 0)
                    eps_growth_rate = info.get("earningsGrowth", 0)
                    if (pe_ratio and 5 < pe_ratio < 15 and
                        dividend_yield and dividend_yield > 0.03 and
                        revenue_growth_rate and revenue_growth_rate > 0.05 and
                        eps_growth_rate and eps_growth_rate > 0.05):
                        criteria[symbol] = {
                            'pe_ratio': pe_ratio,
                            'dividend_yield': float(dividend_yield or 0),
                            'revenue_growth_rate': revenue_growth_rate,
                            'earnings_growth_rate': eps_growth_rate,
                            'market_cap': market_cap
                        }
                        logging.info(f"Criteria for {symbol}: {criteria[symbol]}")
        except Exception as e:
            logging.error(f"Error fetching stock criteria: {e}")
        return criteria

class RoboAdvisor:
    def __init__(self, portfolio):
        self.portfolio = portfolio

    def adjust_portfolio(self, filename, target_allocation, market_condition):
        self.portfolio.import_portfolio_from_csv(filename)
        target_allocation_decimal = {k: Decimal(v) for k, v in target_allocation.items()}
        bought_stocks = self.rebalance_portfolio(target_allocation_decimal, market_condition)
        return bought_stocks

    def rebalance_portfolio(self, target_allocation, market_condition):
        criteria = StockDataFetcher.get_stock_criteria()
        if not criteria:
            logging.error("No stock criteria available.")
            return {}

        picked_stocks = self.pick_stocks(criteria)
        if not picked_stocks:
            logging.error("No stocks picked based on criteria.")
            return {}

        stock_prices = StockDataFetcher.get_stock_prices(picked_stocks)
        if not stock_prices:
            logging.error("Unable to rebalance portfolio.")
            return {}

        stop_loss_threshold = Decimal('0.9') * min(stock_prices.values())
        self.sell_stocks_with_stop_loss(stop_loss_threshold, stock_prices)

        total_value = self.portfolio.portfolio_value(stock_prices)
        bought_stocks = {}
        remaining_cash = self.portfolio.cash

        # Diversify investment
        max_investment_per_stock = total_value * Decimal('0.2')  # For example, no more than 20% of the total portfolio value per stock

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
                    # Ensure we do not exceed the max investment per stock
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

        self.portfolio.save_portfolio_to_csv('portfolio.csv', bought_stocks)
        return bought_stocks

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
    bought_stocks = robo_advisor.adjust_portfolio('current_portfolio.csv', target_allocation[market_condition], market_condition)

    if bought_stocks:
        logging.info(f"Bought stocks: {bought_stocks}")
        stock_prices = StockDataFetcher.get_stock_prices(bought_stocks.keys())
    else:
        logging.error("No stocks were bought.")
