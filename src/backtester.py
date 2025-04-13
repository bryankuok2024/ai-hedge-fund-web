import sys
import math

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import questionary

import matplotlib.pyplot as plt
import pandas as pd
from colorama import Fore, Style, init
import numpy as np
import itertools

from llm.models import LLM_ORDER, get_model_info
from utils.analysts import ANALYST_ORDER
from main import run_hedge_fund
from tools.api import (
    get_company_news,
    get_price_data,
    get_prices,
    get_financial_metrics,
    get_insider_trades,
)
from utils.display import print_backtest_results, format_backtest_row
from typing_extensions import Callable
import io
import contextlib
import argparse

init(autoreset=True)


class Backtester:
    def __init__(
        self,
        agent: Callable,
        tickers: list[str],
        start_date: str,
        end_date: str,
        initial_capital: float,
        model_name: str = "gpt-4o",
        model_provider: str = "OpenAI",
        selected_analysts: list[str] = [],
        initial_margin_requirement: float = 0.0,
    ):
        """
        :param agent: The trading agent (Callable).
        :param tickers: List of tickers to backtest.
        :param start_date: Start date string (YYYY-MM-DD).
        :param end_date: End date string (YYYY-MM-DD).
        :param initial_capital: Starting portfolio cash.
        :param model_name: Which LLM model name to use (gpt-4, etc).
        :param model_provider: Which LLM provider (OpenAI, etc).
        :param selected_analysts: List of analyst names or IDs to incorporate.
        :param initial_margin_requirement: The margin ratio (e.g. 0.5 = 50%).
        """
        self.agent = agent
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.model_name = model_name
        self.model_provider = model_provider
        self.selected_analysts = selected_analysts if selected_analysts else []

        # Initialize portfolio with support for long/short positions
        self.portfolio_values = []
        self.portfolio = {
            "cash": initial_capital,
            "margin_used": 0.0,  # total margin usage across all short positions
            "margin_requirement": initial_margin_requirement,  # The margin ratio required for shorts
            "positions": {
                ticker: {
                    "long": 0,               # Number of shares held long
                    "short": 0,              # Number of shares held short
                    "long_cost_basis": 0.0,  # Average cost basis per share (long)
                    "short_cost_basis": 0.0, # Average cost basis per share (short)
                    "short_margin_used": 0.0 # Dollars of margin used for this ticker's short
                } for ticker in tickers
            },
            "realized_gains": {
                ticker: {
                    "long": 0.0,   # Realized gains from long positions
                    "short": 0.0,  # Realized gains from short positions
                } for ticker in tickers
            }
        }

    def execute_trade(self, ticker: str, action: str, quantity: float, current_price: float):
        """
        Execute trades with support for both long and short positions.
        `quantity` is the number of shares the agent wants to buy/sell/short/cover.
        We will only trade integer shares to keep it simple.
        Handles NaN/inf inputs for quantity and price.
        """
        # --- NaN/inf Checks --- 
        if math.isnan(quantity) or math.isinf(quantity) or quantity <= 0:
            print(f"Warning: Invalid quantity ({quantity}) for {ticker} {action}. Skipping trade.")
            return 0
        if math.isnan(current_price) or math.isinf(current_price) or current_price <= 0:
            print(f"Warning: Invalid current_price ({current_price}) for {ticker} {action}. Skipping trade.")
            return 0
        # --- End NaN/inf Checks ---

        # Convert valid quantity to integer
        quantity = int(quantity)
        position = self.portfolio["positions"][ticker]

        if action == "buy":
            cost = quantity * current_price
            if cost <= self.portfolio["cash"]:
                # Weighted average cost basis for the new total
                old_shares = position["long"]
                old_cost_basis = position["long_cost_basis"]
                new_shares = quantity
                total_shares = old_shares + new_shares

                if total_shares > 0:
                    total_old_cost = old_cost_basis * old_shares
                    total_new_cost = cost
                    # Avoid NaN in cost basis if old_cost_basis was 0 and old_shares was 0
                    if math.isnan(total_old_cost):
                         total_old_cost = 0.0
                    position["long_cost_basis"] = (total_old_cost + total_new_cost) / total_shares

                position["long"] += quantity
                self.portfolio["cash"] -= cost
                return quantity
            else:
                # Calculate maximum affordable quantity, ensure price is valid
                max_quantity_float = self.portfolio["cash"] / current_price
                if math.isnan(max_quantity_float) or math.isinf(max_quantity_float) or max_quantity_float <= 0:
                     max_quantity = 0
                else:
                     max_quantity = int(max_quantity_float)

                if max_quantity > 0:
                    cost = max_quantity * current_price
                    old_shares = position["long"]
                    old_cost_basis = position["long_cost_basis"]
                    total_shares = old_shares + max_quantity

                    if total_shares > 0:
                        total_old_cost = old_cost_basis * old_shares
                        total_new_cost = cost
                        if math.isnan(total_old_cost):
                             total_old_cost = 0.0
                        position["long_cost_basis"] = (total_old_cost + total_new_cost) / total_shares

                    position["long"] += max_quantity
                    self.portfolio["cash"] -= cost
                    return max_quantity
                return 0

        elif action == "sell":
            # You can only sell as many as you own
            sell_quantity = min(quantity, position["long"])
            if sell_quantity > 0:
                # Realized gain/loss using average cost basis
                avg_cost_per_share = position["long_cost_basis"] if position["long"] > 0 else 0
                if math.isnan(avg_cost_per_share):
                     avg_cost_per_share = 0.0 # Treat NaN cost basis as 0

                realized_gain = (current_price - avg_cost_per_share) * sell_quantity
                # Check if realized_gain is NaN/inf
                if not (math.isnan(realized_gain) or math.isinf(realized_gain)):
                    self.portfolio["realized_gains"][ticker]["long"] += realized_gain
                else:
                     print(f"Warning: Calculated NaN/inf realized gain for {ticker} sell. Gain not recorded.")


                position["long"] -= sell_quantity
                self.portfolio["cash"] += sell_quantity * current_price

                if position["long"] == 0:
                    position["long_cost_basis"] = 0.0

                return sell_quantity
            return 0 # Return 0 if sell_quantity is 0

        elif action == "short":
            """
            Typical short sale flow:
              1) Receive proceeds = current_price * quantity
              2) Post margin_required = proceeds * margin_ratio
              3) Net effect on cash = +proceeds - margin_required
            """
            proceeds = current_price * quantity
            margin_required = proceeds * self.portfolio["margin_requirement"]

            # Check for NaN/inf in calculated values
            if math.isnan(proceeds) or math.isinf(proceeds) or math.isnan(margin_required) or math.isinf(margin_required):
                 print(f"Warning: NaN/inf calculated for proceeds/margin in {ticker} short. Skipping trade.")
                 return 0

            if margin_required <= self.portfolio["cash"]:
                # Weighted average short cost basis
                old_short_shares = position["short"]
                old_cost_basis = position["short_cost_basis"]
                new_shares = quantity
                total_shares = old_short_shares + new_shares

                if total_shares > 0:
                    total_old_cost = old_cost_basis * old_short_shares
                    total_new_cost = current_price * new_shares
                    if math.isnan(total_old_cost):
                         total_old_cost = 0.0
                    position["short_cost_basis"] = (total_old_cost + total_new_cost) / total_shares

                position["short"] += quantity

                # Update margin usage
                position["short_margin_used"] += margin_required
                self.portfolio["margin_used"] += margin_required

                # Increase cash by proceeds, then subtract the required margin
                self.portfolio["cash"] += proceeds
                self.portfolio["cash"] -= margin_required
                return quantity
            else:
                # Calculate maximum shortable quantity
                margin_ratio = self.portfolio["margin_requirement"]
                if margin_ratio > 0:
                     max_quantity_float = self.portfolio["cash"] / (current_price * margin_ratio)
                     if math.isnan(max_quantity_float) or math.isinf(max_quantity_float) or max_quantity_float <= 0:
                          max_quantity = 0
                     else:
                          max_quantity = int(max_quantity_float)
                else:
                    # If margin requirement is 0 or less, technically infinite shorting allowed,
                    # but practically limited by available shares or broker rules. Prevent division by zero.
                    # For simulation, let's disallow shorting if margin req is not positive.
                    max_quantity = 0
                    print(f"Warning: Cannot short {ticker} with non-positive margin requirement ({margin_ratio}).")

                if max_quantity > 0:
                    proceeds = current_price * max_quantity
                    margin_required = proceeds * margin_ratio
                    # Re-check for NaN/inf after calculating with max_quantity
                    if math.isnan(proceeds) or math.isinf(proceeds) or math.isnan(margin_required) or math.isinf(margin_required):
                         print(f"Warning: NaN/inf calculated for proceeds/margin (max_quantity) in {ticker} short. Skipping trade.")
                         return 0

                    old_short_shares = position["short"]
                    old_cost_basis = position["short_cost_basis"]
                    total_shares = old_short_shares + max_quantity

                    if total_shares > 0:
                        total_old_cost = old_cost_basis * old_short_shares
                        total_new_cost = current_price * max_quantity
                        if math.isnan(total_old_cost):
                             total_old_cost = 0.0
                        position["short_cost_basis"] = (total_old_cost + total_new_cost) / total_shares

                    position["short"] += max_quantity
                    position["short_margin_used"] += margin_required
                    self.portfolio["margin_used"] += margin_required

                    self.portfolio["cash"] += proceeds
                    self.portfolio["cash"] -= margin_required
                    return max_quantity
                return 0

        elif action == "cover":
            """
            When covering shares:
              1) Pay cover cost = current_price * quantity
              2) Release a proportional share of the margin
              3) Net effect on cash = -cover_cost + released_margin
            """
            cover_quantity = min(quantity, position["short"])
            if cover_quantity > 0:
                cover_cost = current_price * cover_quantity
                avg_short_price = position["short_cost_basis"] if position["short"] > 0 else 0
                if math.isnan(avg_short_price):
                     avg_short_price = 0.0 # Treat NaN cost basis as 0
                if math.isnan(cover_cost) or math.isinf(cover_cost):
                     print(f"Warning: NaN/inf calculated cover cost for {ticker} cover. Skipping trade.")
                     return 0

                realized_gain = (avg_short_price - current_price) * cover_quantity

                portion = 1.0
                if position["short"] > 0:
                    portion = cover_quantity / position["short"]

                margin_to_release = portion * position["short_margin_used"]
                if math.isnan(margin_to_release) or math.isinf(margin_to_release):
                     print(f"Warning: NaN/inf calculated margin to release for {ticker} cover. Setting to 0.")
                     margin_to_release = 0.0

                position["short"] -= cover_quantity
                position["short_margin_used"] -= margin_to_release
                self.portfolio["margin_used"] -= margin_to_release

                # Pay the cost to cover, but get back the released margin
                self.portfolio["cash"] += margin_to_release
                self.portfolio["cash"] -= cover_cost

                if not (math.isnan(realized_gain) or math.isinf(realized_gain)):
                    self.portfolio["realized_gains"][ticker]["short"] += realized_gain
                else:
                     print(f"Warning: Calculated NaN/inf realized gain for {ticker} cover. Gain not recorded.")

                if position["short"] == 0:
                    position["short_cost_basis"] = 0.0
                    position["short_margin_used"] = 0.0

                return cover_quantity
            return 0 # Return 0 if cover_quantity is 0

        return 0 # Return 0 if action is not recognized or quantity is invalid

    def calculate_portfolio_value(self, current_prices):
        """
        Calculate total portfolio value, including:
          - cash
          - market value of long positions
          - unrealized gains/losses for short positions
        """
        total_value = self.portfolio["cash"]

        for ticker in self.tickers:
            position = self.portfolio["positions"][ticker]
            price = current_prices[ticker]

            # Long position value
            long_value = position["long"] * price
            total_value += long_value

            # Short position unrealized PnL = short_shares * (short_cost_basis - current_price)
            if position["short"] > 0:
                total_value += position["short"] * (position["short_cost_basis"] - price)

        return total_value

    def prefetch_data(self):
        """Pre-fetch all data needed for the backtest period."""
        print("\nPre-fetching data for the entire backtest period...")

        # Convert end_date string to datetime, fetch up to 1 year before
        end_date_dt = datetime.strptime(self.end_date, "%Y-%m-%d")
        start_date_dt = end_date_dt - relativedelta(years=1)
        start_date_str = start_date_dt.strftime("%Y-%m-%d")

        for ticker in self.tickers:
            # Fetch price data for the entire period, plus 1 year
            get_prices(ticker, start_date_str, self.end_date)

            # Fetch financial metrics
            get_financial_metrics(ticker, self.end_date, limit=10)

            # Fetch insider trades
            get_insider_trades(ticker, self.end_date, start_date=self.start_date, limit=1000)

            # Fetch company news
            get_company_news(ticker, self.end_date, start_date=self.start_date, limit=1000)

        print("Data pre-fetch complete.")

    def parse_agent_response(self, agent_output):
        """Parse JSON output from the agent (fallback to 'hold' if invalid)."""
        import json

        try:
            decision = json.loads(agent_output)
            return decision
        except Exception:
            print(f"Error parsing action: {agent_output}")
            return {"action": "hold", "quantity": 0}

    def run_backtest(self):
        # Pre-fetch all data at the start
        self.prefetch_data()

        dates = pd.date_range(self.start_date, self.end_date, freq="B")
        table_rows = []
        performance_metrics = {
            'sharpe_ratio': None,
            'sortino_ratio': None,
            'max_drawdown': None,
            'long_short_ratio': None,
            'gross_exposure': None,
            'net_exposure': None
        }

        print("\nStarting backtest...")

        # Initialize portfolio values list with initial capital
        if len(dates) > 0:
            self.portfolio_values = [{"Date": dates[0], "Portfolio Value": self.initial_capital}]
        else:
            self.portfolio_values = []

        for current_date in dates:
            lookback_start = (current_date - timedelta(days=30)).strftime("%Y-%m-%d")
            current_date_str = current_date.strftime("%Y-%m-%d")
            previous_date_str = (current_date - timedelta(days=1)).strftime("%Y-%m-%d")

            # Skip if there's no prior day to look back (i.e., first date in the range)
            if lookback_start == current_date_str:
                continue

            # Get current prices for all tickers
            try:
                current_prices = {}
                missing_data = False
                
                for ticker in self.tickers:
                    try:
                        price_data = get_price_data(ticker, previous_date_str, current_date_str)
                        if price_data.empty:
                            print(f"Warning: No price data for {ticker} on {current_date_str}")
                            missing_data = True
                            break
                        current_prices[ticker] = price_data.iloc[-1]["close"]
                    except Exception as e:
                        print(f"Error fetching price for {ticker} between {previous_date_str} and {current_date_str}: {e}")
                        missing_data = True
                        break
                
                if missing_data:
                    print(f"Skipping trading day {current_date_str} due to missing price data")
                    continue
                
            except Exception as e:
                # If there's a general API error, log it and skip this day
                print(f"Error fetching prices for {current_date_str}: {e}")
                continue

            # ---------------------------------------------------------------
            # 1) Execute the agent's trades
            # ---------------------------------------------------------------
            output = self.agent(
                tickers=self.tickers,
                start_date=lookback_start,
                end_date=current_date_str,
                portfolio=self.portfolio,
                model_name=self.model_name,
                model_provider=self.model_provider,
                selected_analysts=self.selected_analysts,
            )
            decisions = output["decisions"]
            analyst_signals = output["analyst_signals"]

            # Execute trades for each ticker
            executed_trades = {}
            for ticker in self.tickers:
                decision = decisions.get(ticker, {"action": "hold", "quantity": 0})
                action, quantity = decision.get("action", "hold"), decision.get("quantity", 0)

                executed_quantity = self.execute_trade(ticker, action, quantity, current_prices[ticker])
                executed_trades[ticker] = executed_quantity

            # ---------------------------------------------------------------
            # 2) Now that trades have executed trades, recalculate the final
            #    portfolio value for this day.
            # ---------------------------------------------------------------
            total_value = self.calculate_portfolio_value(current_prices)

            # Also compute long/short exposures for final post‐trade state
            long_exposure = sum(
                self.portfolio["positions"][t]["long"] * current_prices[t]
                for t in self.tickers
            )
            short_exposure = sum(
                self.portfolio["positions"][t]["short"] * current_prices[t]
                for t in self.tickers
            )

            # Calculate gross and net exposures
            gross_exposure = long_exposure + short_exposure
            net_exposure = long_exposure - short_exposure
            long_short_ratio = (
                long_exposure / short_exposure if short_exposure > 1e-9 else float('inf')
            )

            # Track each day's portfolio value in self.portfolio_values
            self.portfolio_values.append({
                "Date": current_date,
                "Portfolio Value": total_value,
                "Long Exposure": long_exposure,
                "Short Exposure": short_exposure,
                "Gross Exposure": gross_exposure,
                "Net Exposure": net_exposure,
                "Long/Short Ratio": long_short_ratio
            })

            # ---------------------------------------------------------------
            # 3) Build the table rows to display
            # ---------------------------------------------------------------
            date_rows = []

            # For each ticker, record signals/trades
            for ticker in self.tickers:
                ticker_signals = {}
                for agent_name, signals in analyst_signals.items():
                    if ticker in signals:
                        ticker_signals[agent_name] = signals[ticker]

                bullish_count = len([s for s in ticker_signals.values() if s.get("signal", "").lower() == "bullish"])
                bearish_count = len([s for s in ticker_signals.values() if s.get("signal", "").lower() == "bearish"])
                neutral_count = len([s for s in ticker_signals.values() if s.get("signal", "").lower() == "neutral"])

                # Calculate net position value
                pos = self.portfolio["positions"][ticker]
                long_val = pos["long"] * current_prices[ticker]
                short_val = pos["short"] * current_prices[ticker]
                net_position_value = long_val - short_val

                # Get the action and quantity from the decisions
                action = decisions.get(ticker, {}).get("action", "hold")
                quantity = executed_trades.get(ticker, 0)
                
                # Append the agent action to the table rows
                date_rows.append(
                    format_backtest_row(
                        date=current_date_str,
                        ticker=ticker,
                        action=action,
                        quantity=quantity,
                        price=current_prices[ticker],
                        shares_owned=pos["long"] - pos["short"],  # net shares
                        position_value=net_position_value,
                        bullish_count=bullish_count,
                        bearish_count=bearish_count,
                        neutral_count=neutral_count,
                    )
                )
            # ---------------------------------------------------------------
            # 4) Calculate performance summary metrics
            # ---------------------------------------------------------------
            # Calculate portfolio return vs. initial capital
            # The realized gains are already reflected in cash balance, so we don't add them separately
            portfolio_return = (total_value / self.initial_capital - 1) * 100

            # Add summary row for this day
            date_rows.append(
                format_backtest_row(
                    date=current_date_str,
                    ticker="",
                    action="",
                    quantity=0,
                    price=0,
                    shares_owned=0,
                    position_value=0,
                    bullish_count=0,
                    bearish_count=0,
                    neutral_count=0,
                    is_summary=True,
                    total_value=total_value,
                    return_pct=portfolio_return,
                    cash_balance=self.portfolio["cash"],
                    total_position_value=total_value - self.portfolio["cash"],
                    sharpe_ratio=performance_metrics["sharpe_ratio"],
                    sortino_ratio=performance_metrics["sortino_ratio"],
                    max_drawdown=performance_metrics["max_drawdown"],
                ),
            )

            table_rows.extend(date_rows)
            print_backtest_results(table_rows)

            # Update performance metrics if we have enough data
            if len(self.portfolio_values) > 3:
                self._update_performance_metrics(performance_metrics)

        # Store the final performance metrics for reference in analyze_performance
        self.performance_metrics = performance_metrics
        return performance_metrics

    def _update_performance_metrics(self, performance_metrics):
        """Helper method to update performance metrics using daily returns."""
        values_df = pd.DataFrame(self.portfolio_values).set_index("Date")
        values_df["Daily Return"] = values_df["Portfolio Value"].pct_change()
        clean_returns = values_df["Daily Return"].dropna()

        if len(clean_returns) < 2:
            return  # not enough data points

        # Assumes 252 trading days/year
        daily_risk_free_rate = 0.0434 / 252
        excess_returns = clean_returns - daily_risk_free_rate
        mean_excess_return = excess_returns.mean()
        std_excess_return = excess_returns.std()

        # Sharpe ratio
        if std_excess_return > 1e-12:
            performance_metrics["sharpe_ratio"] = np.sqrt(252) * (mean_excess_return / std_excess_return)
        else:
            performance_metrics["sharpe_ratio"] = 0.0

        # Sortino ratio
        negative_returns = excess_returns[excess_returns < 0]
        if len(negative_returns) > 0:
            downside_std = negative_returns.std()
            if downside_std > 1e-12:
                performance_metrics["sortino_ratio"] = np.sqrt(252) * (mean_excess_return / downside_std)
            else:
                performance_metrics["sortino_ratio"] = float('inf') if mean_excess_return > 0 else 0
        else:
            performance_metrics["sortino_ratio"] = float('inf') if mean_excess_return > 0 else 0

        # Maximum drawdown (ensure it's stored as a negative percentage)
        rolling_max = values_df["Portfolio Value"].cummax()
        drawdown = (values_df["Portfolio Value"] - rolling_max) / rolling_max
        
        if len(drawdown) > 0:
            min_drawdown = drawdown.min()
            # Store as a negative percentage
            performance_metrics["max_drawdown"] = min_drawdown * 100
            
            # Store the date of max drawdown for reference
            if min_drawdown < 0:
                performance_metrics["max_drawdown_date"] = drawdown.idxmin().strftime('%Y-%m-%d')
            else:
                performance_metrics["max_drawdown_date"] = None
        else:
            performance_metrics["max_drawdown"] = 0.0
            performance_metrics["max_drawdown_date"] = None

    def analyze_performance(self):
        """Creates a performance DataFrame, prints summary stats, and plots equity curve."""
        if not self.portfolio_values:
            print("No portfolio data found. Please run the backtest first.")
            return pd.DataFrame()

        performance_df = pd.DataFrame(self.portfolio_values).set_index("Date")
        if performance_df.empty:
            print("No valid performance data to analyze.")
            return performance_df

        final_portfolio_value = performance_df["Portfolio Value"].iloc[-1]
        total_return = ((final_portfolio_value - self.initial_capital) / self.initial_capital) * 100

        print(f"\n{Fore.WHITE}{Style.BRIGHT}PORTFOLIO PERFORMANCE SUMMARY:{Style.RESET_ALL}")
        print(f"Total Return: {Fore.GREEN if total_return >= 0 else Fore.RED}{total_return:.2f}%{Style.RESET_ALL}")
        
        # Print realized P&L for informational purposes only
        total_realized_gains = sum(
            self.portfolio["realized_gains"][ticker]["long"] + 
            self.portfolio["realized_gains"][ticker]["short"] 
            for ticker in self.tickers
        )
        print(f"Total Realized Gains/Losses: {Fore.GREEN if total_realized_gains >= 0 else Fore.RED}${total_realized_gains:,.2f}{Style.RESET_ALL}")

        # Plot the portfolio value over time
        plt.figure(figsize=(12, 6))
        plt.plot(performance_df.index, performance_df["Portfolio Value"], color="blue")
        plt.title("Portfolio Value Over Time")
        plt.ylabel("Portfolio Value ($)")
        plt.xlabel("Date")
        plt.grid(True)
        plt.show()

        # Compute daily returns
        performance_df["Daily Return"] = performance_df["Portfolio Value"].pct_change().fillna(0)
        daily_rf = 0.0434 / 252  # daily risk-free rate
        mean_daily_return = performance_df["Daily Return"].mean()
        std_daily_return = performance_df["Daily Return"].std()

        # Annualized Sharpe Ratio
        if std_daily_return != 0:
            annualized_sharpe = np.sqrt(252) * ((mean_daily_return - daily_rf) / std_daily_return)
        else:
            annualized_sharpe = 0
        print(f"\nSharpe Ratio: {Fore.YELLOW}{annualized_sharpe:.2f}{Style.RESET_ALL}")

        # Use the max drawdown value calculated during the backtest if available
        max_drawdown = getattr(self, 'performance_metrics', {}).get('max_drawdown')
        max_drawdown_date = getattr(self, 'performance_metrics', {}).get('max_drawdown_date')
        
        # If no value exists yet, calculate it
        if max_drawdown is None:
            rolling_max = performance_df["Portfolio Value"].cummax()
            drawdown = (performance_df["Portfolio Value"] - rolling_max) / rolling_max
            max_drawdown = drawdown.min() * 100
            max_drawdown_date = drawdown.idxmin().strftime('%Y-%m-%d') if pd.notnull(drawdown.idxmin()) else None

        if max_drawdown_date:
            print(f"Maximum Drawdown: {Fore.RED}{abs(max_drawdown):.2f}%{Style.RESET_ALL} (on {max_drawdown_date})")
        else:
            print(f"Maximum Drawdown: {Fore.RED}{abs(max_drawdown):.2f}%{Style.RESET_ALL}")

        # Win Rate
        winning_days = len(performance_df[performance_df["Daily Return"] > 0])
        total_days = max(len(performance_df) - 1, 1)
        win_rate = (winning_days / total_days) * 100
        print(f"Win Rate: {Fore.GREEN}{win_rate:.2f}%{Style.RESET_ALL}")

        # Average Win/Loss Ratio
        positive_returns = performance_df[performance_df["Daily Return"] > 0]["Daily Return"]
        negative_returns = performance_df[performance_df["Daily Return"] < 0]["Daily Return"]
        avg_win = positive_returns.mean() if not positive_returns.empty else 0
        avg_loss = abs(negative_returns.mean()) if not negative_returns.empty else 0
        if avg_loss != 0:
            win_loss_ratio = avg_win / avg_loss
        else:
            win_loss_ratio = float('inf') if avg_win > 0 else 0
        print(f"Win/Loss Ratio: {Fore.GREEN}{win_loss_ratio:.2f}{Style.RESET_ALL}")

        # Maximum Consecutive Wins / Losses
        returns_binary = (performance_df["Daily Return"] > 0).astype(int)
        if len(returns_binary) > 0:
            max_consecutive_wins = max((len(list(g)) for k, g in itertools.groupby(returns_binary) if k == 1), default=0)
            max_consecutive_losses = max((len(list(g)) for k, g in itertools.groupby(returns_binary) if k == 0), default=0)
        else:
            max_consecutive_wins = 0
            max_consecutive_losses = 0

        print(f"Max Consecutive Wins: {Fore.GREEN}{max_consecutive_wins}{Style.RESET_ALL}")
        print(f"Max Consecutive Losses: {Fore.RED}{max_consecutive_losses}{Style.RESET_ALL}")

        return performance_df


##### Run the Backtester (Refactored Core Logic) #####
def run_backtest_core(
    tickers: list[str],
    start_date: str,
    end_date: str,
    initial_capital: float = 100000.0, # Add default
    initial_margin_requirement: float = 0.0, # Add default
    selected_analysts: list[str] = None, # Add argument
    model_name: str = "gpt-4o", # Add argument
    model_provider: str = "OpenAI", # Add argument
):
    """Core logic to run the backtest. Callable directly."""
    # Agent function is run_hedge_fund from main.py
    agent_func = run_hedge_fund # Use the refactored wrapper from main.py

    # Ensure selected_analysts is a list for Backtester init
    analysts_to_use = selected_analysts if selected_analysts else []

    # Create Backtester instance
    backtester = Backtester(
        agent=agent_func,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        model_name=model_name,
        model_provider=model_provider,
        selected_analysts=analysts_to_use,
        initial_margin_requirement=initial_margin_requirement,
    )

    print(f"Running backtest for {', '.join(tickers)}...")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    if initial_margin_requirement > 0:
         print(f"Margin Requirement: {initial_margin_requirement:.1%}")
    print(f"Using analysts: {', '.join(a.title().replace('_', ' ') for a in analysts_to_use) if analysts_to_use else 'All'}")
    print(f"Using model: {model_provider} / {model_name}")
    print("-" * 30)

    # Redirect stdout to capture prints during backtest execution if needed
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    trade_log = None
    performance_metrics = None

    try:
         with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
             # Pre-fetch data first (important for efficiency)
             backtester.prefetch_data()
             # Run the backtest simulation loop
             backtester.run_backtest() # Assuming this updates internal state
             # Analyze performance
             performance_metrics = backtester.analyze_performance()
             # Try to get trade_log if it exists as an attribute
             if hasattr(backtester, 'trade_log') and isinstance(backtester.trade_log, pd.DataFrame):
                 trade_log = backtester.trade_log

    except Exception as e:
         print(f"Error during backtest execution: {e}")
         # Optionally re-raise or return error information
         return {
             "error": "Backtest execution failed",
             "details": str(e),
             "stdout": captured_stdout.getvalue(),
             "stderr": captured_stderr.getvalue(),
         }

    print("Backtest complete. Analyzing performance...")

    # Return results in a structured way
    return {
        "performance_metrics": performance_metrics,
        "trade_log": trade_log, # May be None if not implemented/retrieved
        "stdout": captured_stdout.getvalue(), # Include captured output
        "stderr": captured_stderr.getvalue(),
    }


# Keep the CLI entry point functional
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the backtesting system for the AI hedge fund")
    # Keep argparse setup as before, but add args for analysts/model if needed
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=100000.0,
        help="Initial cash position (default: 100000.0)",
    )
    parser.add_argument(
        "--margin-requirement",
        type=float,
        default=0.0,
        help="Initial margin requirement (default: 0.0)",
    )
    parser.add_argument("--tickers", type=str, required=True, help="Comma-separated list of stock ticker symbols")
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD). Defaults to 1 year before end date",
    )
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD). Defaults to today")
    # Add args for analysts and model to bypass questionary if needed
    parser.add_argument(
        "--analysts", type=str, help="Comma-separated list of analyst keys (skips interactive selection if provided)"
    )
    parser.add_argument(
        "--model", type=str, help="LLM model name (skips interactive selection if provided)"
    )

    args = parser.parse_args()

    tickers = [ticker.strip() for ticker in args.tickers.split(",")]

    # --- Interactive Selection (only if args not provided) ---
    selected_analysts = None
    if args.analysts:
         selected_analysts = [a.strip() for a in args.analysts.split(',')]
         print(f"Using specified analysts: {', '.join(Fore.GREEN + a.title().replace('_', ' ') + Style.RESET_ALL for a in selected_analysts)}\\n")
    else:
        # Interactive analyst selection
        choices = questionary.checkbox(
            "Use the Space bar to select/unselect analysts.",
            choices=[questionary.Choice(display, value=value) for display, value in ANALYST_ORDER],
            instruction="\n\nPress 'a' to toggle all.\n\nPress Enter when done to run the hedge fund.",
            validate=lambda x: len(x) > 0 or "You must select at least one analyst.",
            style=questionary.Style(
                [
                    ("checkbox-selected", "fg:green"),
                    ("selected", "fg:green noinherit"),
                    ("highlighted", "noinherit"),
                    ("pointer", "noinherit"),
                ]
            ),
        ).ask()
        if not choices:
            print("\n\nInterrupt received. Exiting...")
            sys.exit(0)
        else:
            selected_analysts = choices
            print(f"\nSelected analysts: {', '.join(Fore.GREEN + choice.title().replace('_', ' ') + Style.RESET_ALL for choice in selected_analysts)}\n")

    model_choice = None
    model_provider = "Unknown"
    if args.model:
        model_choice = args.model
        model_info = get_model_info(model_choice)
        if model_info:
            model_provider = model_info.provider.value
            print(f"\nUsing specified {Fore.CYAN}{model_provider}{Style.RESET_ALL} model: {Fore.GREEN + Style.BRIGHT}{model_choice}{Style.RESET_ALL}\n")
        else:
            print(f"\nUsing specified model (provider unknown): {Fore.GREEN + Style.BRIGHT}{model_choice}{Style.RESET_ALL}\n")
    else:
        # Interactive model selection
        model_choice_q = questionary.select(
            "Select your LLM model for backtesting:",
            choices=[questionary.Choice(display, value=value) for display, value, _ in LLM_ORDER],
            style=questionary.Style([
                ("selected", "fg:green bold"),
                ("pointer", "fg:green bold"),
                ("highlighted", "fg:green"),
                ("answer", "fg:green bold"),
            ])
        ).ask()
        if not model_choice_q:
            print("\n\nInterrupt received. Exiting...")
            sys.exit(0)
        else:
            model_choice = model_choice_q
            model_info = get_model_info(model_choice)
            if model_info:
                model_provider = model_info.provider.value
                print(f"\nSelected {Fore.CYAN}{model_provider}{Style.RESET_ALL} model: {Fore.GREEN + Style.BRIGHT}{model_choice}{Style.RESET_ALL}\n")
            else:
                print(f"\nSelected model: {Fore.GREEN + Style.BRIGHT}{model_choice}{Style.RESET_ALL}\n")

    # --- Date Handling (similar to main.py, maybe centralize later) ---
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    start_date = args.start_date
    if not start_date:
        try:
            end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
            # Default to 1 year for backtesting if start_date is missing
            start_date_dt = end_date_dt - relativedelta(years=1)
            start_date = start_date_dt.strftime("%Y-%m-%d")
        except ValueError:
            print(f"Error parsing end date '{end_date}'. Please use YYYY-MM-DD format.")
            sys.exit(1)
    else:
        try:
            datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            print(f"Error parsing start date '{start_date}'. Please use YYYY-MM-DD format.")
            sys.exit(1)

    if args.end_date:
        try:
            datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            print(f"Error parsing end date '{end_date}'. Please use YYYY-MM-DD format.")
            sys.exit(1)

    # --- Run Core Logic --- 
    results = run_backtest_core(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.initial_cash,
        initial_margin_requirement=args.margin_requirement,
        selected_analysts=selected_analysts,
        model_name=model_choice, # Pass the selected/specified model
        model_provider=model_provider # Pass the determined provider
    )

    # --- Display Results (CLI) --- 
    print("-" * 30)
    print(results.get("stdout", "")) # Print captured stdout
    if results.get("stderr"): 
        print(f"{Fore.YELLOW}Stderr Output:{Style.RESET_ALL}")
        print(results["stderr"])
        print("-" * 30)

    if "error" in results:
        print(f"{Fore.RED}Backtest failed: {results['error']}")
        if "details" in results:
             print(f"{Fore.RED}Details: {results['details']}")
    elif results.get("performance_metrics"):
         print(f"{Fore.GREEN}Backtest Performance Analysis:{Style.RESET_ALL}")
         # Use the existing display function for CLI
         print_backtest_results(results["performance_metrics"])
         
         # Optionally print trade log summary or save it
         if results.get("trade_log") is not None:
             print("\nTrade Log Summary (first 10 rows):")
             print(tabulate(results["trade_log"].head(10), headers='keys', tablefmt='psql'))
             # results["trade_log"].to_csv("backtest_trade_log.csv")
             # print("Full trade log saved to backtest_trade_log.csv")
    else:
        print(f"{Fore.YELLOW}Backtest completed but no performance metrics were returned.")

    print("-" * 30)
