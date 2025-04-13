import streamlit as st
# import subprocess # No longer needed
import os
from datetime import date
import io # For capturing stdout
import contextlib # For capturing stdout
import json # For pretty printing results
import pandas as pd # For displaying backtest results
import math # Import math for isnan check
# from tabulate import tabulate # No longer needed for webapp display
import deepl # NEW: Import DeepL library
from dateutil.relativedelta import relativedelta # NEW: Import relativedelta

# --- Re-enable core logic imports --- 
from main import run_hedge_fund_core 
from backtester import run_backtest_core
# -------------------------------------

# --- OpenAI Translation Imports (Removed as unused) ---
# --- Removed langchain_openai and langchain_core.messages imports ---
# ---------------------------------------

# --- Localization (i18n) Setup --- 
# Define translation strings
TRANSLATIONS = {
    "en": {
        "page_title": "AI Hedge Fund",
        "header_title": "📈 AI Hedge Fund Simulator & Backtester",
        "header_caption": "Use AI agents to simulate trading decisions and backtest strategies.",
        "config_header": "Configuration",
        "tickers_label": "Stock Tickers (comma-separated)",
        "tickers_help": "Enter the stock tickers you want to analyze, separated by commas.",
        "start_date_label": "Start Date (Optional)",
        "end_date_label": "End Date (Optional)",
        "show_reasoning_label": "Show Agent Reasoning (Simulation only)",
        "advanced_options_label": "Advanced Options (Optional)",
        "advanced_options_caption": "Currently uses default analysts and models.",
        "run_simulation_button": "🚀 Run Simulation",
        "run_backtest_button": "📊 Run Backtest",
        "simulation_spinner": "Running simulation for {tickers}...",
        "backtest_spinner": "Running backtest for {tickers}...",
        "simulation_complete": "Simulation complete!",
        "backtest_complete": "Backtest complete!",
        "enter_ticker_warning": "Please enter at least one ticker.",
        "select_start_date_warning": "Please select a Start Date for backtesting.",
        "final_decisions_header": "Final Decisions",
        "error_message": "Error: {error}",
        "error_details": "Details: {details}",
        "no_decisions_warning": "No final decisions were generated.",
        "agent_signals_header": "Individual Agent Signals",
        "investor_agents_header": "Investor Agent Signals", # Keep separate header for structure
        "analytical_agents_header": "Analytical Signals",
        "analysis_for_ticker": "Analysis for {ticker}",
        "signal_label": "Signal",
        "confidence_label": "Confidence",
        "reasoning_label": "Reasoning",
        "action_label": "Action",
        "quantity_label": "Quantity",
        "other_details_label": "Other Details",
        "confidence_not_provided": "Not Provided",
        "confidence_na": "N/A",
        "no_signals_warning": "No individual agent signals were generated or found.",
        "log_expander_title": "Simulation Output Log (Logs/Reasoning)",
        "backtest_log_expander_title": "Backtest Output Log",
        "stderr_warning": "Simulation Standard Error:",
        "backtest_stderr_warning": "Backtest Standard Error:",
        "error_unexpected": "An unexpected error occurred during simulation: {e}",
        "backtest_error_unexpected": "An unexpected error occurred during backtest: {e}",
        "log_before_error": "Captured Output Before Error:",
        "performance_metrics_header": "Performance Metrics",
        "trade_log_header": "Trade Log",
        "no_metrics_warning": "Backtest completed but no performance metrics were returned.",
        "no_trade_log_warning": "No trade log was generated or returned.",
        "disclaimer_header": "Disclaimer",
        "disclaimer_text": "This project is for educational and research purposes only. Not intended for real trading or investment. No warranties or guarantees provided. Past performance does not indicate future results. Consult a financial advisor for investment decisions."
    },
    "zh": {
        "page_title": "AI 对冲基金",
        "header_title": "📈 AI 对冲基金模拟器与回测器",
        "header_caption": "使用 AI Agent 模拟交易决策并回测策略。",
        "config_header": "配置",
        "tickers_label": "股票代码 (英文逗号分隔)",
        "tickers_help": "输入您想要分析的股票代码，用英文逗号分隔。",
        "start_date_label": "开始日期 (可选)",
        "end_date_label": "结束日期 (可选)",
        "show_reasoning_label": "显示 Agent 推理过程 (仅模拟)",
        "advanced_options_label": "高级选项 (可选)",
        "advanced_options_caption": "当前使用默认分析师和模型。",
        "run_simulation_button": "🚀 运行模拟",
        "run_backtest_button": "📊 运行回测",
        "simulation_spinner": "正在为 {tickers} 运行模拟...",
        "backtest_spinner": "正在为 {tickers} 运行回测...",
        "simulation_complete": "模拟完成！",
        "backtest_complete": "回测完成！",
        "enter_ticker_warning": "请输入至少一个股票代码。",
        "select_start_date_warning": "请为回测选择一个开始日期。",
        "final_decisions_header": "最终决策",
        "error_message": "错误: {error}",
        "error_details": "详情: {details}",
        "no_decisions_warning": "未能生成最终决策。",
        "agent_signals_header": "各 Agent 信号",
        "investor_agents_header": "投资策略 Agent 信号",
        "analytical_agents_header": "分析型 Agent 信号",
        "analysis_for_ticker": "对 {ticker} 的分析",
        "signal_label": "信号",
        "confidence_label": "置信度",
        "reasoning_label": "理由",
        "action_label": "操作",
        "quantity_label": "数量",
        "other_details_label": "其他详情",
        "confidence_not_provided": "未提供",
        "confidence_na": "不适用",
        "no_signals_warning": "未能生成或找到任何 Agent 信号。",
        "log_expander_title": "模拟输出日志 (日志/推理过程)",
        "backtest_log_expander_title": "回测输出日志",
        "stderr_warning": "模拟标准错误输出:",
        "backtest_stderr_warning": "回测标准错误输出:",
        "error_unexpected": "模拟过程中发生意外错误: {e}",
        "backtest_error_unexpected": "回测过程中发生意外错误: {e}",
        "log_before_error": "错误发生前的日志输出:",
        "performance_metrics_header": "性能指标",
        "trade_log_header": "交易日志",
        "no_metrics_warning": "回测已完成，但未返回性能指标。",
        "no_trade_log_warning": "未能生成或返回交易日志。",
        "disclaimer_header": "免责声明",
        "disclaimer_text": "本项目仅用于教育和研究目的。不构成真实交易或投资建议。不提供任何保证。过往表现不预示未来结果。投资决策请咨询财务顾问。"
    },
    "zh-Hant": { # NEW: Traditional Chinese
        "page_title": "AI 對沖基金",
        "header_title": "📈 AI 對沖基金模擬器與回測器",
        "header_caption": "使用 AI 代理分析股票並模擬交易決策或進行歷史回測。",
        "config_header": "設定",
        "tickers_label": "股票代號 (以逗號分隔)",
        "tickers_help": "輸入您想要分析的股票代號，例如：AAPL,MSFT,GOOG",
        "start_date_label": "開始日期 (可選)",
        "end_date_label": "結束日期 (可選)",
        "show_reasoning_label": "顯示代理推理過程 (僅模擬)",
        "advanced_options_label": "進階選項 (可選)",
        "advanced_options_caption": "在此處配置模型提供者、特定模型名稱等。",
        "run_simulation_button": "🚀 運行模擬",
        "run_backtest_button": "📊 運行回測",
        "simulation_spinner": "正在為 {tickers} 運行 AI 模擬...",
        "backtest_spinner": "正在為 {tickers} 運行歷史回測...",
        "simulation_complete": "模擬完成！",
        "backtest_complete": "回測完成！",
        "final_decisions_header": "最終決策",
        "agent_signals_header": "個別代理信號",
        "investor_agents_header": "投資者代理", # Grouping Header
        "analytical_agents_header": "分析型代理", # Grouping Header
        "performance_metrics_header": "績效指標",
        "trade_log_header": "交易日誌",
        "disclaimer_header": "免責聲明",
        "disclaimer_text": "此專案僅供教育和研究目的。不用於真實交易或投資建議。不提供任何保證。過去的表現並不代表未來的結果。請諮詢財務顧問。",
        "error_unexpected": "發生意外錯誤：{e}",
        "backtest_error_unexpected": "回測期間發生意外錯誤：{e}",
        "error_message": "錯誤：{error}",
        "error_details": "詳情：{details}",
        "log_expander_title": "查看模擬日誌",
        "backtest_log_expander_title": "查看回測日誌",
        "log_before_error": "錯誤發生前的日誌：",
        "stderr_warning": "標準錯誤輸出 (可能包含警告或錯誤)：",
        "backtest_stderr_warning": "回測標準錯誤輸出：",
        "enter_ticker_warning": "請至少輸入一個股票代號。",
        "select_start_date_warning": "請為回測選擇一個開始日期。",
        "no_decisions_warning": "未能生成最終決策。",
        "no_signals_warning": "未能生成或找到個別代理信號。",
        "no_metrics_warning": "未能計算績效指標。",
        "no_trade_log_warning": "沒有交易記錄可顯示。",
        "analysis_for_ticker": "對 {ticker} 的分析",
        "signal_label": "信號",
        "confidence_label": "置信度",
        "reasoning_label": "理由",
        "action_label": "操作",
        "quantity_label": "數量",
        "other_details_label": "其他詳情",
        "confidence_not_provided": "未提供",
        "confidence_na": "無效數據", # For NaN confidence
        "value_label": "值", # For backtest metrics table header
        # Add any other required keys here, translating them
        "Initial Capital": "初始資本",
        "Final Portfolio Value": "最終投資組合價值",
        "Total Return": "總回報率",
        "Max Drawdown": "最大回撤",
        "Sharpe Ratio": "夏普比率",
        "Sortino Ratio": "索提諾比率",
        "Profit Factor": "盈利因子",
        "Total Trades": "總交易次數",
        "Winning Trades": "盈利交易次數",
        "Losing Trades": "虧損交易次數",
        "Win Rate": "勝率",
        "Average Trade Return": "平均交易回報率",
        "Average Win Return": "平均盈利回報率",
        "Average Loss Return": "平均虧損回報率",
        "Date": "日期",
        "Ticker": "代號",
        "Action": "操作",
        "Quantity": "數量",
        "Price": "價格",
        "Commission": "佣金",
        "Cash": "現金",
        "Portfolio Value": "投資組合價值",
    }
}

# --- DeepL Translator Client Initialization (NEW) ---
deepl_translator = None
if os.getenv("DEEPL_API_KEY"):
    try:
        auth_key = os.getenv("DEEPL_API_KEY")
        deepl_translator = deepl.Translator(auth_key)
        # Verify connection by checking usage (optional but good practice)
        usage = deepl_translator.get_usage()
        if usage.character.limit_exceeded:
            print("WARNING: DeepL character limit exceeded. Translation might fail.")
        else: 
            print(f"INFO: DeepL Translator initialized. Characters used: {usage.character.count}/{usage.character.limit}")
    except Exception as e:
        print(f"ERROR: Failed to initialize DeepL Translator: {e}")
        deepl_translator = None # Ensure it's None if init fails
else:
    print("WARNING: DEEPL_API_KEY not found in environment variables. DeepL translation will be skipped.")
# --------------------------------------------------

# --- Updated translation function using DeepL --- 
def translate_text(text, lang):
    # --- Fixed keywords translation --- 
    signal_map_zh = { # Simplified
        "bullish": "看涨", "bearish": "看跌", "neutral": "中性",
        "buy": "买入", "sell": "卖出", "short": "做空",
        "cover": "平仓", "hold": "持有"
    }
    signal_map_hant = { # Traditional (NEW)
        "bullish": "看漲", "bearish": "看跌", "neutral": "中性",
        "buy": "買入", "sell": "賣出", "short": "做空",
        "cover": "平倉", "hold": "持有"
    }

    if lang == "zh":
        if isinstance(text, str) and text.lower() in signal_map_zh:
             return signal_map_zh[text.lower()]
        target_lang_deepl = "ZH" # Simplified Chinese for DeepL
        current_map = signal_map_zh
    elif lang == "zh-Hant":
        if isinstance(text, str) and text.lower() in signal_map_hant:
             return signal_map_hant[text.lower()]
        target_lang_deepl = "ZH-HANT" # Traditional Chinese for DeepL (NEW)
        current_map = signal_map_hant
    else:
        # Return original text for English or other unsupported languages
        return text 

    # --- Dynamic Text Translation using DeepL (Logic adjusted for target_lang) ---
    if isinstance(text, str) and len(text) > 10 and deepl_translator and text.lower() not in current_map:
        try:
            result = deepl_translator.translate_text(text, target_lang=target_lang_deepl) 
            if result and result.text:
                return result.text
            else:
                print(f"WARNING: DeepL returned empty result for: {text[:50]}...")
                return f"{text} (翻譯失敗)"
        except deepl.DeepLException as e:
            print(f"ERROR: DeepL translation failed: {e}")
            return f"{text} (翻譯錯誤)" 
        except Exception as e:
            print(f"ERROR: Unexpected error during DeepL translation: {e}")
            return f"{text} (翻譯意外錯誤)"
    elif isinstance(text, str):
         # Keep very short strings or strings when client is unavailable, or already translated keywords
         return text 
    else:
        # Return non-strings as is
        return text

# --- Agent Details (Replace with actual URLs if available) ---
# Ideally, move this to a separate config/utils file
AGENT_DETAILS = {
    # Investor Agents
    "ben_graham": {"photo_url": "https://via.placeholder.com/100.png?text=Ben+Graham", "display_name": "Ben Graham"},
    "bill_ackman": {"photo_url": "https://via.placeholder.com/100.png?text=Bill+Ackman", "display_name": "Bill Ackman"},
    "cathie_wood": {"photo_url": "https://via.placeholder.com/100.png?text=Cathie+Wood", "display_name": "Cathie Wood"},
    "charlie_munger": {"photo_url": "https://via.placeholder.com/100.png?text=Charlie+Munger", "display_name": "Charlie Munger"},
    "michael_burry": {"photo_url": "https://via.placeholder.com/100.png?text=Michael+Burry", "display_name": "Michael Burry"},
    "peter_lynch": {"photo_url": "https://via.placeholder.com/100.png?text=Peter+Lynch", "display_name": "Peter Lynch"},
    "phil_fisher": {"photo_url": "https://via.placeholder.com/100.png?text=Phil+Fisher", "display_name": "Phil Fisher"},
    "stanley_druckenmiller": {"photo_url": "https://via.placeholder.com/100.png?text=Stan+Druckenmiller", "display_name": "Stanley Druckenmiller"},
    "warren_buffett": {"photo_url": "https://via.placeholder.com/100.png?text=Warren+Buffett", "display_name": "Warren Buffett"},
    # Analytical Agents
    "technical_analyst": {"photo_url": "https://via.placeholder.com/100.png?text=Technicals", "display_name": "Technical Analyst"},
    "fundamentals": {"photo_url": "https://via.placeholder.com/100.png?text=Fundamentals", "display_name": "Fundamentals Analyst"},
    "sentiment": {"photo_url": "https://via.placeholder.com/100.png?text=Sentiment", "display_name": "Sentiment Analyst"},
    "valuation": {"photo_url": "https://via.placeholder.com/100.png?text=Valuation", "display_name": "Valuation Analyst"},
    # Manager Agents (Optional, if their output is ever shown here)
    "risk_manager": {"photo_url": "https://via.placeholder.com/100.png?text=Risk+Manager", "display_name": "Risk Manager"},
    "portfolio_manager": {"photo_url": "https://via.placeholder.com/100.png?text=Portfolio+Mgr", "display_name": "Portfolio Manager"},
}
# Define which keys correspond to analytical agents for grouping
ANALYTICAL_AGENT_KEYS = ["technical_analyst", "fundamentals", "sentiment", "valuation"]

# Removed DEFAULT_AGENT_DETAIL to avoid showing 'Unknown Agent'

# Removed the display_dict_as_expander helper function

# --- Set Page Config FIRST --- 
st.set_page_config(page_title="AI Hedge Fund / AI 对冲基金 / AI 對沖基金", layout="wide") # Updated neutral title

# --- Streamlit App Layout --- 

# Language selection in the sidebar (Updated options)
def format_language(lang_code):
    if lang_code == "en":
        return "English"
    elif lang_code == "zh":
        return "简体中文"
    elif lang_code == "zh-Hant":
        return "繁體中文" # NEW
    return lang_code # Fallback

language = st.sidebar.selectbox(
    "Language / 语言 / 語言", 
    options=["en", "zh", "zh-Hant"], # Added zh-Hant
    format_func=format_language, 
    key="language_selector"
)

# Get translated texts based on selection
TXT = TRANSLATIONS[language]

# Now set title and other elements using translated text
st.title(TXT["header_title"])
st.caption(TXT["header_caption"])

# --- Input Section ---
st.header(TXT["config_header"])
tickers_input = st.text_input(
    TXT["tickers_label"],
    value="AAPL,MSFT,NVDA",
    help=TXT["tickers_help"],
    key="tickers"
)

# --- Calculate default dates (NEW) ---
end_default = date.today()
start_default = end_default - relativedelta(months=3)
# ------------------------------------

col1, col2 = st.columns(2)
with col1:
    start_date_input = st.date_input(TXT["start_date_label"], value=start_default, key="start_date") # Use calculated default
with col2:
    # Default end date to today, ensuring it's a valid date object
    end_date_input = st.date_input(TXT["end_date_label"], value=end_default, key="end_date") # Use calculated default

show_reasoning_input = st.checkbox(TXT["show_reasoning_label"], key="show_reasoning")

# --- Advanced Options (Optional) ---
with st.expander(TXT["advanced_options_label"]):
    st.caption(TXT["advanced_options_caption"])

st.markdown("---") # Separator

# --- Action Buttons & Results Area ---
results_area = st.container()
log_expander_area = st.container() # Separate area for log expander

col_run, col_backtest = st.columns(2)

with col_run:
    if st.button(TXT["run_simulation_button"], use_container_width=True):
        if tickers_input:
            tickers_list = [ticker.strip().upper() for ticker in tickers_input.split(",")]
            start_date_str = start_date_input.strftime('%Y-%m-%d') if start_date_input else None
            end_date_str = end_date_input.strftime('%Y-%m-%d') if end_date_input else date.today().strftime('%Y-%m-%d')
            initial_portfolio_sim = { # Keep this setup for context
                "cash": 100000.0,
                "margin_used": 0.0,
                "margin_requirement": 0.0,
                "positions": {t: {"long": 0, "short": 0, "long_cost_basis": 0.0, "short_cost_basis": 0.0, "short_margin_used": 0.0} for t in tickers_list},
                "realized_gains": {t: {"long": 0.0, "short": 0.0} for t in tickers_list}
            }

            with results_area:
                results_area.empty()
                with log_expander_area:
                    log_expander_area.empty()

                with st.spinner(TXT["simulation_spinner"].format(tickers=tickers_input)):
                    # --- Re-enable core function call --- 
                    # Capture stdout/stderr
                    sim_stdout = io.StringIO()
                    sim_stderr = io.StringIO()
                    results = None
                    try:
                        with contextlib.redirect_stdout(sim_stdout), contextlib.redirect_stderr(sim_stderr):
                            results = run_hedge_fund_core(
                                tickers=tickers_list,
                                start_date=start_date_str, 
                                end_date=end_date_str,     
                                portfolio=initial_portfolio_sim, 
                                show_reasoning=show_reasoning_input,
                            )
                        st.success(TXT["simulation_complete"])
                    except Exception as e:
                        results_area.error(TXT["error_unexpected"].format(e=e))
                        # Display logs even on error
                        stdout_val = sim_stdout.getvalue()
                        stderr_val = sim_stderr.getvalue()
                        with log_expander_area:
                            st.warning(TXT["log_before_error"])
                            if stdout_val: st.code(stdout_val)
                            if stderr_val: st.code(stderr_val)
                    # ----------------------------------------
                    
                    # --- Display logic (using the results variable) --- 
                    results_area.subheader(TXT["final_decisions_header"])
                    # Display Final Decisions
                    if results and not results.get("error") and results.get("decisions"):
                        decisions_data = results["decisions"]
                        if isinstance(decisions_data, dict):
                            translated_decisions = {}
                            for ticker, data in decisions_data.items():
                                translated_data = data.copy()
                                translated_data['action'] = translate_text(data.get('action', ''), language)
                                # Ensure reasoning is handled even if missing
                                translated_data['reasoning'] = translate_text(data.get('reasoning', ''), language)
                                translated_decisions[ticker] = translated_data
                            
                            df_decisions = pd.DataFrame.from_dict(translated_decisions, orient='index')
                            df_decisions.index.name = 'Ticker'
                            cols_order = ['action', 'quantity', 'confidence', 'reasoning']
                            df_decisions = df_decisions[[col for col in cols_order if col in df_decisions.columns]]
                            df_decisions.rename(columns={
                                'action': TXT['action_label'], 'quantity': TXT['quantity_label'],
                                'confidence': TXT['confidence_label'], 'reasoning': TXT['reasoning_label']
                            }, inplace=True)
                            results_area.dataframe(df_decisions, use_container_width=True)
                        else:
                            results_area.json(decisions_data)
                    elif results and results.get("error"):
                        results_area.error(TXT["error_message"].format(error=results['error']))
                        if "details" in results:
                            results_area.error(TXT["error_details"].format(details=results['details']))
                    else:
                        results_area.warning(TXT["no_decisions_warning"])

                    # Display Individual Agent Signals
                    results_area.subheader(TXT["agent_signals_header"])
                    signals_displayed = False
                    if results and results.get("analyst_signals"):
                        signals_data = results["analyst_signals"]
                        if isinstance(signals_data, dict):
                            # Separate keys
                            investor_keys = [k for k in signals_data if k not in ANALYTICAL_AGENT_KEYS]
                            analytical_keys = [k for k in signals_data if k in ANALYTICAL_AGENT_KEYS]

                            # Display Investor Agents
                            if investor_keys:
                                results_area.markdown(f"### {TXT['investor_agents_header']}")
                                for agent_key in investor_keys:
                                    # ... (Agent display logic - unchanged, uses translate_text) ...
                                    details = AGENT_DETAILS.get(agent_key.replace('_agent', ''))
                                    if not details: continue
                                    signals_displayed = True
                                    display_name = details["display_name"]
                                    photo_url = details["photo_url"]
                                    results_area.markdown("---") 
                                    col1, col2 = results_area.columns([1, 5])
                                    with col1:
                                        st.image(photo_url, width=100, caption=display_name)
                                    with col2:
                                        st.markdown(f"#### {display_name}")
                                        agent_signals_per_ticker = signals_data.get(agent_key, {})
                                        if isinstance(agent_signals_per_ticker, dict):
                                            for ticker, signal_data in agent_signals_per_ticker.items():
                                                with st.expander(TXT["analysis_for_ticker"].format(ticker=ticker)):
                                                    # ... (Display signal, confidence, reasoning using translate_text) ...
                                                    if isinstance(signal_data, dict):
                                                        signal = signal_data.get("signal")
                                                        confidence = signal_data.get("confidence")
                                                        if signal:
                                                            translated_signal = translate_text(signal, language)
                                                            color = "green" if signal == "bullish" else "red" if signal == "bearish" else "orange"
                                                            st.markdown(f"**{TXT['signal_label']}:** :{color}[{translated_signal}]")
                                                        if confidence is not None:
                                                            try:
                                                                conf_float = float(confidence)
                                                                if not math.isnan(conf_float):
                                                                    st.write(f"**{TXT['confidence_label']}:** {conf_float:.1f}%")
                                                                else:
                                                                    st.write(f"**{TXT['confidence_label']}:** {TXT['confidence_na']}")
                                                            except (ValueError, TypeError):
                                                                 st.write(f"**{TXT['confidence_label']}:** {confidence}")
                                                        else:
                                                            st.write(f"**{TXT['confidence_label']}:** {TXT['confidence_not_provided']}")
                                                        reasoning = signal_data.get("reasoning")
                                                        if reasoning:
                                                            st.markdown(f"**{TXT['reasoning_label']}:**")
                                                            if isinstance(reasoning, str):
                                                                translated_reasoning = translate_text(reasoning, language)
                                                                st.markdown(f"> _{translated_reasoning}_ ") 
                                                            else:
                                                                 st.json(reasoning, expanded=False) # Display non-string reasoning as JSON
                                                        other_data = {k: v for k, v in signal_data.items() if k not in ["signal", "confidence", "reasoning"]}
                                                        if other_data:
                                                            st.markdown(f"**{TXT['other_details_label']}:**")
                                                            st.json(other_data, expanded=False)
                                                    else:
                                                        st.json(signal_data) # Fallback for unexpected format
                                        else:
                                            st.json(agent_signals_per_ticker) # Fallback
                                        
                            # Display Analytical Agents
                            if analytical_keys:
                                results_area.markdown(f"### {TXT['analytical_agents_header']}")
                                for agent_key in analytical_keys:
                                    # ... (Agent display logic - same as above, uses translate_text) ...
                                    details = AGENT_DETAILS.get(agent_key.replace('_agent', ''))
                                    if not details: continue
                                    signals_displayed = True
                                    display_name = details["display_name"]
                                    photo_url = details["photo_url"]
                                    results_area.markdown("---") 
                                    col1, col2 = results_area.columns([1, 5])
                                    with col1:
                                        st.image(photo_url, width=100, caption=display_name)
                                    with col2:
                                        st.markdown(f"#### {display_name}")
                                        agent_signals_per_ticker = signals_data.get(agent_key, {})
                                        if isinstance(agent_signals_per_ticker, dict):
                                            for ticker, signal_data in agent_signals_per_ticker.items():
                                                with st.expander(TXT["analysis_for_ticker"].format(ticker=ticker)):
                                                    # ... (Display signal, confidence, reasoning using translate_text) ...
                                                    if isinstance(signal_data, dict):
                                                        signal = signal_data.get("signal")
                                                        confidence = signal_data.get("confidence")
                                                        if signal:
                                                            translated_signal = translate_text(signal, language)
                                                            color = "green" if signal == "bullish" else "red" if signal == "bearish" else "orange"
                                                            st.markdown(f"**{TXT['signal_label']}:** :{color}[{translated_signal}]")
                                                        if confidence is not None:
                                                            try:
                                                                conf_float = float(confidence)
                                                                if not math.isnan(conf_float):
                                                                    st.write(f"**{TXT['confidence_label']}:** {conf_float:.1f}%")
                                                                else:
                                                                    st.write(f"**{TXT['confidence_label']}:** {TXT['confidence_na']}")
                                                            except (ValueError, TypeError):
                                                                 st.write(f"**{TXT['confidence_label']}:** {confidence}")
                                                        else:
                                                            st.write(f"**{TXT['confidence_label']}:** {TXT['confidence_not_provided']}")
                                                        reasoning = signal_data.get("reasoning")
                                                        if reasoning:
                                                            st.markdown(f"**{TXT['reasoning_label']}:**")
                                                            if isinstance(reasoning, str):
                                                                translated_reasoning = translate_text(reasoning, language)
                                                                st.markdown(f"> _{translated_reasoning}_ ") 
                                                            else:
                                                                 st.json(reasoning, expanded=False)
                                                        other_data = {k: v for k, v in signal_data.items() if k not in ["signal", "confidence", "reasoning"]}
                                                        if other_data:
                                                            st.markdown(f"**{TXT['other_details_label']}:**")
                                                            st.json(other_data, expanded=False)
                                                    else:
                                                        st.json(signal_data)
                                        else:
                                            st.json(agent_signals_per_ticker)
                        else:
                            results_area.warning("Analyst signals data is not in the expected dictionary format.")

                    if not signals_displayed and not (results and results.get("error")):
                        results_area.warning(TXT["no_signals_warning"])

                    # --- Log Display --- 
                    stdout_val = sim_stdout.getvalue()
                    stderr_val = sim_stderr.getvalue()
                    with log_expander_area:
                        if stdout_val:
                            with st.expander(TXT["log_expander_title"]):
                                st.code(stdout_val)
                        if stderr_val:
                            st.warning(TXT["stderr_warning"])
                            st.code(stderr_val)
                    # -----------------------------------------------------
        else:
            with results_area:
                st.warning(TXT["enter_ticker_warning"])

with col_backtest:
    if st.button(TXT["run_backtest_button"], use_container_width=True):
        if tickers_input:
            tickers_list = [ticker.strip().upper() for ticker in tickers_input.split(",")]
            # Backtesting typically requires a start date
            if not start_date_input:
                 with results_area:
                      results_area.empty() # Clear previous results
                      with log_expander_area:
                          log_expander_area.empty() # Clear previous logs
                      st.warning(TXT["select_start_date_warning"])
            else:
                start_date_str = start_date_input.strftime('%Y-%m-%d')
                end_date_str = end_date_input.strftime('%Y-%m-%d') if end_date_input else date.today().strftime('%Y-%m-%d')

                with results_area: # Display main results here
                    results_area.empty() # Clear previous results
                    with log_expander_area:
                         log_expander_area.empty() # Clear previous logs

                    with st.spinner(TXT["backtest_spinner"].format(tickers=tickers_input)):
                         # --- Re-enable core function call --- 
                         results = None # Initialize results
                         backtest_stdout = io.StringIO() # Capture stdout for backtest
                         backtest_stderr = io.StringIO() # Capture stderr for backtest
                         try:
                             with contextlib.redirect_stdout(backtest_stdout), contextlib.redirect_stderr(backtest_stderr):
                                 results = run_backtest_core(
                                     tickers=tickers_list,
                                     start_date=start_date_str,
                                     end_date=end_date_str,
                                 )
                             st.success(TXT["backtest_complete"])
                         except Exception as e:
                             results_area.error(TXT["backtest_error_unexpected"].format(e=e))
                             # Display logs even on error
                             stdout_val = backtest_stdout.getvalue()
                             stderr_val = backtest_stderr.getvalue()
                             with log_expander_area:
                                 st.warning(TXT["log_before_error"])
                                 if stdout_val: st.code(stdout_val)
                                 if stderr_val: st.code(stderr_val)
                         # ------------------------------------

                         # --- Display logic (using results) --- 
                         results_area.subheader(TXT["performance_metrics_header"])
                         if results and not results.get("error") and results.get("performance_metrics"):
                             metrics_df = pd.DataFrame.from_dict(results["performance_metrics"], orient='index', columns=[TXT['value_label']]) # Use translated label
                             # Translate index (metric names)
                             metrics_df.index = [translate_text(idx, language) for idx in metrics_df.index]
                             # Format values
                             for idx in metrics_df.index:
                                 try:
                                     metric_val = float(results["performance_metrics"][translate_text(idx, 'en')]) # Use original key for value lookup
                                     # ... (formatting logic remains same) ...
                                     # Determine if percentage or currency based on the English name before translation
                                     original_key = translate_text(idx, 'en') # Get original english key back
                                     if 'Return' in original_key or 'Rate' in original_key or 'Ratio' in original_key or 'Volatility' in original_key or 'Drawdown' in original_key:
                                          metrics_df.loc[idx, TXT['value_label']] = f"{metric_val:.2%}"
                                     elif 'Profit' in original_key or 'Capital' in original_key or 'Value' in original_key:
                                          metrics_df.loc[idx, TXT['value_label']] = f"${metric_val:,.2f}"
                                     else: # Default formatting for other numbers
                                         metrics_df.loc[idx, TXT['value_label']] = f"{metric_val:,.2f}"
                                 except (ValueError, TypeError, KeyError):
                                     # Keep original string value if conversion fails or key missing
                                     metrics_df.loc[idx, TXT['value_label']] = results["performance_metrics"].get(translate_text(idx, 'en'), '')
                             results_area.dataframe(metrics_df, use_container_width=True)
                         elif results and results.get("error"):
                             results_area.error(TXT["error_message"].format(error=results['error']))
                             if "details" in results:
                                 results_area.error(TXT["error_details"].format(details=results['details']))
                         else:
                             results_area.warning(TXT["no_metrics_warning"])

                         results_area.subheader(TXT["trade_log_header"])
                         trade_log_df = results.get("trade_log") if results else None
                         if trade_log_df is not None and not trade_log_df.empty:
                             # Translate trade log columns
                             trade_log_df.columns = [translate_text(col, language) for col in trade_log_df.columns]
                             results_area.dataframe(trade_log_df, use_container_width=True)
                         elif not (results and results.get("error")):
                             results_area.info(TXT["no_trade_log_warning"])

                         # --- Log Display --- 
                         stdout_val = backtest_stdout.getvalue()
                         stderr_val = backtest_stderr.getvalue()
                         with log_expander_area:
                             if stdout_val:
                                 with st.expander(TXT["backtest_log_expander_title"]):
                                     st.code(stdout_val)
                             if stderr_val:
                                 st.warning(TXT["backtest_stderr_warning"])
                                 st.code(stderr_val)
                         # -----------------------------------------------------
        else:
             with results_area:
                 st.warning(TXT["enter_ticker_warning"])

# --- Disclaimer ---
st.markdown("---")
st.warning(f"**{TXT['disclaimer_header']}**")
st.warning(TXT["disclaimer_text"]) 