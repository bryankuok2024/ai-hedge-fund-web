import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from colorama import Fore, Back, Style, init
import questionary
from agents.ben_graham import ben_graham_agent
from agents.bill_ackman import bill_ackman_agent
from agents.fundamentals import fundamentals_agent
from agents.portfolio_manager import portfolio_management_agent
from agents.technicals import technical_analyst_agent
from agents.risk_manager import risk_management_agent
from agents.sentiment import sentiment_agent
from agents.warren_buffett import warren_buffett_agent
from graph.state import AgentState
from agents.valuation import valuation_agent
from utils.display import print_trading_output
from utils.analysts import ANALYST_ORDER, get_analyst_nodes
from utils.progress import progress
from llm.models import LLM_ORDER, get_model_info
import io
import contextlib
import json

import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
from tabulate import tabulate
from utils.visualize import save_graph_as_png

# Load environment variables from .env file
load_dotenv()

# --- DEBUGGING --- 
import os
print(f"DEBUG: OPENAI_API_KEY loaded: {'*****' if os.getenv('OPENAI_API_KEY') else 'Not Found'}")
print(f"DEBUG: OPENAI_API_BASE loaded: {os.getenv('OPENAI_API_BASE', 'Not Found')}")
# --- END DEBUGGING ---

init(autoreset=True)


def create_workflow(selected_analysts=None):
    """Create the workflow with selected analysts."""
    workflow = StateGraph(AgentState)
    workflow.add_node("start_node", start)

    # Get analyst nodes from the configuration
    analyst_nodes = get_analyst_nodes()

    # Default to all analysts if none selected or empty list passed
    if not selected_analysts:
        selected_analysts = list(analyst_nodes.keys())

    # Add selected analyst nodes
    for analyst_key in selected_analysts:
        # Handle potential KeyError if an invalid analyst key is passed
        if analyst_key in analyst_nodes:
            node_name, node_func = analyst_nodes[analyst_key]
            workflow.add_node(node_name, node_func)
            workflow.add_edge("start_node", node_name)
        else:
            print(f"Warning: Analyst key '{analyst_key}' not found in configuration. Skipping.")

    # Always add risk and portfolio management
    workflow.add_node("risk_management_agent", risk_management_agent)
    workflow.add_node("portfolio_management_agent", portfolio_management_agent)

    # Connect selected analysts to risk management
    for analyst_key in selected_analysts:
        if analyst_key in analyst_nodes: # Check again to avoid errors if skipped above
            node_name = analyst_nodes[analyst_key][0]
            workflow.add_edge(node_name, "risk_management_agent")

    workflow.add_edge("risk_management_agent", "portfolio_management_agent")
    workflow.add_edge("portfolio_management_agent", END)

    workflow.set_entry_point("start_node")
    return workflow


def parse_hedge_fund_response(response):
    """Parses a JSON string and returns a dictionary."""
    try:
        # If the response is already a dictionary (might happen if called directly), return it
        if isinstance(response, dict):
            return response
        # If it's a HumanMessage content (string), parse it
        elif isinstance(response, str):
            # Handle potential empty string or non-JSON string gracefully
            if not response.strip() or not response.strip().startswith('{'):
                print(f"Invalid response format for JSON parsing: {repr(response)}")
                return {"error": "Invalid response format from agent", "details": repr(response)}
            return json.loads(response)
        else:
            # Log unexpected type
            print(f"Unexpected response type for parsing: {type(response).__name__}")
            return {"error": "Unexpected response type", "details": f"Got {type(response).__name__}"}

    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}\nResponse: {repr(response)}")
        # Return a structured error instead of None
        return {"error": "JSON decoding failed", "details": str(e), "response": repr(response)}
    except TypeError as e:
        print(f"Invalid response type (expected string or dict, got {type(response).__name__}): {e}")
        return {"error": "Type error during parsing", "details": str(e)}
    except Exception as e:
        print(f"Unexpected error while parsing response: {e}\nResponse: {repr(response)}")
        return {"error": "Unexpected parsing error", "details": str(e), "response": repr(response)}


##### Run the Hedge Fund (Refactored Core Logic) #####
def run_hedge_fund_core(
    tickers: list[str],
    start_date: str,
    end_date: str,
    portfolio: dict,
    show_reasoning: bool = False,
    selected_analysts: list[str] = None, # Changed default to None
    model_name: str = "gpt-4o",
    model_provider: str = "OpenAI",
):
    """Core logic to run the hedge fund simulation. Callable directly."""
    # Start progress tracking (if needed, consider making it optional for non-CLI)
    # progress.start() # Commented out for now, might add too much noise in webapp

    try:
        # Create and compile the workflow based on selected analysts
        # Use default analysts if None or empty list is provided
        current_selected_analysts = selected_analysts if selected_analysts else list(get_analyst_nodes().keys())
        workflow = create_workflow(current_selected_analysts)
        agent = workflow.compile()

        # Prepare initial state
        initial_state = {
            "messages": [
                HumanMessage(
                    content="Make trading decisions based on the provided data.",
                )
            ],
            "data": {
                "tickers": tickers,
                "portfolio": portfolio,
                "start_date": start_date,
                "end_date": end_date,
                "analyst_signals": {}, # Initialize as empty dict
            },
            "metadata": {
                "show_reasoning": show_reasoning,
                "model_name": model_name,
                "model_provider": model_provider,
            },
        }

        # Invoke the agent graph
        final_state = agent.invoke(initial_state)

        # Ensure final_state and messages exist before accessing
        if not final_state or "messages" not in final_state or not final_state["messages"]:
            print("Error: Agent invocation did not return expected final state or messages.")
            return {"error": "Agent invocation failed", "details": "Missing final state or messages"}

        # Parse the last message content
        last_message_content = final_state["messages"][-1].content
        decisions = parse_hedge_fund_response(last_message_content)

        # Ensure analyst_signals exists in the final state data
        analyst_signals = final_state.get("data", {}).get("analyst_signals", {})

        return {
            "decisions": decisions,
            "analyst_signals": analyst_signals,
        }
    except Exception as e:
        print(f"Error during hedge fund execution: {e}")
        # Consider more specific error handling/logging
        return {"error": "Hedge fund execution failed", "details": str(e)}
    # finally:
        # Stop progress tracking
        # progress.stop() # Commented out

# Keep the old run_hedge_fund function signature for compatibility with backtester.py
# It now acts as a simple wrapper around the core logic.
def run_hedge_fund(
    tickers: list[str],
    start_date: str,
    end_date: str,
    portfolio: dict,
    show_reasoning: bool = False,
    selected_analysts: list[str] = [], # Kept default as empty list for backward compatibility
    model_name: str = "gpt-4o",
    model_provider: str = "OpenAI",
):
    """Directly call the core function."""
    return run_hedge_fund_core(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        portfolio=portfolio,
        show_reasoning=show_reasoning,
        selected_analysts=selected_analysts if selected_analysts else None, # Pass None if empty list
        model_name=model_name,
        model_provider=model_provider,
    )


def start(state: AgentState):
    """Initialize the workflow with the input message."""
    # Potentially add logging here if needed
    # print(f"Starting workflow with state: {state}")
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the hedge fund trading system")
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=100000.0,
        help="Initial cash position. Defaults to 100000.0)"
    )
    parser.add_argument(
        "--margin-requirement",
        type=float,
        default=0.0,
        help="Initial margin requirement. Defaults to 0.0"
    )
    parser.add_argument("--tickers", type=str, required=True, help="Comma-separated list of stock ticker symbols")
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD). Defaults to 3 months before end date",
    )
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD). Defaults to today")
    parser.add_argument("--show-reasoning", action="store_true", help="Show reasoning from each agent")
    parser.add_argument(
        "--show-agent-graph", action="store_true", help="Show the agent graph"
    )
    parser.add_argument(
        "--analysts", type=str, help="Comma-separated list of analyst keys (skips interactive selection if provided)"
    )
    parser.add_argument(
        "--model", type=str, help="LLM model name (skips interactive selection if provided)"
    )

    args = parser.parse_args()

    # Parse tickers from comma-separated string
    tickers = [ticker.strip() for ticker in args.tickers.split(",")]

    selected_analysts = None
    if args.analysts:
        selected_analysts = [a.strip() for a in args.analysts.split(',')]
        print(f"Using specified analysts: {', '.join(Fore.GREEN + a.title().replace('_', ' ') + Style.RESET_ALL for a in selected_analysts)}\n")
    else:
        # Interactive analyst selection only if --analysts not provided
        choices = questionary.checkbox(
            "Select your AI analysts.",
            choices=[questionary.Choice(display, value=value) for display, value in ANALYST_ORDER],
            instruction="\n\nInstructions: \n1. Press Space to select/unselect analysts.\n2. Press 'a' to select/unselect all.\n3. Press Enter when done to run the hedge fund.\n",
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

    # Select LLM model
    model_choice = None
    model_provider = "Unknown" # Default provider
    if args.model:
        model_choice = args.model
        model_info = get_model_info(model_choice)
        if model_info:
            model_provider = model_info.provider.value
            print(f"\nUsing specified {Fore.CYAN}{model_provider}{Style.RESET_ALL} model: {Fore.GREEN + Style.BRIGHT}{model_choice}{Style.RESET_ALL}\n")
        else:
            print(f"\nUsing specified model (provider unknown): {Fore.GREEN + Style.BRIGHT}{model_choice}{Style.RESET_ALL}\n")
    else:
        # Interactive model selection only if --model not provided
        model_choice_q = questionary.select(
            "Select your LLM model:",
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
                # Keep provider as Unknown if info not found
                print(f"\nSelected model: {Fore.GREEN + Style.BRIGHT}{model_choice}{Style.RESET_ALL}\n")

    # Show agent graph if requested (needs compiled app)
    if args.show_agent_graph:
        temp_workflow = create_workflow(selected_analysts)
        temp_app = temp_workflow.compile()
        file_path = "agent_graph.png" # Simplified naming
        # if selected_analysts:
        #     file_path = "_".join(selected_analysts) + "_graph.png"
        save_graph_as_png(temp_app, file_path)
        print(f"Agent graph saved to {file_path}")

    # Validate and set dates
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    start_date = args.start_date
    if not start_date:
        try:
            end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
            start_date_dt = end_date_dt - relativedelta(months=3)
            start_date = start_date_dt.strftime("%Y-%m-%d")
        except ValueError:
            print(f"Error parsing end date '{end_date}'. Please use YYYY-MM-DD format.")
            sys.exit(1)
    else:
        # Validate provided start_date format
        try:
            datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            print(f"Error parsing start date '{start_date}'. Please use YYYY-MM-DD format.")
            sys.exit(1)

    # Validate end_date format if provided via args
    if args.end_date:
        try:
            datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            print(f"Error parsing end date '{end_date}'. Please use YYYY-MM-DD format.")
            sys.exit(1)

    # Set up initial portfolio state
    initial_portfolio = {
        "cash": args.initial_cash,
        "margin_used": 0.0,
        "margin_requirement": args.margin_requirement,
        "positions": {
            ticker: {
                "long": 0, "short": 0, "long_cost_basis": 0.0, "short_cost_basis": 0.0, "short_margin_used": 0.0
            } for ticker in tickers
        },
        "realized_gains": {
            ticker: {"long": 0.0, "short": 0.0} for ticker in tickers
        }
    }

    print(f"Running AI Hedge Fund for {', '.join(tickers)}...")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Cash: ${initial_portfolio['cash']:,.2f}")
    if initial_portfolio['margin_requirement'] > 0:
        print(f"Margin Requirement: {initial_portfolio['margin_requirement']:.1%}")
    print("-" * 30)

    # Call the core function directly
    results = run_hedge_fund_core(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        portfolio=initial_portfolio,
        show_reasoning=args.show_reasoning,
        selected_analysts=selected_analysts,
        model_name=model_choice, # Use the selected/provided model name
        model_provider=model_provider, # Use the determined provider
    )

    # Print the final results
    print("-" * 30)
    print("Processing Complete.")

    if results and "error" in results:
        print(f"{Fore.RED}An error occurred during execution:")
        print(f"{Fore.RED}Error: {results['error']}")
        if "details" in results:
            print(f"{Fore.RED}Details: {results['details']}")
        if "response" in results and results["response"]:
            print(f"{Fore.RED}Agent Response causing error: {results['response']}")

    elif results and "decisions" in results:
        print_trading_output(results["decisions"], results["analyst_signals"])
    else:
        print(f"{Fore.YELLOW}No decisions or errors returned from the agent.")

    print("-" * 30)
