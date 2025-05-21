import os
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.tools.yahoo_finance import YahooFinanceToolSpec
from llama_index.tools.tavily_research import TavilyToolSpec
from llama_index.core.workflow import Context
from llama_index.llms.groq import Groq
import yfinance as yf

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", key="chatbot_api_key", type="password")
    "[Get a Groq API key](https://console.groq.com/keys?_gl=1*12oc5uy*_ga*MTA0MDc4MTIzLjE3NDc3MzQyMDE.*_ga_4TD0X2GEZG*czE3NDc4MTYyMzYkbzMkZzEkdDE3NDc4MTY4NjEkajAkbDAkaDA.)"
    "[View the source code](https://github.com/adicangit/Financial-Investment-Research-CoPilot/blob/main/app.py)"

# Initialize LLM
llm = Groq(model="llama-3.3-70b-versatile", api_key = groq_api_key)

# Define tools
def returns(ticker: str, start_date: str, end_date: str) -> float:
    """
    Return the percentage return of a stock between start_date and end_date.
    Args:
        ticker (str): The stock ticker (e.g., 'GOOGL')
        start_date (str): The starting date in YYYY-MM-DD format
        end_date (str): The ending date in YYYY-MM-DD format
    Returns:
        float: The percentage return
    """
    try:
        stock = yf.Ticker(ticker)
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        data = stock.history(start=start_date, end=end_date, auto_adjust=True)
        if data.empty:
            raise ValueError(f"No data available for {ticker} in the given date range.")
        if len(data) == 1:
            return round(((data['Open'].iloc[0] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100, 2)
        return round(((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100, 2)
    except Exception as e:
        return f"Error calculating returns for {ticker}: {str(e)}"

def price_history(ticker: str, start_date: str, end_date: str):
    """
    Return the price history of a stock between start_date and end_date.
    Args:
        ticker (str): The stock ticker (e.g., 'GOOGL')
        start_date (str): The starting date in YYYY-MM-DD format
        end_date (str): The ending date in YYYY-MM-DD format
    Returns:
        str: JSON string of the price history
    """
    try:
        stock = yf.Ticker(ticker)
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        data = stock.history(start=start_date, end=end_date, auto_adjust=True)
        if data.empty:
            raise ValueError(f"No data available for {ticker} in the given date range.")
        return data.to_json()
    except Exception as e:
        return f"Error fetching price history for {ticker}: {str(e)}"

def CAGR(ticker: str, start_date: str, end_date: str) -> float:
    """
    Calculate CAGR based on historical price data.
    Automatically calculates period in years from the date range.
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        years = (end_dt - start_dt).days / 365.25

        data = get_stock_data(ticker, start_date, end_date)
        if data.empty:
            return f"No data available for {ticker} in the given range."

        initial_price = data['Close'].iloc[0]
        final_price = data['Close'].iloc[-1]
        cagr = ((final_price / initial_price) ** (1 / years)) - 1

        return round(cagr * 100, 2)
    except Exception as e:
        if "HTTPError" in str(e) or "429" in str(e):
            return "Rate limit exceeded. Please wait and try again shortly."
        return f"Error calculating CAGR for {ticker}: {str(e)}"

def get_date() -> str:
    """
    Return today's date.
    Returns:
        str: Today's date in YYYY-MM-DD format
    """
    return datetime.today().strftime('%Y-%m-%d')

# Create FunctionTool instances
price_history_tool = FunctionTool.from_defaults(price_history)
returns_tool = FunctionTool.from_defaults(returns)
CAGR_tool = FunctionTool.from_defaults(CAGR)
today_date_tool = FunctionTool.from_defaults(get_date)

# Initialize Yahoo Finance and Tavily tools
try:
    finance_tools = YahooFinanceToolSpec().to_tool_list()[:-1]
    tavily_tool = TavilyToolSpec(api_key=TAVILY_API_KEY).to_tool_list()
    search_tool = tavily_tool[0]
except Exception as e:
    st.error(f"Failed to initialize tools: {str(e)}")
    st.stop()

# Initialize ReActAgent
agent = ReActAgent.from_tools(
    tools = [price_history_tool, today_date_tool, returns_tool, search_tool, CAGR_tool] + finance_tools,
    llm=llm,
    verbose=True,
    system_prompt="You're a helpful assistant.",
)

# Streamlit UI
with st.container():
    st.header("📈 Investment Research CoPilot")

    st.markdown("""
    <div style="background-color:#495057; padding:10px; border-radius:8px; margin-bottom:10px;">
        💬 What are the analysts recommendations for Tesla?
    </div>
    <div style="background-color:#495057; padding:10px; border-radius:8px;">
        💬 What did Uber's management team say about margins?
    </div>
    """, unsafe_allow_html=True)




# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help with your financial queries?"}
    ]
if "llm_messages" not in st.session_state:
    st.session_state.llm_messages = [
        ChatMessage(role="system", content="You are a useful financial assistant."),
        ChatMessage(role="assistant", content="How can I help with your financial queries?")
    ]

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    if not groq_api_key:
        st.info("Please add your Groq API key to continue.")
        st.stop()
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.llm_messages.append(ChatMessage(role="user", content=prompt))

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(f"**{message['content']}**" if message["role"] == "user" else message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            response_text = agent.chat(prompt, chat_history=st.session_state.llm_messages)

            # st.write(response_text.response)
            # st.markdown(f"```\n{response_text.response}\n```")
            # st.code(response_text.response, language="text")
            st.text(response_text.response)


            st.session_state.messages.append({"role": "assistant", "content": response_text.response})
            st.session_state.llm_messages.append(ChatMessage(role="assistant", content=response_text.response))
