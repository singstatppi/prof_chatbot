__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import ast
import altair as alt
from dotenv import load_dotenv
import json
from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
import logging
import os
import pandas as pd
import requests
import streamlit as st
import sys
import tiktoken
from typing import List
import uuid
from urllib.request import Request,urlopen
import time

# Load environment variables
#load_dotenv("credentials.env")
#OPENAI_API_KEY = os.getenv("OPENAI_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Set Constants
CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-large"

CHUNK_COLLECTION_NAME = "chunk_vector_store"
LONG_CONTEXT_COLLECTION_NAME = "longcontext_vector_store"

SINGAPORE_PRICE_INDEX_MAP = {
    1: "Accounting Services Price Index",
    2: "Accounting And Auditing Price Index",
    3: "Book-Keeping Price Index",
    4: "Cargo Handling Price Index",
    5: "Container Depot Services Price Index",
    6: "Crane Services Price Index",
    7: "Stevedoring Services Price Index",
    8: "Computer Consultancy And Information Services Price Index",
    9: "Computer Programming And Consultancy Price Index",
    10: "Information Services And Online Marketplace Price Index",
    11: "Freight Forwarding Price Index",
    12: "Sea Freight Forwarding Price Index",
    13: "Air Freight Forwarding Price Index",
    14: "Land Freight Forwarding Price Index",
    15: "Sea Freight Transport Price Index",
    16: "Containerised Freight Transport Price Index",
    17: "Dry Bulk Freight Transport Price Index",
    18: "Liquid Bulk And Gas Freight Transport Price Index",
    19: "Telecommunications Services Price Index",
    20: "Wired And Wireless Telecommunications Services Price Index",
    21: "Internet Access Providers And Other Telecommunications Services Price Index",
    22: "Warehousing And Storage Price Index",
    23: "General And Refrigerated Warehousing Price Index",
    24: "Dangerous Goods Storage Price Index"
}

FINLAND_PRODUCT_MAP = {
    "502": 15,
    "521": 22,
    "5224": 4,
    "61": 8,
    "62": 9,
    "63": 10,
    "692": 1
}

# PROMPT COLLECTION
CONTEXTUALISE_QUERY = """
Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. For info, PPI = Producer Price Index, SPPI = Services Producer Price Index, Legal SPPI = Legal Services PPI, Postal SPPI = Postal Services PPI and so on for other industries. Look for plotting/analysis verbs not limited to "plot," "analyze," "compare," "show," "graph" and queries with them should keep the actions stated. Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
"""

QUERY_CATEGORISER_PROMPT = """
Your task is to analyze user queries, categorize them as either "Plotting/Data Analytics" or "Other questions," and process them accordingly to provide helpful responses based on multiple document sources.

Analyse this query:
<<<<<<<<<<query>>>>>>>>>>

For Plotting/Data Analytics Queries:

First, explicitly state that you've identified this as a Plotting/Data Analytics query
Extract and structure the following data: "<query></query><action></action><country><country><frequency></frequency>"
Provide a tailored response for data visualization or analysis

For Other Queries:

Extract relevant keywords following the guidelines below
Format as: <query>User's question</query> followed by <keywords>extracted keywords</keywords>
Provide a comprehensive answer based on the retrieved document chunks

IMPORTANT:
Be precise in categorizing queries - look for plotting/analysis verbs like "plot," "analyze," "compare," "show," "graph"
Ensure exact extraction of index names - they must match available data sources
For keyword extraction, prioritize specific, searchable terms over general language
Do not make up data or index information if not provided in the context
Always maintain the specified output format for each query type
If the user did not specify the country, assume that the country relevant to the query is Singapore
Only respond with information required without any additional explanations
Always assume that the user is asking about annual data, then quarterly

Plotting/Data Analytics Examples:

Q:"Plot China's Book Keeping index in 2020-2024"
A:<query>Plot China's Book Keeping index in 2020-2024</query><action>Plot</action><country>China<country><frequency>Annual</frequency>
Q:"Plot Singapore and China's Book Keeping index in 2020-2024"
A:<query>Plot Singapore and China's Book Keeping index in 2020-2024</query><action>Plot</action><country>Singapore,China<country><frequency>Annual</frequency>
Q:"Plot all of Singapore and China's accounting index in 2020-2024"
A:<query>Plot all of Singapore and China's accounting index in 2020-2024</query><action>Plot</action><country>Singapore,China<country><frequency>Annual</frequency>
Q:"Analyse all of Singapore and China's accounting index in 2020-2024"
A:<query>Analyse all of Singapore and China's accounting index in 2020-2024</query><action>Analyse</action><country>Singapore,China<country><frequency>Annual</frequency>
Q:"Compare all of Singapore and China's accounting index in 2020-2024"
A:<query>Compare all of Singapore and China's accounting index in 2020-2024</query><action>Analyse</action><country>Singapore,China<country><frequency>Annual</frequency>
Q:"Plot all accounting index in 2020-2024"
A:<query>Plot all accounting index in 2020-2024</query><action>Plot</action><country>Singapore<country><frequency>Annual</frequency>
Q:"Plot all quarter's accounting index in 2020-2024"
A:<query>Plot all quarter's accounting index in 2020-2024</query><action>Plot</action><country>Singapore<country><frequency>Quarterly</frequency>

Other Queries Examples:

Q:"What is the impact of inflation on consumer spending patterns?"
A:<query>Impact of inflation on consumer spending patterns</query><keywords>inflation, consumer spending patterns</keywords>
Q:"How does the Federal Reserve's interest rate policy affect stock market volatility?"
A:<query>Federal Reserve's interest rate policy and stock market volatility</query><keywords>Federal Reserve, interest rate policy, stock market volatility</keywords>
Q:"Explain the relationship between oil prices and transportation costs in the US from 2010 to 2020."
A:<query>Relationship between oil prices and transportation costs in the US from 2010 to 2020</query><keywords>oil prices, transportation costs, US, 2010, 2020</keywords>
Q:"Where was the voorburg meeting held in 2020?"
A:<query>Voorburg meeting location in 2020</query><keywords>voorburg meeting, 2020, location</keywords>

"""

RELATED_ID_PROMPT = """
You will be given a list containing a mapping table. Your task is to analyze the query and provide a appropriate response based on the data in the mapping table.

First, here is the mapping table:
(
    1: "Accounting Services Price Index",
    2: "Accounting And Auditing Price Index",
    3: "Book-Keeping Price Index",
    4: "Cargo Handling Price Index",
    5: "Container Depot Services Price Index",
    6: "Crane Services Price Index",
    7: "Stevedoring Services Price Index",
    8: "Computer Consultancy And Information Services Price Index",
    9: "Computer Programming And Consultancy Price Index",
    10: "Information Services And Online Marketplace Price Index",
    11: "Freight Forwarding Price Index",
    12: "Sea Freight Forwarding Price Index",
    13: "Air Freight Forwarding Price Index",
    14: "Land Freight Forwarding Price Index",
    15: "Sea Freight Transport Price Index",
    16: "Containerised Freight Transport Price Index",
    17: "Dry Bulk Freight Transport Price Index",
    18: "Liquid Bulk And Gas Freight Transport Price Index",
    19: "Telecommunications Services Price Index",
    20: "Wired And Wireless Telecommunications Services Price Index",
    21: "Internet Access Providers And Other Telecommunications Services Price Index",
    22: "Warehousing And Storage Price Index",
    23: "General And Refrigerated Warehousing Price Index",
    24: "Dangerous Goods Storage Price Index"
)

Here is the query:
<query>
<<<<<<<<<<query>>>>>>>>>>
</query>

To process this task, follow these steps:

1. Carefully read the data provided.
2. Analyze the query to determine what data it is looking for.
3. Identify the relevant data points in the matching table that match the query.
4. For Singapore data, understand the index structure, and look for the best fit based on the query.
For example, if user asks for all of a certain index, you should return all indices available for that main index. However, if user specified only a certain index, you should return only that index.
Index Structure for Singapore:
    Accounting Services Price Index:
    Main Index: Accounting Services Price Index
    Sub Indexes: ‘Accounting & Auditing’, ‘Book-keeping’

    Cargo Handling Price Index:
    Main Index: Cargo Handling Price Index
    Sub Indexes: ‘Container Depot Services’, ‘Crane Services’, ‘Stevedoring Services’

    Computer Consultancy & Information Services Price Index:
    Main Index: Computer Consultancy & Information Services Price Index
    Sub Indexes: ‘Computer Programming & Consultancy’,’Information Services & Online Marketplace’

    Freight Forwarding Price Index:
    Main Index: Freight Forwarding Price Index
    Sub Indexes: ‘Sea Freight Forwarding’,’Air Freight Forwarding’,’Land Freight Forwarding’

    Sea Freight Transport Price Index:
    Main Index: Sea Freight Transport Price Index
    Sub Indexes: ‘Containerised Freight Transport’,’Dry Bulk Freight Transport’,’Liquid Bulk & Gas Freight Transport’

    Telecommunications Services Price Index:
    Main Index: Telecommunications Services Price Index
    Sub Indexes: ‘Wired & Wireless Telecommunications Services’,’Internet Access Providers & Other Telecommunications Services’

    Warehousing & Storage Price Index:
    Main Index: Warehousing & Storage Price Index
    Sub Indexes: ‘General & Refrigerated Warehousing’,’Dangerous Goods Storage’

Your answer here, respond only with the relavant ID(s) in this format: 
Example 1
<id>id1</id>
Example 2
<id>id1,id2</id>
Example 3
<id>id1</id>
Example 4
<id>id1,id2</id>

Remember, do not include any explanations or additional text outside of the <response> tags. Your response should be either the plot data or a direct answer, nothing more."""

PLOT_ANALYTICS_PROMPT = """
You will be given a JSON Data. Your task is to analyze the query and provide a appropriate response based on the data.

First, here is the JSON Data:
<<<<<<<<<<json_data>>>>>>>>>>

Now, I will provide you with a query. You should analyze this query and respond in one of two ways:

1. If the query asks to plot or display data, you should return only the relevant data from the JSON, prefixed with "Plot ".
2. If the query asks a specific question about the data, you should provide a direct answer based on the information in the JSON string.

Here is the query:
<query>
<<<<<<<<<<query>>>>>>>>>>
</query>

To process this task, follow these steps:

1. Carefully read the data provided.
2. Analyze the query 
3. If the query asks to plot or display data, you should return the data from the JSON, prefixed with "Plot ".
4. If the query asks a specific question about the data, you should provide a direct answer based on the information in the JSON string.

Provide your response in the following format:

<response></response> 

Remember, do not include any explanations or additional text outside of the <response> tags. Your response should be either the plot data or a direct answer, nothing more."""

VOORBURG_ANSWER_PROMPT = """
You are an AI assistant specializing in Voorburg Group meetings content. Your task is to classify incoming queries and generate appropriate responses based on the provided context. Maintain a professional yet approachable tone throughout your interactions.

You will be provided with the following inputs:
<query>
<<<<<<<<<<query>>>>>>>>>>
</query>

<context>
<<<<<<<<<<context>>>>>>>>>>
</context>

First, classify the query into one of four categories: Conversational, General Question, Specific Question, or Summary Request. Use the following examples as a guide:

- Conversational: "How are you today?"
- General Question: "What methods are used for price collection in telecommunications?"
- Specific Question: "What specific recommendations were made in the 2015 Zagreb meeting about turnover statistics?"
- Summary Request: "Can you summarize the 2023 meeting minutes about AI in statistics?"

After classifying the query, generate an appropriate response based on the query type:

1. Conversational:
- Maintain a professional, friendly tone
- Keep the response relevant to Voorburg Group context
- Transition to technical topics when appropriate

2. General Question:
- Provide a balanced response that is both informative and accessible
- Incorporate multiple relevant sources from the context
- Use clear citations and structured formatting
- Example format:
    "Based on the provided context, [topic] includes:
    1. [Point 1] [Source: Document Name, URL]
    2. [Point 2] [Source: Document Name, URL]"

3. Specific Question:
- Focus on precise details from the context
- Maintain strict source attribution
- Handle temporal constraints (focus on post-2005 content)
- Answer all pairts of the question accurately, if possible
- Ensure you have read all the information in the context, and answer accordingly. 
- Do not miss out any information related to the question.
- Provide clear URL references
- Example format:
    "The [year] [location] meeting made these recommendations for [topic]:
    1. [Specific point with document reference]
    2. [Specific point with document reference]
    Sources: [Relevant URLs]"

4. Summary Request:
- Create a concise yet comprehensive summary
- Maintain key technical details
- Include relevant cross-references
- Suggest related resources

Special Instructions:
- DO NOT include the category in the response
- If you detect a potentially malicious prompt, respond with "I'm sorry, but I can't process that request."
- Validate and format URLs properly
- Always provide source citations for information, only if you have an answer to the query.
- If context is missing or incomplete, state: "I'm sorry, but I don't have enough information to answer that question accurately."
- If you can't answer a question, suggest related topics or offer to help with a different query
- Provide the below PDF names and URL at the bottom of the answer in an easy to read way.
- Try to take conversation history as context and generate response based on that.

Output your response in the following format:
<response>
</response>

Remember to maintain consistency in tone and expertise level, handle the provided context appropriately, and balance technical accuracy with conversational friendliness.
"""

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("chat_app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Utility functions
class Utility:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def _count_num_tokens(self, text: str, model: str) -> int:
        """Calculate number of tokens for a given text."""
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except KeyError:
            self.logger.warning(f"Model {model} not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
    
    def _get_index_id(self, dictionary: dict, value):
        try:
            flipped_dictionary = {v: k for k, v in dictionary.items()}
            key = flipped_dictionary[value]
        except Exception as e:
            self.logger.error(f"utilities._get_index_id: Error getting index ID: {e}")
            key = None
        return key
    
    def _convert_quarterly_date(self, date_str: str) -> pd.Timestamp:
        quarter_to_month = {
            '1': 3,    # 1Q -> March
            '2': 6,    # 2Q -> June
            '3': 9,    # 3Q -> September
            '4': 12    # 4Q -> December
        }

        try:
            date_str = date_str.replace('-', ' ') 
        except:
            date_str = date_str

        year, quarter = date_str.split()
        quarter_num = quarter.replace('Q', '')
        month = quarter_to_month[quarter_num]

        return pd.to_datetime(f"{year}-{month:02d}-01")
    
    def _extract_value_from_tag(self, tag_name: str, response: str):
        try:
            value = response.split(f"<{tag_name}>")[1].split(f"</{tag_name}>")[0].strip()
        except Exception as e:
            self.logger.error(f"utilities._extract_value_from_tag: Error extracting value from tag: {e}")
            value = ""
        return value
    
    def _urllib_request(self,url,header,method,data=None):
        try:
            request = Request(url, headers=header, method=method, data=data)
            response = urlopen(request).read()
            json_response = json.loads(response.decode('utf-8'))
            return json_response
        except Exception as e:
            self.logger.error(f"utilities._urllib_request: Error making request: {e}")
            return None
        
    def _requests_request(self,url,header,method,data=None):
        try:
            response = requests.request(method=method, url=url, headers=header, json=data)
            json_response = response.json()
            return json_response
        except Exception as e:
            self.logger.error(f"utilities._requests_request: Error making request: {e}")
            return None

# Chat Class
class ChatAssistant:
    """Main chat assistant class handling conversation and data processing."""
    
    def __init__(self):
        """Initialize the chat assistant with required components."""
        self.utility_functions = Utility()
        self.logger = logging.getLogger(__name__)
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self.llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.1)
        self.tokens_used = 0
        
        # Initialize Chroma collections
        self.chunk_collection = Chroma(collection_name=CHUNK_COLLECTION_NAME,embedding_function=self.embeddings, persist_directory='./'+CHUNK_COLLECTION_NAME)
        self.long_context_collection = Chroma(collection_name=LONG_CONTEXT_COLLECTION_NAME,embedding_function=self.embeddings, persist_directory='./'+LONG_CONTEXT_COLLECTION_NAME)

    def contextualise_query(self,chat_history,query):
        """Contextualise user queries based on chat history."""
        try:
            # Contextualise query logic here
            chat_history = "\n <chat_history> \n" + chat_history + "\n </chat_history> \n"
            query = "\n <query> \n" + query + "\n </query> \n"
            prompt = CONTEXTUALISE_QUERY + chat_history + query
            response = self.llm.invoke(prompt).content
            return response
        except Exception as e:
            self.logger.error(f"Error contextualising query: {e}")
            return "An error occurred while processing your request."

    def _process_singapore_data(self, raw_data: dict) -> pd.DataFrame:
        """Process Singapore data into structured DataFrame."""
        # Collect DataFrames for each entry
        dataframes = []
        
        # Parse JSON data
        for entry in raw_data:
            row_text = entry["rowText"]
            row_id = self.utility_functions._get_index_id(SINGAPORE_PRICE_INDEX_MAP, row_text)
            columns = entry["columns"]
            dates = [col['key'] for col in columns]
            values = [float(col['value']) for col in columns]

            temp_df = pd.DataFrame({"Date": dates, row_id: values})
            dataframes.append(temp_df) # Append to temp list
        
        # Merge all DataFrames
        merged_df = dataframes[0]
        for df in dataframes[1:]:
            merged_df = pd.merge(merged_df, df, on="Date", how="outer")
        
        # Convert dates
        merged_df['Date'] = merged_df['Date'].apply(
            self.utility_functions._convert_quarterly_date if 'Q' in str(merged_df['Date'].iloc[0]) 
            else lambda x: pd.to_datetime(x, format='%Y')
        )

        merged_df = merged_df.sort_values("Date").reset_index(drop=True)

        df_melt = merged_df.melt(id_vars=['Date'], var_name='Index ID', value_name='Index Value')
        df_melt['Country'] = "Singapore"
        df_melt.dropna(inplace=True)
        df_melt = df_melt[['Date','Country','Index ID','Index Value']]
        df_melt['Date'] = pd.to_datetime(df_melt['Date'], format='%Y-%m-%d')
        
        return df_melt

    def _process_finland_data(self, raw_data: dict, frequency: str) -> pd.DataFrame:
        """Process Finland statistics data into structured DataFrame."""
        date_key = "Vuosineljännes" if frequency == "quarterly" else "Vuosi"
        dates = raw_data["dimension"][date_key]["category"]
        products = raw_data["dimension"]["Tuotteet toimialoittain (CPA 2015)"]["category"]
        values = raw_data["value"]

        records = []
        value_index = 0
        
        for date_code, date_idx in dates["index"].items():
            date_str = self.utility_functions._convert_quarterly_date(date_code) if frequency == "quarterly" else pd.to_datetime(f"{date_code}-01-01")
            
            for product_code, product_idx in products["index"].items():
                value = values[value_index] if value_index < len(values) else None
                records.append({
                    "Date": date_str,
                    "Country": "Finland",
                    "Index ID": FINLAND_PRODUCT_MAP.get(product_code, "Unknown"),
                    "Index Value": value
                })
                value_index += 1
                
        return pd.DataFrame(records)

    def _generate_response(self, prompt: str) -> str:
        """Generate response using LLM with error handling."""
        try:
            response = self.llm.invoke(prompt).content
            self.tokens_used += self.utility_functions._count_num_tokens(prompt + response, CHAT_MODEL)
            return response
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return "Sorry, I encountered an error processing your request."

    def handle_query(self, query: str, chat_history):
        """Main method to handle user queries.""" 
        try:
            # Query Handling logic here

            # Step 1: Query category classification
            # Handle chat_history first (Count if more than 1. If 0, set empty string)
            context_query = self.contextualise_query(chat_history,query)
            print("context_query")
            print(context_query)
            query_category, response = self.query_classification(chat_history,context_query)
            
            if query_category == 1:
                print("Plotting/Data Analytics Query")
                
                # Extract details from response
                user_query = self.utility_functions._extract_value_from_tag("query", response)
                if user_query == "":
                    user_query = query

                query_country = self.utility_functions._extract_value_from_tag("country", response)

                query_frequency = self.utility_functions._extract_value_from_tag("frequency", response)
                if query_frequency == "":
                    query_frequency = "Annual"

                # Put query's specified countries into a list
                try:
                    query_countries = query_country.split(",")
                except:
                    query_countries = [query_country]

                # Retrieve and process data for those countries
                countries_data, countries_data_link = self.retrieve_countries_data(query_countries, query_frequency)

                # Put them into a df for further querying
                try:
                    if len(countries_data) > 1:
                        df = pd.concat(countries_data)
                    else:
                        df = countries_data[0]
                except Exception as e:
                    df = pd.DataFrame(columns=["Date","Country","Index ID","Index Value"])
                    self.logger.error(f"Nothing in countries_data: {e}")

                # With Mapping Table and query, get the most related Index IDs of indexes
                id_list = self.retrieve_related_ids_from_query(user_query)

                # Filter the data based on the ID list and join them into a single df
                filtered_df = []
                for id in id_list:
                    id_df = df[df['Index ID'] == int(id)]
                    filtered_df.append(id_df)
                joint_df = pd.concat(filtered_df)

                # use joint_df in json format as context to answer user's query
                json_joint_df = joint_df.to_json(orient='records')

                # Generate response for Plotting/Data Analytics queries
                response_text = self.get_plot_analyse_response(str(json_joint_df), user_query)
                
                response = [response_text, countries_data_link] # Respond with final data and links to data source tables
            elif query_category == 2:
                print("Chat History Query")
                response = self.utility_functions._extract_value_from_tag("response", response)
            else:
                print("Other Query")
                if "<classification>Conversational</classification>" in response:
                    print("Skipping retrieval") # Conversational does not require any context
                else:
                    print("Retrieving context")
                    try:
                        rewritten_query = self.utility_functions._extract_value_from_tag("query", response)
                        query_keywords = self.utility_functions._extract_value_from_tag("keywords", response)
                    except Exception as e:
                        print(f"Error in parsing response: {e}")
                        logging.error(f"Error in parsing response: {e}")
                    
                    # Step 2: Get relevant material (if needed)

                    # 2.1: Get relevant chunk from chunk_vector_store
                    # Join rewritten query and keywords to make like a 'long question' for the retriever
                    #long_query = rewritten_query + "\nAdditional Context:" + query_keywords # Experimental
                    long_query = rewritten_query

                    # Retrieve content from Embedded DB
                    retrieved_dict = self.retrieve_context(rewritten_query)

                    # Fill up next query with context, but has to keep within context limit of LLM Model
                    filled_prompt = self.fill_prompt_with_context(long_query, chat_history, retrieved_dict, max_template_tokens=128000)
                    
                    # Step 3: Generate response with long context and VOORBURG_ANSWER_PROMPT
                    response_text = self._generate_response(filled_prompt)
                    response = self.utility_functions._extract_value_from_tag("response", response_text)

        except Exception as e:
            self.logger.error(f"Error handling query: {e}")
            response = "An error occurred while processing your request. Please try again or contact technical staff if issue persists."
        
        return response

    # Additional methods for query classification, context retrieval, 
    def query_classification(self, chat_history, query: str):
        """Classify user queries into Plotting/Data Analytics or Other questions."""
        try:
            # Query classification logic here
            category = 0
            query = QUERY_CATEGORISER_PROMPT.replace("<<<<<<<<<<query>>>>>>>>>>", query)
            response = self._generate_response(query)
            
            if "<action>Plot</action>" in response or "<action>Analyse</action>" in response:
                category = 1 # Plotting/Analytics Questions
            elif "<category>2</category>" in response:
                category = 2 # Chat History Questions
            
            return category,response
        except Exception as e:
            self.logger.error(f"Error classifying query: {e}")
            return category, "An error occurred while classifying your query."
    
    def retrieve_related_ids_from_query(self, query: str):
        """Retrieve related IDs from the query."""
        try:
            id_list = []
            # Query classification logic here
            query = RELATED_ID_PROMPT.replace("<<<<<<<<<<query>>>>>>>>>>", query)
            response = self._generate_response(query)
            related_ids = self.utility_functions._extract_value_from_tag("id", response)
            try:
                id_list = related_ids.split(",")
            except:
                id_list = [related_ids]

        except Exception as e:
            self.logger.error(f"Error classifying query: {e}")
        return id_list
    
    def get_plot_analyse_response(self, json_data: str, query: str):
        #{str(select_df)}
        """Generate response for Plotting/Data Analytics queries."""
        try:
            response = ""
            # Query classification logic here
            query = PLOT_ANALYTICS_PROMPT.replace("<<<<<<<<<<json_data>>>>>>>>>>", json_data).replace("<<<<<<<<<<query>>>>>>>>>>", query)
            query_response = self._generate_response(query)
            response = self.utility_functions._extract_value_from_tag("response", query_response)
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            response = "An error occurred while processing your request."

        return response

    def retrieve_countries_data(self, query_countries: List[str], query_frequency: str):
        """Retrieve and process data for specified countries."""
        try:
            # Data retrieval and processing logic here
            # Data retrieval and processing from different countries
            countries_data, countries_data_link = [],[]
            for country in query_countries:
                # Prepare for API request
                if country.strip() == "Singapore":
                    if 'Annual' in query_frequency:
                        key = 'M213521'
                    else:
                        key = 'M213211'
                    request_method = "GET"
                    request_url = "https://tablebuilder.singstat.gov.sg/api/table/tabledata/"+key # API link to table data
                    table_link = "https://tablebuilder.singstat.gov.sg/table/TS/" + key # Link to table site in browser
                    request_header = {'User-Agent': 'Mozilla/5.0', "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,/;q=0.8"}
                    request_data = None
                elif country.strip() == "Finland":
                    if 'Annual' in query_frequency:
                        key = 'f'
                    else:
                        key = 'd'
                    request_method = "POST"
                    request_url = f"https://pxdata.stat.fi:443/PxWeb/api/v1/en/StatFin/pthi/statfin_pthi_pxt_117{key}.px"
                    table_link = f"https://pxdata.stat.fi/PxWeb/pxweb/en/StatFin/StatFin__pthi/statfin_pthi_pxt_117{key}.px/"
                    request_header = {"Content-Type": "application/json"}
                    request_data = {
                        "query": [
                            {
                            "code": "Tuotteet toimialoittain (CPA 2015)",
                            "selection": {
                                "filter": "item",
                                "values": [
                                "502","521","5224","61","62","63","692"
                                ]
                            }
                            },
                            {
                            "code": "Tiedot",
                            "selection": {
                                "filter": "item",
                                "values": [
                                "pisteluku15"
                                ]
                            }
                            }
                        ],
                        "response": {
                            "format": "json-stat2"
                        }
                    }
                
                
                # Process response based on country
                if country.strip() == "Singapore":
                    json_response = self.utility_functions._urllib_request(request_url,request_header,request_method,request_data)
                    country_data = self._process_singapore_data(json_response['Data']['row'])
                elif country.strip() == "Finland":
                    json_response = self.utility_functions._requests_request(request_url,request_header,request_method,request_data)
                    country_data = self._process_finland_data(json_response, query_frequency)

                # Append data to list
                countries_data.append(country_data)
                countries_data_link.append(table_link)
        except Exception as e:
            self.logger.error(f"Error retrieving countries data: {e}")
        return countries_data, countries_data_link

    def retrieve_context(self, query):
        """Retrieve relevant context for user queries."""
        try:
            retrieved_docs = []
            retrieved_dict = []

            #Attemt to rank eurostat methodology and singstat files first
            docs_and_scores = self.chunk_collection.similarity_search_with_score(
                query, k=40  # Fetch more than needed for better reranking
            )
            boosted_docs = []
            for doc, score in docs_and_scores:
                # Lower score = more relevant
                if doc.metadata.get("topic") == "Manuals":
                    score *= 0.8  # Boost priority docs (make them more relevant)
                boosted_docs.append((doc, score))
            # Sort by adjusted scores
            boosted_docs.sort(key=lambda x: x[1])
            # Extract only documents, drop scores
            top_k_docs = [doc for doc, _ in boosted_docs[:20]]
            compressor = CohereRerank(model="rerank-english-v3.0", top_n=10)
            retrieved_chunks = compressor.compress_documents(top_k_docs, query)
            
            # Retrieve documents from related chunks
            for doc in retrieved_chunks:
                # retrieved_document = self.long_context_collection.get(where={"file_link": doc.metadata["file_link"]}) # Matching by file's link
                # # if longdoc not in longcontext_retrieved_docs
                # if retrieved_document in retrieved_docs: 
                #     continue # Skip if already in retrieved_docs
                # retrieved_docs.append(retrieved_document)
                row = {
                    "id": doc.id,
                    "chunk_context": doc.page_content,
                    "long_context": self.long_context_collection.get(where={"file_link": doc.metadata["file_link"]})['documents'][0],
                    "metadata": str(doc.metadata),
                    "file_link": doc.metadata["file_link"]
                }
                retrieved_dict.append(row)
                self.tokens_used += self.utility_functions._count_num_tokens(row["long_context"], EMBEDDING_MODEL)
                
            
            self.tokens_used += self.utility_functions._count_num_tokens(query, EMBEDDING_MODEL)
            
            #populate dictionary

        except Exception as e:
            self.logger.error(f"Error retrieving context: {e}")
        return retrieved_dict
        
    def fill_prompt_with_context(self, query, chat_history, retrieved_docs, max_template_tokens=128000):
        context = ""
        filled_prompt = VOORBURG_ANSWER_PROMPT
        current_tokens = 0
        added_long_contexts = []
        try:
            # Test template to see base amount of tokens required (without context)
            test_template = VOORBURG_ANSWER_PROMPT.replace("<<<<<<<<<<query>>>>>>>>>>", query).replace("<<<<<<<<<<context>>>>>>>>>>","")
            test_template_required_tokens = self.utility_functions._count_num_tokens(test_template, "gpt-4o-mini")

            # Maximum tokens available
            max_context_tokens = max_template_tokens - test_template_required_tokens

            for i in range(len(retrieved_docs)):
                # check if long context exist in context already

                # Get the content and metadata for current document
                document = retrieved_docs[i]
                chunk_context = document["chunk_context"]
                long_context = document["long_context"]
                metadata = document["metadata"]
                file_link = document["file_link"]
                
                if file_link in added_long_contexts:
                    print("file already in context, skipping")
                    continue

                new_content = f"Content: \n\n {long_context} \n\n Document Metadata: \n{metadata}\n\n" # Create the potential new content
                new_content_tokens = self.utility_functions._count_num_tokens(new_content, "gpt-4o-mini") # Calculate tokens for new content
                
                # Check if adding this long context would exceed the token limit
                if current_tokens + new_content_tokens > max_context_tokens:
                    print("Long Context is too long, checking if chunk context can be used instead")
                    # If exceed just use relevant chunk instead of long context
                    new_content = f"Content: \n\n {chunk_context} \n\n Document Metadata: \n{metadata}\n\n" # Create the potential new content
                    new_content_tokens = self.utility_functions._count_num_tokens(new_content, "gpt-4o-mini")
                    if current_tokens + new_content_tokens > max_context_tokens:
                        print("Chunk Context is too long, skipping")
                        continue
                else:
                    #means long context does not exceed, and added into context
                    print("Long context fits within context limit!")
                    added_long_contexts.append(file_link)

                #append the content and update token count
                context += new_content
                current_tokens += new_content_tokens
            
        except Exception as e:
            self.logger.error(f"Error filling prompt with context: {e}")
            context = "An error occurred while retrieving context." # Experimental
        
        filled_prompt = VOORBURG_ANSWER_PROMPT.replace("<<<<<<<<<<query>>>>>>>>>>", query).replace("<<<<<<<<<<context>>>>>>>>>>", context)
        return filled_prompt
    
class ChatInterface:
    """Streamlit-based chat interface."""
    
    def __init__(self):
        self.chat_assistant = ChatAssistant()
        self._initialize_session_state()
        
    def _initialize_session_state(self):
        """Initialize Streamlit session state."""
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
    def _display_chart(self, chart_data, sources: List[str]):
        """Display Altair chart from processed data."""
        try:
            # load data_dict to df
            df = pd.DataFrame(chart_data)
            df.columns = ['Date', 'Country', 'Index ID', 'Index Value']

            # Map line to index name
            df['Date'] = pd.to_datetime(df['Date'], unit='ms')
            df['Name'] = df['Country'] + " " + df['Index ID'].map(SINGAPORE_PRICE_INDEX_MAP)

            # Scale Chart
            y_min, y_max = df['Index Value'].min(), df['Index Value'].max()
            padding = (y_max - y_min) * 0.1  # 10% padding
            y_min, y_max = y_min - padding, y_max + padding  # Adjusted limits

            # Create Altair chart
            line_chart = alt.Chart(df).mark_line().encode(
                x=alt.X('Date', title='Date', axis=alt.Axis(format='%Y-%m-%d')),
                y=alt.Y('Index Value', title='Index Value', scale=alt.Scale(domain=[y_min, y_max])),
                    color=alt.Color('Name', scale=alt.Scale(scheme='viridis')),  # Different colors for each line
                tooltip=['Date', 'Index Value', 'Name']  # Show value on hover
            ).interactive()

            # Add circles for better hover visibility
            points = alt.Chart(df).mark_circle(size=50).encode(
                x='Date',
                y='Index Value',
                color=alt.Color('Name', scale=alt.Scale(scheme='viridis')),
                tooltip=['Date', 'Index Value', 'Name']  # Ensure values appear on hover
            )

            # Combine line chart and points
            final_chart = (line_chart + points)

            # Display in Streamlit
            st.altair_chart(final_chart, use_container_width=True)
        
            st.markdown(f"Data Sources: {', '.join(sources)}")
        except Exception as e:
            st.error(f"Error displaying chart, please try again or contact technical staff if issue persists.\n Error: {e}")

    def _process_user_input(self, prompt: str):
        try:
            if not st.session_state.messages:
                chat_history = "[]"
            else:
                chat_history = json.dumps(st.session_state.messages, indent=4)
            response = self.chat_assistant.handle_query(prompt, chat_history)

            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response
            })

        except Exception as e:
            st.error(f"Error processing user input, please try again or contact technical staff if issue persists.\n Error: {e}")

        return response
    
    def _display_chat_history(self):
        try:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    # Check if message["content"] is a string
                    if isinstance(message["content"], list):
                        if len(message["content"]) == 2:
                            response = message["content"][0]
                            if 'Plot ' in response:
                                response = response.strip("Plot ") # Remove 'Plot ' from response
                                data_dict = ast.literal_eval(response)

                                # Display chart
                                self._display_chart(data_dict, message["content"][1])
                            else:
                                st.markdown(response)
                    else:
                        st.markdown(message["content"])

        except Exception as e:
            st.error(f"Error displaying chat history, please try again or contact technical staff if issue persists.\n Error: {e}")

    def run(self):
        """Main method to run the Streamlit interface."""
        st.logo("images/icons/prof_logo_w_line.png")
        st.markdown(
                """
                <style>
                    img[alt="Logo"] {
                        height: 100px !important;
                    }
                </style>
                """,
                unsafe_allow_html=True,
            )
        st.sidebar.page_link("chat_app.py", label="Home")
        st.sidebar.page_link("pages/about_us.py", label="About Us")
        st.sidebar.page_link("pages/contact_us.py", label="Contact Us")

        if not OPENAI_API_KEY:
            st.error("Missing OpenAI API key. Please check your configuration.")
            return

        
        st.title("Price Researcher Optimised for Friends")
        st.markdown("<h11 style='text-align: left; color: grey;'>Hi, my name is Price Researcher Optimised for Friends (PROF). I am here to satisfy your curiosity about Service Producer Price Indices (SPPIs).<br/>To help me answer your questions better, please be as specific as possible. Feel free to ask away!😊</h11>", unsafe_allow_html=True)

        # df_list = []

        # df = pd.read_excel("1010_testquestions.xlsx")
        prompt = st.chat_input("Ask me about economic indicators...")
        if prompt:
        # for prompt in df["Question"].tolist():
            method_placeholder = st.empty()
            #Start time
            start_time = time.time()

            #tokens used
            start_tokens = self.chat_assistant.tokens_used

            with method_placeholder.container():
                with st.spinner("Generating response..."):                  
                    response = self._process_user_input(prompt)


            #time_taken
            time_taken = time.time() - start_time
            #tokens used
            tokens_used = self.chat_assistant.tokens_used - start_tokens

            #logging chats

            self.chat_assistant.logger.info("==============================================")
            self.chat_assistant.logger.info(f"query uuid (random number): {uuid.uuid4()}")
            self.chat_assistant.logger.info(f"User Query: {prompt}")
            self.chat_assistant.logger.info(f"Response: {response}")
            self.chat_assistant.logger.info(f"Time Taken: {time_taken:.2f} seconds")
            self.chat_assistant.logger.info(f"Tokens Used: {tokens_used}")
            self.chat_assistant.logger.info("==============================================")

            #chatlog_df = pd.DataFrame({"User Query": [prompt], "Response": [response], "Time Taken": [time_taken], "Tokens Used": [tokens_used]})
            #chatlog_df.to_csv("chatlog.csv", index=False, mode="a", header=False)
        
        # if df_list:
        #     df_concat = pd.concat(df_list, ignore_index=True)
        #     df_concat.to_csv("1010_test_response.csv", index=False)
        self._display_chat_history()
        
chat_interface = ChatInterface()
chat_interface.run()
