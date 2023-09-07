import os
import openai
import chainlit as cl
import pandas as pd
import chromadb

from chainlit import user_session
from sqlalchemy import create_engine
from typing import List, Tuple, Any
from pydantic import BaseModel, Field
from llama_index import Document
from llama_index import SQLDatabase
from llama_index.agent import OpenAIAgent
from llama_index.tools.query_engine import QueryEngineTool
from llama_index.indices.struct_store.sql_query import NLSQLTableQueryEngine
from llama_index import ServiceContext
from llama_index.llms import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index import VectorStoreIndex
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.tools import FunctionTool
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.types import (
    VectorStoreInfo,
    MetadataInfo,
    ExactMatchFilter,
    MetadataFilters,
)

openai.api_key = os.environ["OPENAI_API_KEY"]

# preparation
def get_df_from_workbook(sheet_name,
                         workbook_id = '1MB1ZsQul4AB262AsaY4fHtGW4HWp2-56zB-E5xTbs2A'):
    url = f'https://docs.google.com/spreadsheets/d/{workbook_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}'
    return pd.read_csv(url)

docEmailSample1 = Document(
    text="Hey KD, let's grab dinner after our next game, Steph", 
    metadata={'from_to': 'Stephen Curry to Kevin Durant',}
)
docEmailSample2 = Document(
    text="Yo Joker, you were a monster last year, can't wait to play against you in the opener! Draymond", 
    metadata={'from_to': 'Draymond Green to Nikola Jokic',}
)
docEmailSample3 = Document(
    text="Hey LeBron, you ready for another showdown? Let's see if you can handle the Splash Bros again! 😜",
    metadata={'from_to': 'Klay Thompson to LeBron James'}
)
docEmailSample4 = Document(
    text="Yo Giannis, you sure you want to come to the Bay? We don't have any deer to hunt here! 😂",
    metadata={'from_to': 'Draymond Green to Giannis Antetokounmpo'}
)
docEmailSample5 = Document(
    text="Hey Luka, you're a beast on the court. Let's swap jerseys after our game and show some love! 💪",
    metadata={'from_to': 'Andrew Wiggins to Luka Dončić'}
)
docEmailSample5 = Document(
    text="Devin Booker, we could use your scoring in the Bay. Think about it, bro! 🏀",
    metadata={'from_to': 'Klay Thompson to Devin Booker'}
)
docEmailSample6 = Document(
    text="Tatum, you've got the skills. Let's team up and bring some championships to the Warriors! 💍",
    metadata={'from_to': 'Draymond Green to Jayson Tatum'}
)
docAdditionalSamples = [docEmailSample1, docEmailSample2, docEmailSample3, 
                        docEmailSample4, docEmailSample5, docEmailSample6]



class AutoRetrieveModel(BaseModel):
    query: str = Field(..., description="natural language query string")
    filter_key_list: List[str] = Field(
        ..., description="List of metadata filter field names"
    )
    filter_value_list: List[str] = Field(
        ...,
        description=(
            "List of metadata filter field values (corresponding to names specified in filter_key_list)"
        )
    )
    
def auto_retrieve_fn(
    query: str, filter_key_list: List[str], filter_value_list: List[str]
):
    """Auto retrieval function.

    Performs auto-retrieval from a vector database, and then applies a set of filters.

    """
    query = query or "Query"
    
    # for i, (k, v) in enumerate(zip(filter_key_list, filter_value_list)):
    #     if k == 'token_list':
    #         if token not in v:
    #             v = ''

    exact_match_filters = [
        ExactMatchFilter(key=k, value=v)
        for k, v in zip(filter_key_list, filter_value_list)
    ]
    retriever = VectorIndexRetriever(
        vector_index, filters=MetadataFilters(filters=exact_match_filters), top_k=top_k
    )
    # query_engine = vector_index.as_query_engine(filters=MetadataFilters(filters=exact_match_filters))
    query_engine = RetrieverQueryEngine.from_args(retriever)

    response = query_engine.query(query)
    return str(response)

def auto_retrieve_fn_strategy(
    query: str, filter_key_list: List[str], filter_value_list: List[str]
):
    """Auto retrieval function.

    Performs auto-retrieval from a vector database, and then applies a set of filters.

    """
    query = query or "Query"
    
    # for i, (k, v) in enumerate(zip(filter_key_list, filter_value_list)):
    #     if k == 'token_list':
    #         if token not in v:
    #             v = ''

    exact_match_filters = [
        ExactMatchFilter(key=k, value=v)
        for k, v in zip(filter_key_list, filter_value_list)
    ]
    retriever = VectorIndexRetriever(
        vector_index_strategy, filters=MetadataFilters(filters=exact_match_filters), top_k=top_k
    )
    query_engine = RetrieverQueryEngine.from_args(retriever)

    response = query_engine.query(query)
    return str(response)

# loading CSV data
sheet_names = ['Teams', 'Players', 'Schedule', 'Player_Stats', 'GSW_Players_Salary']
dict_of_dfs = {sheet: get_df_from_workbook(sheet) for sheet in sheet_names}

engine = create_engine("sqlite+pysqlite:///:memory:")

for df in dict_of_dfs:
    dict_of_dfs[df].to_sql(df, con=engine)

sql_database = SQLDatabase(
    engine,
    include_tables=list(dict_of_dfs.keys())
    )

# setting up llm & service content
embed_model = OpenAIEmbedding()
chunk_size = 1000
llm = OpenAI(
    temperature=0, 
    model="gpt-3.5-turbo",
    streaming=True
)
service_context = ServiceContext.from_defaults(
    llm=llm, 
    chunk_size=chunk_size,
    embed_model=embed_model
)

# setting up vector store
chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection("all_data")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
vector_index = VectorStoreIndex([], storage_context=storage_context, service_context=service_context)

vector_index.insert_nodes(docAdditionalSamples)

# setting up metadata
top_k = 3
info_emails_players = VectorStoreInfo(
    content_info="Emails exchanged between NBA players.",
    metadata_info=[
        MetadataInfo(
            name="from_to",
            type="str",
            # description="email sent to an NBA player by a Golden States Warriors player, one of  [Stephen Curry, Andrew Wiggins, 'Draymond Green', 'Klay Thompson'], to any other NBA player"
            description="""
email sent from a Golden States Warriors player to another NBA player, one of 
Stephen Curry to Kevin Durant,
Draymond Green to Nikola Jokic,
Klay Thompson to LeBron James,
Draymond Green to Giannis Antetokounmpo,
Andrew Wiggins to Luka Dončić,
Klay Thompson to Devin Booker,
Draymond Green to Jayson Tatum
"""
        ), 
    ]
)

strategy1 = Document(
    text="Against the Phoenix Suns, we'll focus on ball movement and three-point shooting. Our starting five will consist of Stephen Curry, Klay Thompson, Andrew Wiggins, Draymond Green, and James Wiseman. We'll exploit their defense with our perimeter shooting while Draymond Green handles the playmaking and defense in the paint.",
    metadata={'opponent': 'Phoenix Suns'}
)
strategy2 = Document(
    text="Facing the Lakers, we'll emphasize defensive intensity and fast-break opportunities. Our starting lineup will feature Stephen Curry, Klay Thompson, Andrew Wiggins, Draymond Green, and Kevon Looney. We need to limit LeBron's impact and push the pace on offense to tire their older roster.",
    metadata={'opponent': 'Lakers'}
)
strategy3 = Document(
    text="Against the Denver Nuggets, our strategy is to control the boards and exploit their interior defense. Starting with Stephen Curry, Klay Thompson, Andrew Wiggins, Draymond Green, and James Wiseman, we aim to dominate the paint on both ends. We'll also look for opportunities to run in transition.",
    metadata={'opponent': 'Denver Nuggets'}
)
strategy4 = Document(
    text="Facing the Milwaukee Bucks, we'll prioritize perimeter defense and transition play. Our starting five will consist of Stephen Curry, Klay Thompson, Andrew Wiggins, Draymond Green, and Kevon Looney. We must limit Giannis' drives to the basket and exploit their defense with quick ball movement.",
    metadata={'opponent': 'Milwaukee Bucks'}
)
strategy5 = Document(
    text="In the matchup against the Brooklyn Nets, we'll focus on high-scoring games and exploiting defensive weaknesses. Our starting lineup will include Stephen Curry, Klay Thompson, Andrew Wiggins, Draymond Green, and Kevon Looney. We'll aim to outshoot and outpace them in a high-octane offensive battle.",
    metadata={'opponent': 'Brooklyn Nets'}
)

chroma_client_strategy = chromadb.Client()
chroma_collection = chroma_client_strategy.create_collection("coach_data")
vector_store_strategy = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context_strategy = StorageContext.from_defaults(vector_store=vector_store_strategy)
vector_index_strategy = VectorStoreIndex([], storage_context=storage_context_strategy, service_context=service_context)

vector_index_strategy.insert_nodes([strategy1, strategy2, strategy3, strategy4, strategy5])

# setting up metadata
top_k = 1
info_strategy = VectorStoreInfo(
    content_info="Game strategy by Steve Kerr for the Golden States Warriors against opponent NBA teams.",
    metadata_info=[
        MetadataInfo(
            name="opponent",
            type="str",
            description="""
Game strategy for Golden State Warriors against opponent NBA teams, one of [Phoenix Suns, Lakers, Nuggets, Milwaukee Buck, Brooklyn Nets].
"""
        ), 
    ]
)

@cl.on_chat_start
def main():
   
    sql_query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=list(dict_of_dfs.keys())
    )
    
    sql_nba_tool = QueryEngineTool.from_defaults(
        query_engine=sql_query_engine, # 
        name='sql_nba_tool', 
        description=("""Do not use this tool for queries related to game strategy.
                        Use this tool for translating a natural language query into a SQL query over tables containing:
                        1. teams, containing historical information about NBA teams
                        2. players, containing information about the team that each player plays for
                        3. schedule, containing information related to the entire NBA game schedule
                        4. player_stats, containing information related to all NBA player stats
                        5. GSW_players_salary, containing information related to the salary of Golden State Warriors players
                        """
        ),
    )
    
    description_emails = f"""\
    Use this tool to retrieve emails between NBA players.
    The vector database schema is given below:
    {info_emails_players.json()}
    """
    auto_retrieve_tool_emails = FunctionTool.from_defaults(
        fn=auto_retrieve_fn, 
        name='auto_retrieve_tool_emails',
        description=description_emails, 
        fn_schema=AutoRetrieveModel
    )
    
    description_strategy = f"""\
    Use this tool to look up information about the game strategy of Golden State Warriors against other NBA teams.
    The vector database schema is given below:
    {info_strategy.json()}
    """
    auto_retrieve_tool_strategy = FunctionTool.from_defaults(
        fn=auto_retrieve_fn_strategy, 
        name='auto_retrieve_tool_strategy',
        description=description_strategy, 
        fn_schema=AutoRetrieveModel
    )
    
    agent = OpenAIAgent.from_tools(
    # agent = ReActAgent.from_tools(
        tools = [auto_retrieve_tool_strategy,
                 sql_nba_tool, 
                 auto_retrieve_tool_emails,
                ], 
        llm=llm, 
        verbose=True,
    )
    
    cl.user_session.set("agent", agent)
    
@cl.on_message
async def main(message):
    agent = cl.user_session.get("agent") 
    
    # response = agent.chat(message.content)
    response = agent.chat(message)
    
    response_message = cl.Message(content="")

    # for token in response.response:
    #     await response_message.stream_token(token=token)

    if response.response:
        response_message.content = response.response

    await response_message.send()
