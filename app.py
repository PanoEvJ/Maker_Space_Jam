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

docEmailSample = Document(
    text="Hey KD, let's grab dinner after our next game, Steph", 
    metadata={'from_to': 'Stephen Curry to Kevin Durant',}
)
docEmailSample2 = Document(
    text="Yo Joker, you were a monster last year, can't wait to play against you in the opener! Draymond", 
    metadata={'from_to': 'Draymond Green to Nikola Jokic',}
)
docAdditionalSamples = [docEmailSample, docEmailSample2]

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

# loading CSV data
sheet_names = ['Teams', 'Players', 'Schedule', 'Player_Stats']
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
    content_info="emails exchanged between NBA players",
    metadata_info=[
        MetadataInfo(
            name="from_to",
            type="str",
            description="""
email sent by a player of the Golden State Warriors to any other NBA player, one of [
Stephen Curry to any NBA player, 
Klay Thompson to any NBA player, 
Chris Paul to any NBA player, 
Andrew Wiggins to any NBA player, 
Draymond Green to any NBA player, 
Gary Payton II to any NBA player, 
Kevon Looney to any NBA player, 
Jonathan Kuminga to any NBA player, 
Moses Moody to any NBA player, 
Brandin Podziemski to any NBA player, 
Cory Joseph to any NBA player, 
Dario Šarić to any NBA player]
Access these emails only when you are one of the player that sent the email."""
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
        description=("""Useful for translating a natural language query into a SQL query over tables containing:
                        1. teams, containing information related to all NBA teams
                        2. players, containing information about the team that each player plays for
                        3. schedule, containing information related to the entire NBA game schedule
                        4. player_stats, containing information related to all NBA player stats
                        """
        ),
    )
    
    description_emails = f"""\
    Use this tool to look up information about emails exchanged betweed players of the Golden State Warriors and any other NBA player.
    Use this tool only when you are the player that actually sent the email.
    The vector database schema is given below:
    {info_emails_players.json()}
    """
    auto_retrieve_tool_emails = FunctionTool.from_defaults(
        fn=auto_retrieve_fn, 
        name='auto_retrieve_tool_emails',
        description=description_emails, 
        fn_schema=AutoRetrieveModel
    )
    
    agent = OpenAIAgent.from_tools(
    # agent = ReActAgent.from_tools(
        tools = [sql_nba_tool, 
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
