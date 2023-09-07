import os
import openai
import chainlit as cl
import pandas as pd

from chainlit import user_session
from sqlalchemy import create_engine
from llama_index import SQLDatabase
from llama_index.agent import OpenAIAgent
from llama_index.tools.query_engine import QueryEngineTool
from llama_index.indices.struct_store.sql_query import NLSQLTableQueryEngine
from llama_index import ServiceContext
from llama_index.llms import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

openai.api_key = os.environ["OPENAI_API_KEY"]

def get_df_from_workbook(sheet_name,
                         workbook_id = '1MB1ZsQul4AB262AsaY4fHtGW4HWp2-56zB-E5xTbs2A'):
    url = f'https://docs.google.com/spreadsheets/d/{workbook_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}'
    return pd.read_csv(url)

sheet_names = ['Teams', 'Players', 'Schedule', 'Player_Stats']
dict_of_dfs = {sheet: get_df_from_workbook(sheet) for sheet in sheet_names}


engine = create_engine("sqlite+pysqlite:///:memory:")

for df in dict_of_dfs:
    dict_of_dfs[df].to_sql(df, con=engine)

sql_database = SQLDatabase(
    engine,
    include_tables=list(dict_of_dfs.keys())
    )


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
    
    agent = OpenAIAgent.from_tools(
    # agent = ReActAgent.from_tools(
        [sql_nba_tool, 
        # auto_retrieve_tool_emails,
        #  auto_retrieve_tool_schedule
        ], 
        llm=llm, verbose=True,
    )
    
    cl.user_session.set("agent", agent)
    
@cl.on_message
async def main(message):
    agent = cl.user_session.get("agent") 
    
    # response = agent.chat(message.content)
    response = agent.chat(message)
    
    response_message = cl.Message(content="")

    for token in response.response:
        await response_message.stream_token(token=token)

    if response.response:
        response_message.content = response.response

    await response_message.send()
