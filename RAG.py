import os
import pandas as pd
import tiktoken
import yaml
import shutil
from graphrag.query.indexer_adapters import read_indexer_entities, read_indexer_reports
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.structured_search.global_search.community_context import (
    GlobalCommunityContext,
)
from graphrag.query.structured_search.global_search.search import GlobalSearch
from dotenv import load_dotenv

load_dotenv()

# 讀取環境變量
api_key = os.getenv("GRAPHRAG_API_KEY")
llm_model = os.getenv("GRAPHRAG_LLM_MODEL")


llm = ChatOpenAI(
    api_key=api_key,
    model=llm_model,
    api_type=OpenaiApiType.OpenAI,  # OpenaiApiType.OpenAI or OpenaiApiType.AzureOpenAI
    max_retries=20,
)

token_encoder = tiktoken.get_encoding("cl100k_base")

# parquet files generated from indexing pipeline
INPUT_DIR = "./inputs/operation dulce"
COMMUNITY_REPORT_TABLE = "create_final_community_reports"
ENTITY_TABLE = "create_final_nodes"
ENTITY_EMBEDDING_TABLE = "create_final_entities"

# community level in the Leiden community hierarchy from which we will load the community reports
# higher value means we use reports from more fine-grained communities (at the cost of higher computation cost)
COMMUNITY_LEVEL = 2
context_builder_params = {
    "use_community_summary": False,  # False means using full community reports. True means using community short summaries.
    "shuffle_data": True,
    "include_community_rank": True,
    "min_community_rank": 0,
    "community_rank_name": "rank",
    "include_community_weight": True,
    "community_weight_name": "occurrence weight",
    "normalize_community_weight": True,
    "max_tokens": 12_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
    "context_name": "Reports",
}

map_llm_params = {
    "max_tokens": 1000,
    "temperature": 0.0,
    "response_format": {"type": "json_object"},
}

reduce_llm_params = {
    "max_tokens": 2000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 1000-1500)
    "temperature": 0.0,
}


class RAG:
    def __init__(self, address):
        self.address = address
        os.makedirs(self.address, exist_ok=True)
        os.makedirs(f"{self.address}/input", exist_ok=True)
    def setting(self):
        print("start setting")
        os.system(f"python -m graphrag.index --init --root {self.address}")
        with open(f"{self.address}/settings.yaml", 'r') as file:
            settings = yaml.safe_load(file)
            settings['llm']['api_key'] = api_key
            settings['llm']['model'] = llm_model
            settings['embeddings']['llm']['api_key'] = api_key
        with open(f"{self.address}/settings.yaml", "w") as file:
            yaml.safe_dump(settings, file)
    def indexing(self):
        print("start indexing")
        os.system(f"python -m graphrag.index --root {self.address}")
        print("indexing over")
        """
        output_dir = os.path.join(self.address, "output")
        result_dir = os.path.join(self.address, "result")
        # 找出 output 資料夾內唯一的子資料夾
        subfolders = [f.path for f in os.scandir(output_dir) if f.is_dir()]
        if len(subfolders) == 1:
            unique_subfolder = subfolders[0]
            # 如果已存在 result 資料夾，先刪除
            if os.path.exists(result_dir):
                shutil.rmtree(result_dir)
                print(f"已刪除原本存在的資料夾: {result_dir}")
            # 移動子資料夾並重命名為 result
            shutil.move(unique_subfolder, result_dir)
            print(f"已將子資料夾重新命名並移動為: {result_dir}")
            # 刪除 output 資料夾
            shutil.rmtree(output_dir)
            print(f"已刪除原始的資料夾: {output_dir}")
        else:
            print(f"子資料夾數量不唯一，找到的子資料夾：{subfolders}")
        """

    def global_search(self, query):
        print(f"start global search: {query}")
        result_address = f"{self.address}/output"
        entity_df = pd.read_parquet(f"{result_address}/{ENTITY_TABLE}.parquet")
        report_df = pd.read_parquet(f"{result_address}/{COMMUNITY_REPORT_TABLE}.parquet")
        entity_embedding_df = pd.read_parquet(f"{result_address}/{ENTITY_EMBEDDING_TABLE}.parquet")
        reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)
        entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)
        print(f"Total report count: {len(report_df)}")
        print(
            f"Report count after filtering by community level {COMMUNITY_LEVEL}: {len(reports)}"
        )
        report_df.head()

        context_builder = GlobalCommunityContext(
            community_reports=reports,
            entities=entities,  # default to None if you don't want to use community weights for ranking
            token_encoder=token_encoder,
        )
        search_engine = GlobalSearch(
            llm=llm,
            context_builder=context_builder,
            token_encoder=token_encoder,
            max_data_tokens=12_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
            map_llm_params=map_llm_params,
            reduce_llm_params=reduce_llm_params,
            allow_general_knowledge=False,  # set this to True will add instruction to encourage the LLM to incorporate general knowledge in the response, which may increase hallucinations, but could be useful in some use cases.
            json_mode=True,  # set this to False if your LLM model does not support JSON mode.
            context_builder_params=context_builder_params,
            concurrent_coroutines=32,
            response_type="multiple paragraphs",  # free form text describing the response type and format, can be anything, e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
        )
        result = search_engine.search(query)
        print(result.response)
        # inspect the data used to build the context for the LLM responses
        result.context_data["reports"]
        # inspect number of LLM calls and tokens
        print(f"LLM calls: {result.llm_calls}. LLM tokens: {result.prompt_tokens}")
        return result.response

if __name__ == "__main__":
    test = RAG('./test')
    test.setting()
    print("請將文件放入 input 資料夾中，按下 Enter 鍵以繼續...")
    input()  
    test.indexing()
    test.global_search("What happened to Chinese?")











