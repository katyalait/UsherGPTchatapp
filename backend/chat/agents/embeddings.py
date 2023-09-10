# # imports
# import ast  # for converting embeddings saved as strings back to arrays
# import openai  # for calling the OpenAI API
# import tiktoken  # for counting tokens
# from scipy import spatial  # for calculating vector similarities for search
#
# # Use a pipeline as a high-level helper
# from transformers import pipeline
# import torch
# import pandas as pd
#
# import faiss
# import numpy as np
#
# # models
# EMBEDDING_MODEL = "text-embedding-ada-002"
# GPT_MODEL = "gpt-3.5-turbo"
# MODEL = "medalpaca/medalpaca-7b"
# embeddings_path = "data/vector_embeddings.csv"
#
# df = pd.read_csv(embeddings_path)
#
# # convert embeddings from CSV str type back to list type
# df['embedding'] = df['embedding'].apply(ast.literal_eval)
#
# # search function
# def strings_ranked_by_faiss(
#         query: str,
#         df: pd.DataFrame,
#         vector_db,
#         top_n: int = 100
# ) -> tuple[list[str], list[float]]:
#     """Returns a list of strings and relatednesses, sorted from most related to least."""
#     query_embedding_response = openai.Embedding.create(
#         model=EMBEDDING_MODEL,
#         input=query,
#     )
#     # Get the query embedding
#     query_embedding = query_embedding_response["data"][0]["embedding"]
#     # Convert the embedding to array type
#     query_embedding = np.array([np.asarray(query_embedding, dtype=np.float32)])
#     # Do the same normalization step as we did when preparing the vector database
#     faiss.normalize_L2(query_embedding)
#     # Search
#     distances, ann = vector_db.search(query_embedding, k=top_n)
#     # TODO: Figure out why this is needed
#     distances = distances[0]
#     ann = ann[0]
#
#     strings = []
#     scores = []
#
#     for idx, entry in enumerate(distances):
#         strings.append(df["text"][ann[idx]])
#         scores.append(entry)
#
#     return strings[:top_n], scores[:top_n]
#
#
# def create_faiss_db(df):
#     # Load df into faiss
#     embeddings = []
#
#     for i, row in df.iterrows():
#         embedding = np.asarray(row["embedding"], dtype=np.float32)
#         embeddings.append(embedding)
#
#     embeddings = np.asarray(embeddings)
#     vector_dimension = embeddings.shape[1]
#     db = faiss.IndexFlatL2(vector_dimension)
#     faiss.normalize_L2(embeddings)
#     db.add(embeddings)
#
#     return db
#
#
# def strings_ranked_by_relatedness(
#     query: str,
#     df: pd.DataFrame,
#     relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
#     top_n: int = 100
# ) -> tuple[list[str], list[float]]:
#     """Returns a list of strings and relatednesses, sorted from most related to least."""
#     query_embedding_response = openai.Embedding.create(
#         model=EMBEDDING_MODEL,
#         input=query,
#     )
#     query_embedding = query_embedding_response["data"][0]["embedding"]
#     strings_and_relatednesses = [
#         (row["text"], relatedness_fn(query_embedding, row["embedding"]))
#         for i, row in df.iterrows()
#     ]
#     strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
#     strings, relatednesses = zip(*strings_and_relatednesses)
#     return strings[:top_n], relatednesses[:top_n]
#
#
# def num_tokens(text: str, model: str = GPT_MODEL) -> int:
#     """Return the number of tokens in a string."""
#     encoding = tiktoken.encoding_for_model(model)
#     return len(encoding.encode(text))
#
#
# def query_message(
#         query: str,
#         df: pd.DataFrame,
#         model: str,
#         token_budget: int,
#         ranking_method: str = 'relatednesses',
# ) -> str:
#     """Return a message for GPT, with relevant source texts pulled from a dataframe."""
#
#     if ranking_method == 'relatednesses':
#         strings, relatednesses = strings_ranked_by_relatedness(query, df)
#     elif ranking_method == 'distance':
#         db = create_faiss_db(df)
#         strings, distances = strings_ranked_by_faiss(query, df, db)
#     introduction = 'Use the below textbook excerpts to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer. Where possible give quotes and references."'
#     question = f"\n\nQuestion: {query}"
#     message = introduction
#     for string in strings:
#         next_entry = f'\n\nTextbook information sections:\n"""\n{string}\n"""'
#         if (
#                 num_tokens(message + next_entry + question, model=model)
#                 > token_budget
#         ):
#             break
#         else:
#             message += next_entry
#     return message + question
#
#
# def ask_medalpaca(
#         query: str,
#         df: pd.DataFrame = df,
#         model: str = GPT_MODEL,
#         token_budget: int = 4096 - 500,
#         print_message: bool = False,
#         ranking_method: str = 'relatednesses',
# ) -> str:
#     """Answers a query using medalpaca and a dataframe of relevant texts and embeddings."""
#     pipe = pipeline(
#         "text-generation",
#         model=MODEL,
#         torch_dtype=torch.bfloat16,
#         device_map="auto",
#         model_kwargs={"max_length": 5000}
#     )
#     message = query_message(query, df, model=model, token_budget=token_budget, ranking_method=ranking_method)
#     answer = pipe(message)
#     print(f"Got answer: {answer}")
#     answer2 = answer[0]["generated_text"]
#     answer_split = answer2.split("\n")
#     answer_mod = [x for x in answer_split if x != query]
#     answer_final = " ".join(answer_mod)
#     return answer_final
