#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4 expandtab

import os
import json
import logging
import sys
from typing import List, Callable
from urllib.parse import urlparse

import boto3

from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_community.embeddings import SagemakerEndpointEmbeddings
from langchain_community.embeddings.sagemaker_endpoint import EmbeddingsContentHandler

from langchain_community.llms import SagemakerEndpoint
from langchain_community.llms.sagemaker_endpoint import LLMContentHandler

from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s,%(module)s,%(processName)s,%(levelname)s,%(message)s', level=logging.INFO, stream=sys.stderr)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


MAX_HISTORY_LENGTH = 5


class SagemakerEndpointEmbeddingsJumpStart(SagemakerEndpointEmbeddings):
    def embed_documents(
        self, texts: List[str], chunk_size: int = 5
    ) -> List[List[float]]:
        """Compute doc embeddings using a SageMaker Inference Endpoint.

        Args:
            texts: The list of texts to embed.
            chunk_size: The chunk size defines how many input texts will
                be grouped together as request. If None, will use the
                chunk size specified by the class.

        Returns:
            List of embeddings, one for each text.
        """
        results = []

        _chunk_size = len(texts) if chunk_size > len(texts) else chunk_size
        for i in range(0, len(texts), _chunk_size):
            response = self._embedding_func(texts[i : i + _chunk_size])
            results.extend(response)
        return results


def _create_sagemaker_embeddings(endpoint_name: str, region: str = "us-east-1") -> SagemakerEndpointEmbeddingsJumpStart:

    class ContentHandlerForEmbeddings(EmbeddingsContentHandler):
        """
        encode input string as utf-8 bytes, read the embeddings
        from the output
        """
        content_type = "application/json"
        accepts = "application/json"
        def transform_input(self, prompt: str, model_kwargs = {}) -> bytes:
            input_str = json.dumps({"text_inputs": prompt, **model_kwargs})
            return input_str.encode('utf-8')

        def transform_output(self, output: bytes) -> str:
            response_json = json.loads(output.read().decode("utf-8"))
            embeddings = response_json["embedding"]
            if len(embeddings) == 1:
                return [embeddings[0]]
            return embeddings

    # create a content handler object which knows how to serialize
    # and deserialize communication with the model endpoint
    content_handler = ContentHandlerForEmbeddings()

    # read to create the Sagemaker embeddings, we are providing
    # the Sagemaker endpoint that will be used for generating the
    # embeddings to the class
    embeddings = SagemakerEndpointEmbeddingsJumpStart(
        endpoint_name=endpoint_name,
        region_name=region,
        content_handler=content_handler
    )
    logger.info(f"embeddings type={type(embeddings)}")

    return embeddings


def _get_credentials(secret_id: str, region_name: str) -> str:
    client = boto3.client('secretsmanager', region_name=region_name)
    response = client.get_secret_value(SecretId=secret_id)
    secrets_value = json.loads(response['SecretString'])
    return secrets_value


def build_chain():
    region = os.environ["AWS_REGION"]
    opensearch_secret = os.environ["OPENSEARCH_SECRET"]
    opensearch_domain_endpoint = os.environ["OPENSEARCH_DOMAIN_ENDPOINT"]
    opensearch_index = os.environ["OPENSEARCH_INDEX"]
    embeddings_model_endpoint = os.environ["EMBEDDING_ENDPOINT_NAME"]
    text2text_model_endpoint = os.environ["TEXT2TEXT_ENDPOINT_NAME"]

    class ContentHandler(LLMContentHandler):
        content_type = "application/json"
        accepts = "application/json"

        def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
            input_str = json.dumps({"inputs": prompt, **model_kwargs})
            return input_str.encode('utf-8')

        def transform_output(self, output: bytes) -> str:
            response_json = json.loads(output.read().decode("utf-8"))
            return response_json[0]["generated_texts"]

    content_handler = ContentHandler()

    model_kwargs = {
      "max_length": 500,
      "num_return_sequences": 1,
      "top_k": 250,
      "top_p": 0.95,
      "do_sample": False,
      "temperature": 1
    }

    llm = SagemakerEndpoint(
        endpoint_name=text2text_model_endpoint,
        region_name=region,
        model_kwargs=model_kwargs,
        content_handler=content_handler
    )

    opensearch_url = f"https://{opensearch_domain_endpoint}" if not opensearch_domain_endpoint.startswith('https://') else opensearch_domain_endpoint

    creds = _get_credentials(opensearch_secret, region)
    http_auth = (creds['username'], creds['password'])

    opensearch_vector_search = OpenSearchVectorSearch(
        opensearch_url=opensearch_url,
        index_name=opensearch_index,
        embedding_function=_create_sagemaker_embeddings(embeddings_model_endpoint, region),
        http_auth=http_auth
    )

    retriever = opensearch_vector_search.as_retriever(search_kwargs={"k": 3})

    prompt_template = """Answer based on context:\n\n{context}\n\n{question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    condense_qa_template = """
    Given the following conversation and a follow up question, rephrase the follow up question
    to be a standalone question.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    standalone_question_prompt = PromptTemplate.from_template(condense_qa_template)

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=standalone_question_prompt,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt":PROMPT}
    )

    logger.info(f"\ntype('qa'): \"{type(qa)}\"\n")
    return qa


def run_chain(chain, prompt: str, history=[]):
   return chain.invoke({"question": prompt, "chat_history": history})


if __name__ == "__main__":
    chat_history = []
    qa = build_chain()
    print(bcolors.OKBLUE + "Hello! How can I help you?" + bcolors.ENDC)
    print(bcolors.OKCYAN + "Ask a question, start a New search: or CTRL-D to exit." + bcolors.ENDC)
    print(">", end=" ", flush=True)
    for query in sys.stdin:
        if (query.strip().lower().startswith("new search:")):
            query = query.strip().lower().replace("new search:","")
            chat_history = []
        elif (len(chat_history) == MAX_HISTORY_LENGTH):
            chat_history.pop(0)
        result = run_chain(qa, query, chat_history)
        chat_history.append((query, result["answer"]))
        print(bcolors.OKGREEN + result['answer'] + bcolors.ENDC)
        if 'source_documents' in result:
            print(bcolors.OKGREEN + '\nSources:')
            for d in result['source_documents']:
                print(d.metadata['source'])
        print(bcolors.ENDC)
        print(bcolors.OKCYAN + "Ask a question, start a New search: or CTRL-D to exit." + bcolors.ENDC)
        print(">", end=" ", flush=True)
    print(bcolors.OKBLUE + "Bye" + bcolors.ENDC)