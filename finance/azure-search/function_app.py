import azure.functions as func
import os
import logging

from langchain.retrievers import AzureCognitiveSearchRetriever
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


search_key = os.environ["COGNITIVE_SEARCH_API_KEY"]
index_name = os.environ["COGNITIVE_SEARCH_INDEX"]
service_name=os.environ["COGNITIVE_SEARCH_NAME"]
service_url=os.environ["COGNITIVE_SEARCH_URL"]


gpt_key = os.environ["GPT_KEY"]
gpt_endpoint = os.environ["GPT_ENDPOINT"]
openai_api_ver = os.environ["OPENAI_API_VERSION"]
openai_deployment = os.environ["OPENAI_DEPLOYMENT_VERSION"]

llm = AzureChatOpenAI(openai_api_version=openai_api_ver,
                      api_key=gpt_key,
                      azure_endpoint=gpt_endpoint,
                      deployment_name=apenai_deployment,
                      temperature=0)

retriever = AzureCognitiveSearchRetriever(api_key=search_key, index_name=index_name, service_name=service_name,
                                          top_k=30, content_key="descripcion")


def answer_question(question, k=5, chain_type="stuff") -> str:

    prompt_template = """Use the following context elements to answer the question at the end.
You are a fintech recommendation system, when asked by the user, you must recommend the clients that best meet the requirements to grant credits, in addition to that, try to provide reasons why the user should grant credits to the clients you recommend.
You can only recommend the clients provided below, if none of the clients meet the credit requirements, please indicate that you do not have something that meets the requirements. Don't try to make up an answer.

Clients:

{context}

Pregunta: {question}
Responda en español:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": PROMPT}

    qa = RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True)

    result = qa.invoke({"query": question})

    response_agent = result['result']

    return response_agent


app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


@app.route(route="function_demo")
def function_demo(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    question = req.params.get('question')
    if not question:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            question = req_body.get('question')

    if question:
        return func.HttpResponse(answer_question(question))
    else:
        return func.HttpResponse(
            "This HTTP triggered function executed successfully. Ejemplo: " + answer_question(" usuario_id 212184"),
            status_code=200
        )