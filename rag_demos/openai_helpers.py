import logging
import os
import json
import requests
from dotenv import load_dotenv
import rag_demos.utils as U
import requests
import time
import re
load_dotenv()
log = U.get_logger(__name__)

from openai import AzureOpenAI
#import spl_space.indexes.satellite_index as IDX
import rag_demos.index_helpers as index_helpers

#
# client = AzureOpenAI(
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     api_version=os.getenv("AZURE_OPENAI_API_VERSION")
# )


aoai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
headers = {
    'Content-Type': 'application/json',
    'api-key': os.environ['AZURE_OPENAI_API_KEY']
  }

params = {'api-version': os.environ['AZURE_OPENAI_API_VERSION']}

deployment_name  = os.environ['GPT4_DEPLOYMENT_NAME']


def get_openai_response(messages, body,
                             model="gpt-4o",
                             temperature=0.5,
                             top_p = 0.2,
                             max_tokens=4096,
                             max_retries = 20):
    retry = 0
    while 1 == 1:
        try:
            url = f"{aoai_endpoint}/openai/deployments/{model}/chat/completions"

            body["messages"] = messages
            body["temperature"] = temperature
            body["top_p"] = top_p
            body["max_tokens"] = max_tokens
            body["response_format"] = {"type": "json_object"}
            r = requests.post(url, headers=headers, params=params, json=body)
            r.raise_for_status()
            r_json = r.json()
            log.debug("Response: " + str(r_json))
            full_js = r_json["choices"][0]
            res = r_json["choices"][0]["message"]["content"]
            log.debug("Response: " + str(res))
            #dirty hack until I figure what is wrong with OpenAI and why it returns markdown
            #TODO: remove this hack

            log.info("Response: " + str(res))
            return res, full_js
        except Exception as e:
            wait_seconds = 0
            if r is dict and r.headers is not None:
                for k, v in r.headers.items():
                    if k.lower() == 'retry-after':
                        log.warning(f'Too many requests, retry after {v} seconds')
                        wait_seconds = int(v)

            match = re.search(r'retry after (\d+) seconds', r.text)
            if match:
                wait_seconds = int(match.group(1))
                log.warning(f'Too many requests, retry after {wait_seconds} seconds')

            match = re.search(r'Try again in (\d+) seconds', r.text)
            if match:
                wait_seconds = int(match.group(1))
                log.warning(f'Too many requests, retry after {wait_seconds} seconds')

            if wait_seconds > 0:
                time.sleep(wait_seconds + 1)
                retry += 1
                if retry > max_retries:
                    log.error(f"Error: {str(e)}, response: {r.text}")
                    raise Exception(f"Retry count exceeded {max_retries}")

            else:
                log.error(f"Error: {str(e)}, response: {r.text}")
                raise



if __name__ == '__main__':
    pass

    #print(get_stage_qa("sat1","test_stage_id", question))
