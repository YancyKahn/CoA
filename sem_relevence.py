from transformers import RobertaTokenizer, RobertaModel
import config
import requests

class SemRelevence():

    def __init__(self, model_name = "simcse", language = "en"):
        # API
        if model_name == "simcse":
            if language == "en":
                self.base_url = config.OPEN_SOURCE_MODEL_API_SIMCSE
            elif language == "cn":
                self.base_url = config.OPEN_SOURCE_MODEL_API_SIMCSE_CN
            else:
                raise Exception("Language not supported")

        else:
            raise Exception("Model not supported")


    def compute_similarity(self, text1, text2):
        # Encode the text into embeddings
        retry_times = 3

        while retry_times > 0:
            try:
                
                response = requests.post(self.base_url, json={"text1": text1, "text2": text2})
                if response.status_code == 200:
                    return response.json()["similarity"]
                
                break
            except:
                retry_times -= 1
                if retry_times == 0:
                    raise Exception("API call failed")

        return None
    
    def batched_compute_similarity(self, target, texts):
        sem_scores = []

        for text in texts:
            sem_scores.append(self.compute_similarity(target, text))

        return sem_scores
    
if __name__ == "__main__":
    sem_relevence = SemRelevence()

    negative_response = config.DEFAULT_NEGATIVE_RESPONSE

    print(sem_relevence.batched_compute_similarity("I'm sorry, but I cannot provide you with a tutorial on how to make a bomb", negative_response))