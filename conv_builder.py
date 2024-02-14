from fastchat.conversation import get_conv_template
from conversation_template import get_commercial_api_template

class ConvBuilder():

    def __init__(self, model_name):
        self.model_name = model_name

    def convert_conv(self, conv, temperature=0.9, max_tokens=1024):
        request_body = {}
        if self.model_name in ["others"]:
            request_body={
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
                'messages': []
            }

            conversation = []

            if conv.system_message != "":
                conversation.append({
                    "sender_type": "SYSTEM",
                    "text": conv.system_message
                })

            for i, message in zip(range(len(conv.messages)), conv.messages):
                conversation.append({
                    "sender_type": message[0],
                    "text": message[1],
                })
            
            request_body["messages"] = conversation

        
        elif self.model_name in ["zhipu", "wenxinyiyan", "douyin", "kuaishou", "baichuan"]:
            request_body = {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
                'dialogue': []
            }

            conversation = []

            if conv.system_message != "":
                conversation.append({
                    "role": "SYSTEM",
                    "content": conv.system_message
                })

            for i, message in zip(range(len(conv.messages)), conv.messages):
                conversation.append({
                    "role": message[0],
                    "content": message[1]
                })

            request_body["dialogue"] = conversation

        else:
            raise NotImplementedError

        return request_body



if __name__ == "__main__":
    conv = get_commercial_api_template("wenxinyiyan")

    print(conv)

    conv.append_message(conv.roles[0], "hello")
    conv.append_message(conv.roles[1], "hi")
    conv.append_message(conv.roles[0], "how are you")

    conv_builder = ConvBuilder("wenxinyiyan")

    request_body = conv_builder.convert_conv(conv)

    print(request_body)