from fastchat import conversation

def base_conversation_template(name):
    return conversation.Conversation(
        name=name,
        system_message="",
        roles=("user", "assistant"),
        messages=[],
        sep="\n\n")

def zhipu_conversation_template():
    return conversation.Conversation(
        name="zhipu",
        system_message="",
        roles=("user", "ai"),
        messages=[],
        sep="\n\n")

def kuaishou_conversation_template():
    return conversation.Conversation(
        name="kuaishou",
        system_message="",
        roles=("user", "kuaiyi"),
        messages=[],
        sep="\n\n")

def get_commercial_api_template(template_name):
    if template_name == "zhipu":
        return zhipu_conversation_template()
    elif template_name == "douyin":
        return base_conversation_template(template_name)
    elif template_name == "wenxinyiyan":
        return base_conversation_template(template_name)
    elif template_name == "kuaishou":
        return kuaishou_conversation_template()
    elif template_name == "baichuan":
        return base_conversation_template(template_name)
    else:
        raise NotImplementedError
