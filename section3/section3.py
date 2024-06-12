from langchain import prompts, chains, memory
from llm import gemini

# Memory management
if __name__ == "__main__":   
    # Init Model
    model = gemini.model

    # Create Memory 
    # Option 1: Store conversation into a JSON file
    memory_file = memory.ConversationBufferMemory(
        chat_memory=memory.FileChatMessageHistory('messages_history.json'),
        memory_key='messages', 
        return_messages=True
    )
    
    # Option 2: Create a summary
    memory_summary = memory.ConversationSummaryMemory(
        llm=model,
        memory_key='messages', 
        return_messages=True
    )

    # Create Chat Prompts
    prompt = prompts.ChatPromptTemplate(
        input_variables=['content', 'messages'],
        messages=[
            prompts.MessagesPlaceholder(variable_name='messages'),
            prompts.HumanMessagePromptTemplate.from_template('{content}'),
        ]
    )
    
    # Init chain
    chain = chains.LLMChain(
        llm=model,
        prompt=prompt,
        memory=memory_summary,
        verbose=True
    )
    
    while True:
        # Ask for user's input
        content = input('>> ')
        if len(content) == 0:
            break
        
        # Get AI's answer
        result = chain({'content': content})
        
        print(result['text'])