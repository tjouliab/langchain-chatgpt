from langchain import prompts, chains
from llm import gemini
import argparse

# LLM and LangChain integration
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='return a list of numbers')
    parser.add_argument('--language', default='Python')
    args = parser.parse_args()

    # Init Model
    model = gemini.model

    # Create Prompts
    code_prompt = prompts.PromptTemplate(
        template='Write a very short {language} function that will {task}',
        input_variables=['language', 'task'],
    )

    test_prompt = prompts.PromptTemplate(
        template='Write a test for the following {language} code:\n{code}',
        input_variables=['language', 'code'],
    )

    # Init chains
    code_chain = chains.LLMChain(
        llm=model,
        prompt=code_prompt,
        output_key='code',
    )

    test_chain = chains.LLMChain(
        llm=model,
        prompt=test_prompt,
        output_key='test',
    )

    # Create the merged chain
    chain = chains.SequentialChain(
        chains=[code_chain, test_chain],
        input_variables=['language', 'task'],
        output_variables=['test', 'code'],
    )

    # Generate a response
    response = chain({
        'language': args.language,
        'task': args.task
    })

    print(response)
