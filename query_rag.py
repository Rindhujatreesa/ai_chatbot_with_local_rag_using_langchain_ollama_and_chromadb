import argparse
from langchain_chroma.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_ollama.llms import OllamaLLM

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chromadb"

sys_instructions = SystemMessagePromptTemplate.from_template("You are a middle school math teacher. Your task is to respond to 6th graders' doubts from their curriculum. You should have an encouraging and polite attitude to the students."  
"You should use the data in the vector database to find the most relevant portions to explain them the concepts."
"Before starting the response, translate the students' question into a title. Then, solve the {question} and format it into step-by-step explanations abiding to pedagogical standards for a 6th grader. Include all steps to reach the answer without skipping any. Use standard approaches."
"After the explanation for the question, you should give two different examples of different difficulty levels. The first example should be a direct and easy one, and the second should invoke some reasoning in the student."
"Finally, provide a third example without solution that covers all the topics from the explanation for the student to solve. Provide the answer as a hint. Ensure all the mathematical equations are LaTeX-formatted throughout the response.")

rag_context = HumanMessagePromptTemplate.from_template("Answer the question based on the following context: {context} Question: {question}")

# PROMPT_TEMPLATE = """
# Answer the question based only on the following context:

# {context} You are a middle school math teacher. Your task is to respond to 6th graders' doubts from their curriculum. You should have an encouraging and polite attitude to the students.  
# You should use the documents in the database to find the most relevant portions to explain them the concepts. 
# Before starting the response, translate the students' question into a title and provide explanation from the beginning. Format the responses into step-by-step explanations.
# You should give two different examples of different difficulty levels. The first example should be a direct and easy one, and the second should invoke some reasoning in the student.
# Finally, provide a question that covers all the topics from the explanation for the student to solve. Provide the answer as a hint. Ensure all the mathematical equations are LaTeX-formatted.

# ---

# Answer the question based on the above context: {question}
# """

chat_prompt = ChatPromptTemplate.from_messages([sys_instructions, rag_context])

def main():
    # The parser helps to provide the prompt along with the python call in CLI. -- python3 query_rag.py "prompt"
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args() 
    query_text = args.query_text
    # query_text= input("User: ") # Use this line instead of parsing the prompt
    response = query_rag(query_text)
    print("Response: ", response)

def query_rag(query_text: str):
    # Call the function to embed the documents and store it in a vector database, here, ChromaDB
    embedding_function = get_embedding_function()
    db = Chroma(collection_name="chunks", persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB to find the chunks that are the most similar with the context
    results = db.similarity_search_with_score(query_text, k=5)
    # print(results)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    ## The below code snippet can be used if you do not prefer to set system instructions
    # prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    # prompt = prompt_template.format(context=context_text, question=query_text)
    
    # This code allows to add system instructions along with the rag context and prompt as the query
    prompt = chat_prompt.format(context=context_text, question=query_text)
    # print(prompt)

    ## Once the prompt and context are combined and ready, we invoke the LLM to perform the text generation
    # model = OllamaLLM(model="mistral")
    model = OllamaLLM(model="gemma3:1b") #much lighter and faster model
    response_text = model.invoke(prompt)
    
    ## Use the below code if you we wish to add source for the generated text
    # sources = [doc.metadata.get("id", None) for doc, _score in results]
    # formatted_response = f"Response: {response_text}\nSources: {sources}"
    # formatted_response = f"Response: {response_text}"
    # print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()