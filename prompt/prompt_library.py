
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

document_analysis_prompt = ChatPromptTemplate.from_template("""
You are a highly capable assistant trained to analyze and summarize documents.
Return only valid JSON matching the exact schema below.

{format_instructions}

Analyze this document:
{document_text}
""")

document_comparison_prompt = ChatPromptTemplate.from_template("""
You will be provided with content from two documents. Your tasks are as follows:

1. Compare the content in two documents
2. Identify the difference in documents and note down the page numbers.
3. The output you provide must be page wise comparison content
4. If any page do not have any change. Mention as "No Change"

Input documents:

{combined_docs}

Your response should follow this format:

{format_instruction}
""")

#prompt for contextual question rewriting
contextual_question_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Given a conversation history and the most recent user query, rewrite user query as a standalone question. "
        "that makes sense without relying on the previous context. Do not provide an answer-only reformulate the"
        "question if necessary; otherwise, return it unchanged."
    )),
    MessagesPlaceholder("chat_history"), 
    ("human", "{input}")
    ])

#prompt for answering based on context
context_qa_prompt = ChatPromptTemplate.from_messages([ 
    ("system", (
        "You are an assistant designed to answer questions using the provided context. Rely only on the retrieved "
        "information to form your response. If the answer is not found in the context, response with 'I don't know.' "
        "keep your answer concise and no longer than three sentences. \n\n{context}"
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    ])


PROMPT_REGISTRY = {
    "document_analysis":document_analysis_prompt, 
    "document_comparison":document_comparison_prompt,
    "contextualize_question":contextual_question_prompt,
    "context_qa":context_qa_prompt
    }