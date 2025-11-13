
from langchain_core.prompts import ChatPromptTemplate

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


PROMPT_REGISTRY = {"document_analysis":document_analysis_prompt, "document_comparison":document_comparison_prompt}