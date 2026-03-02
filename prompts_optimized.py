# PROMPTS FOR MODEL TO GENERATE EXAMPLES OF A FORMAT
GENERATE_EXAMPLE_PROMPT = '''
    You are a helpful assistant that generates high-quality examples of specific concepts.  
    Your goal is to produce an example that shows a deep and nuanced understanding of the concept.  
    The example should:  
    1. Be original and creative.  
    2. Clearly demonstrate the core principles of the concept in action.  
    3. Think internally as needed, but do not reveal hidden reasoning. Your output MUST begin with exactly: 
    FINAL ANSWER: <your generated example here>
    Do not include anything else.
'''

CLASSIFICATION_PROMPT='''
    You are a helpful assistant for classification tasks. Your job is to decide if the given response 
    is correct (Class 1) or incorrect (Class -1) with respect to the concept.

    Instructions:
    1. Think internally as needed, but do not reveal hidden reasoning.
    2. Your output MUST begin with exactly:
    FINAL ANSWER: <1 or -1>
    3. After that, provide a short explanation in this format:

    Class 1 (CORRECT): <explain why the response accurately reflects the concept>
    OR
    Class -1 (INCORRECT): <explain why the response misrepresents or contradicts the concept>
    Do not include anything else.
'''

DEFINE_CONCEPT_PROMPT = '''
    You are a helpful assistant. Your task is to provide an accurate and concise definition of the given concept.

    Instructions:
    1. You may think internally as much as needed, but do not reveal hidden reasoning.  
    2. Your output MUST begin with exactly this prefix (no extra characters, no extra lines):  
    FINAL ANSWER: <your final answer here>  
    3. Output only the definition. Do not include explanations, examples, or anything else.

    '''

DEFINE_CONCEPT_INCORRECTLY_PROMPT ='''
    You are a helpful assistant. Your task is to provide a wrong definition of the given concept.

    Instructions:
    1. You may think internally as much as needed, but do not reveal hidden reasoning.  
    2. Your output MUST begin with exactly this prefix (no extra characters, no extra lines):  
    FINAL ANSWER: <your final answer here>  
    3. Output only the definition. Do not include explanations, examples, or anything else.
'''
# judge_prompt = '''Your goal is to determine whether the response to a question demonstrates true understanding of a specific concept. A response shows true understanding if it applies the concept correctly and in a way that goes beyond surface-level knowledge or memorization. It should reflect a deep comprehension of the concept's principles and be able to use it in novel or complex situations.'''

GENERAL_PROMPT='''
    You are a helpful assistant. Provide a correct, concise answer. You can think as much as you like but
    begin your final answer with EXACTLY this prefix (no extra characters):  
    FINAL ANSWER: <your final answer here>     
'''



# PROMPT FOR JUDGE LLM
DEFINE_CONCEPT_JUDGE_PROMPT = '''
You are an impartial judge. Evaluate whether the definition of the concept is factually correct.

Evaluation Rules:
- Class 1 (CORRECT): The definition is consistent with the concept. Extra details are allowed if they do not contradict the definition.
- Class -1 (INCORRECT): The definition contradicts, misrepresents, or is inconsistent with the concept.

Guidelines:
- Base your judgment ONLY on the provided concept and definition. Do not rely on external knowledge.
- Keep reasoning concise (2-3 lines).
- Always follow the exact answer format below.

Answer Format (no deviations):
FINAL ANSWER: <1 or -1>  
Explanation: <2-3 line justification>

'''


GENERATE_CONCEPT_JUDGE_PROMPT= '''
You are an impartial judge. Evaluate whether the LLM's response (example) is factually correct based only on the given concept and your own understanding of it.

Evaluation Rules:
- Class 1 (CORRECT): The example is consistent with the concept and represents it accurately.  
- Class -1 (INCORRECT): The example contradicts, misrepresents, or is inconsistent with the concept.  

Guidelines:
- Use only the provided concept, the response, and your understanding of the concept. Do not rely on external knowledge.  
- Keep reasoning concise (2-3 lines).  
- Follow the exact answer format strictly.  

Answer Format (no deviations):  
FINAL ANSWER: <1 or -1>  
Explanation: <2-3 line justification>
'''

EDIT_CONCEPT_JUDGE_PROMPT = '''
You are an impartial judge. Evaluate whether the LLM's response (example) is correct based only on the given concept and your own understanding of it.

Evaluation Rules:
- Class 1 (CORRECT): The example is consistent with the concept and represents it accurately.  
- Class -1 (INCORRECT): The example contradicts, misrepresents, or is inconsistent with the concept.  

Guidelines:
- Use only the provided concept, the response, and your understanding of the concept. Do not rely on external knowledge.  
- Keep reasoning concise (2-3 lines).  
- Follow the exact answer format strictly.  

Answer Format (no deviations):  
FINAL ANSWER: <1 or -1>  
Explanation: <2-3 line justification>
'''


EDIT_EXAMPLE_PROMPT = '''
    You are a helpful assistant that edits examples so they accurately reflect a given concept while changing as little as possible.
    
    Instructions:
    1. Make precise edits that correct inaccuracies or misrepresentations of the concept while preserving the original structure, context, tone, and intent.
    2. Keep changes minimal: prefer small wording fixes, clarifications, and corrections rather than wholesale rewrites.
    3. Ensure the edited example clearly demonstrates the core principles of the concept in action.

    You can think as much as you like but begin your final answer with EXACTLY this prefix (no extra characters):  
    FINAL ANSWER: <your final answer here>     

    Do not include anything else.
'''


