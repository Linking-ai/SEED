# 0-shot
gsm8k_thought_prompt = '''Given a [question]: ##{initial_prompt}##, the previous sub-question and sub-answer is: ##{state_text}##
Please output the next sub-question to further reason the [question].'''

gsm8k_answer_prompt = '''Given a [question]: ##{initial_prompt}##, the sub-question is: ##{state_text}##
Please answer the sub-question based on the [question].'''

gsm8k_evaluate_prompt = '''Given a [question]: ##{initial_prompt}##, the sub-question is: ##{state_text}##, the answer is: ##{answer}##
Please output a number between 0 and 10 to evaluate the answer. The higher number represent more helpful for answer [question].
'''

gsm8k_solution_prompt = '''Given a [question]: ##{initial_prompt}##, the previous sub-question and sub-answer is: ##{state_text}##
Please output the answer of the [question].'''