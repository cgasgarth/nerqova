"""Fixed request used for matched model and complete-decision timings."""

from kev.model import user_tokens


def workload(tok, state_tokens, questions):
    sentence = "Customer ordered a blue shirt on Monday, paid for express delivery, and received a red shirt on Friday. "
    state = ""
    while len(user_tokens(tok, state)) < state_tokens:
        state += sentence
    return {
        "state": state,
        "questions": [
            {
                "instr": f"Which action best fits customer request {i}?",
                "options": ["Refund the item", "Replace it with the blue shirt", "Ask for more information"],
                "label": 1,
            }
            for i in range(questions)
        ],
    }
