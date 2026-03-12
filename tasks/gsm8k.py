"""
GSM8K evaluation.
https://huggingface.co/datasets/openai/gsm8k

Example problem instance:

Question:
Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer:
Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
#### 10

Notice that GSM8K uses tool calls inside << >> tags.
"""

import re
from datasets import load_dataset
from tasks.common import Task


GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
def extract_answer(completion):
    """
    Extract the numerical answer after #### marker.
    Follows official code for normalization:
    https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/dataset.py#L28
    """
    match = GSM_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    return None


class GSM8K(Task):

    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["main", "socratic"], "GSM8K subset must be main|socratic"
        assert split in ["train", "test"], "GSM8K split must be train|test"
        self.ds = load_dataset("openai/gsm8k", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        """ Get a single problem from the dataset. """
        row = self.ds[index]
        question = row['question'] # string of the question prompt
        answer = row['answer'] # string of the full solution and the answer after #### marker
        # Create and return the Conversation object
        # This is tricky because GSM8K uses tool calls, which we need to parse here.
        assistant_message_parts = []
        parts = re.split(r'(<<[^>]+>>)', answer)
        for part in parts:
            if part.startswith('<<') and part.endswith('>>'):
                # This is a calculator tool call
                inner = part[2:-2]  # Remove << >>
                # Split on = to get expression and result
                if '=' in inner:
                    expr, result = inner.rsplit('=', 1)
                else:
                    expr, result = inner, ""
                # Add the tool call as a part
                assistant_message_parts.append({"type": "python", "text": expr})
                # Add the result as a part
                assistant_message_parts.append({"type": "python_output", "text": result})
            else:
                # Regular text in between tool calls
                assistant_message_parts.append({"type": "text", "text": part})
        # Now put it all together
        messages = [
            {"role": "user", "content": question}, # note: simple string
            {"role": "assistant", "content": assistant_message_parts}, # note: list of parts (as dicts)
        ]
        conversation = {
            "messages": messages,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        """
        Given (conversation, completion), return evaluation outcome (0 = wrong, 1 = correct)
        Note that:
        - the conversation has both user AND assistant message (containing the ground truth answer)
        - the assistant_response is usually the alternative assistant message achieved via sampling

        TODO: Technically, assistant_response should be a Message (either a string or a list of parts)
              We can handle this later possibly. For now just assume string.
        """
        assert isinstance(assistant_response, str), "Assuming simple string response for now"
        # First extract the ground truth answer
        assistant_message = conversation['messages'][-1]
        assert assistant_message['role'] == "assistant", "Last message must be from the Assistant"
        assert isinstance(assistant_message['content'], list), "This is expected to be a list of parts"
        last_text_part = assistant_message['content'][-1]['text'] # this contains the final answer in GSM8K
        # Extract both the ground truth answer and the predicted answer
        ref_num = extract_answer(last_text_part)
        pred_num = extract_answer(assistant_response)
        # Compare and return the success as int
        is_correct = int(pred_num == ref_num)
        return is_correct

    def reward(self, conversation, assistant_response):
        """
        Used during RL. To keep things simple, just re-use the evaluation above.
        Later this could be made more complex (e.g. format matching etc.)
        """
        is_correct = self.evaluate(conversation, assistant_response)
        is_correct_float = float(is_correct)
        return is_correct_float

UNITS = {
    # Time
    "s": "second",
    "sec": "second",
    "secs": "second",
    "second": "second",
    "seconds": "second",

    "min": "minute",
    "mins": "minute",
    "minute": "minute",
    "minutes": "minute",

    "h": "hour",
    "hr": "hour",
    "hrs": "hour",
    "hour": "hour",
    "hours": "hour",

    "d": "day",
    "day": "day",
    "days": "day",

    "wk": "week",
    "wks": "week",
    "week": "week",
    "weeks": "week",

    "mo": "month",
    "mos": "month",
    "month": "month",
    "months": "month",

    "yr": "year",
    "yrs": "year",
    "year": "year",
    "years": "year",

    # Length
    "mm": "millimetre",
    "millimeter": "millimetre",
    "millimeters": "millimetre",
    "millimetre": "millimetre",
    "millimetres": "millimetre",

    "cm": "centimetre",
    "centimeter": "centimetre",
    "centimeters": "centimetre",
    "centimetre": "centimetre",
    "centimetres": "centimetre",

    "meter": "metre",
    "meters": "metre",
    "metre": "metre",
    "metres": "metre",

    "km": "kilometre",
    "kilometer": "kilometre",
    "kilometers": "kilometre",
    "kilometre": "kilometre",
    "kilometres": "kilometre",

    "in": "inch",
    "inch": "inch",
    "inches": "inch",

    "ft": "foot",
    "foot": "foot",
    "feet": "foot",

    "yd": "yard",
    "yard": "yard",
    "yards": "yard",

    "mi": "mile",
    "mile": "mile",
    "miles": "mile",

    # Mass
    "mg": "milligram",
    "milligram": "milligram",
    "milligrams": "milligram",
    "milligramme": "milligram",
    "milligrammes": "milligram",

    "g": "gram",
    "gram": "gram",
    "grams": "gram",
    "gramme": "gram",
    "grammes": "gram",

    "kg": "kilogram",
    "kilogram": "kilogram",
    "kilograms": "kilogram",
    "kilogramme": "kilogram",
    "kilogrammes": "kilogram",

    "oz": "ounce",
    "ounce": "ounce",
    "ounces": "ounce",

    "lb": "pound",
    "lbs": "pound",
    "pound": "pound",
    "pounds": "pound",

    "ton": "tonne",
    "tons": "tonne",
    "tonne": "tonne",
    "tonnes": "tonne",

    # Volume
    "ml": "millilitre",
    "milliliter": "millilitre",
    "milliliters": "millilitre",
    "millilitre": "millilitre",
    "millilitres": "millilitre",

    "l": "litre",
    "L": "litre",
    "liter": "litre",
    "liters": "litre",
    "litre": "litre",
    "litres": "litre",

    "cup": "cup",
    "cups": "cup",

    "pt": "pint",
    "pint": "pint",
    "pints": "pint",

    "qt": "quart",
    "quart": "quart",
    "quarts": "quart",

    "gal": "gallon",
    "gallon": "gallon",
    "gallons": "gallon",

    # Temperature
    # Note we parse all as "degree", including the angular degree, because
    # we can't reliably differentiate.
    "degree": "degree",
    "degrees": "degree",
    "celsius": "degree",
    "f": "degree",
    "°f": "degree",
    "fahrenheit": "degree",

    # Speed
    "m/s": "metre per second",
    "meter per second": "metre per second",
    "meters per second": "metre per second",
    "metre per second": "metre per second",
    "metres per second": "metre per second",

    "km/h": "kilometre per hour",
    "kph": "kilometre per hour",
    "kilometer per hour": "kilometre per hour",
    "kilometers per hour": "kilometre per hour",
    "kilometre per hour": "kilometre per hour",
    "kilometres per hour": "kilometre per hour",

    "mph": "mile per hour",
    "mile per hour": "mile per hour",
    "miles per hour": "mile per hour",

    # Money
    "$": "dollar",
    "dollar": "dollar",
    "dollars": "dollar",

    "cent": "cent",
    "cents": "cent",
}
