import re
def extract_code_blocks(text):
    """
    Function to extract content inside markdown code blocks from a given text.

    Args:
        text (str): The text from which to extract code blocks.

    Returns:
        list: A list of strings, each representing a code block.
    """
    pattern = r"```(.*?)```"
    code_blocks = re.findall(pattern, text, re.DOTALL)
    # remove trailing newline characters
    code_blocks = [block.rstrip() for block in code_blocks]
    return code_blocks