import os, re
import shutil
import tiktoken
import subprocess
from typing import List, Dict, Optional, Union


def compile_latex(
    latex_code: str,
    compile: bool = True,
    output_filename: str = "output.pdf",
    timeout: int = 30,
) -> str:
    """Compiles LaTeX code into a PDF document.

    Args:
        latex_code (str): The LaTeX source code to compile.
        compile (bool, optional): Whether to actually compile the code. Defaults to True.
        output_filename (str, optional): Name of the output PDF file. Defaults to "output.pdf".
        timeout (int, optional): Maximum time in seconds for compilation. Defaults to 30.

    Returns:
        str: A message indicating compilation success or error details.
    """
    latex_code = latex_code.replace(
        r"\documentclass{article}",
        "\\documentclass{article}\n\\usepackage{amsmath}\n\\usepackage{amssymb}\n\\usepackage{array}\n\\usepackage{algorithm}\n\\usepackage{algorithmicx}\n\\usepackage{algpseudocode}\n\\usepackage{booktabs}\n\\usepackage{colortbl}\n\\usepackage{color}\n\\usepackage{enumitem}\n\\usepackage{fontawesome5}\n\\usepackage{float}\n\\usepackage{graphicx}\n\\usepackage{hyperref}\n\\usepackage{listings}\n\\usepackage{makecell}\n\\usepackage{multicol}\n\\usepackage{multirow}\n\\usepackage{pgffor}\n\\usepackage{pifont}\n\\usepackage{soul}\n\\usepackage{sidecap}\n\\usepackage{subcaption}\n\\usepackage{titletoc}\n\\usepackage[symbol]{footmisc}\n\\usepackage{url}\n\\usepackage{wrapfig}\n\\usepackage{xcolor}\n\\usepackage{xspace}",
    )
    # print(latex_code)
    dir_path = "research_dir/tex"
    tex_file_path = os.path.join(dir_path, "temp.tex")
    # Write the LaTeX code to the .tex file in the specified directory
    with open(tex_file_path, "w") as f:
        f.write(latex_code)

    if not compile:
        return f"Compilation successful"

    # Compiling the LaTeX code using pdflatex with non-interactive mode and timeout
    try:
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "temp.tex"],
            check=True,  # Raises a CalledProcessError on non-zero exit codes
            stdout=subprocess.PIPE,  # Capture standard output
            stderr=subprocess.PIPE,  # Capture standard error
            timeout=timeout,  # Timeout for the process
            cwd=dir_path,
        )

        # If compilation is successful, return the success message
        return f"Compilation successful: {result.stdout.decode('utf-8')}"

    except subprocess.TimeoutExpired:
        # If the compilation takes too long, return a timeout message
        return "[CODE EXECUTION ERROR]: Compilation timed out after {} seconds".format(
            timeout
        )
    except subprocess.CalledProcessError as e:
        # If there is an error during LaTeX compilation, return the error message
        return f"[CODE EXECUTION ERROR]: Compilation failed: {e.stderr.decode('utf-8')} {e.output.decode('utf-8')}. There was an error in your latex."


def count_tokens(messages: List[Dict[str, str]], model: str = "gpt-4") -> int:
    """Counts the number of tokens in a list of messages for a specific model.

    Args:
        messages (List[Dict[str, str]]): List of message dictionaries containing 'content' key.
        model (str, optional): The model to use for token counting. Defaults to "gpt-4".

    Returns:
        int: Total number of tokens in all messages.
    """
    enc = tiktoken.encoding_for_model(model)
    num_tokens = sum([len(enc.encode(message["content"])) for message in messages])
    return num_tokens


def remove_figures() -> None:
    """Removes all PNG files in the current directory that start with 'Figure_'.

    Searches the current working directory for files matching the pattern 'Figure_*.png'
    and deletes them.
    """
    for _file in os.listdir("."):
        if "Figure_" in _file and ".png" in _file:
            os.remove(_file)


def remove_directory(dir_path: str) -> None:
    """Removes a directory and all its contents if it exists.

    Args:
        dir_path (str): Path to the directory to be removed.
    """
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        try:
            shutil.rmtree(dir_path)
            print(f"Directory {dir_path} removed successfully.")
        except Exception as e:
            print(f"Error removing directory {dir_path}: {e}")
    else:
        print(f"Directory {dir_path} does not exist or is not a directory.")


def save_to_file(location: str, filename: str, data: str) -> None:
    """Saves text data to a file in the specified location.

    Args:
        location (str): Directory path where the file should be saved.
        filename (str): Name of the file to create.
        data (str): Text content to write to the file.
    """
    filepath = os.path.join(location, filename)
    try:
        with open(filepath, "w") as f:
            f.write(data)  # Write the raw string instead of using json.dump
        print(f"Data successfully saved to {filepath}")
    except Exception as e:
        print(f"Error saving file {filename}: {e}")


def clip_tokens(
    messages: List[Dict[str, str]], model: str = "gpt-4", max_tokens: int = 100000
) -> List[Dict[str, str]]:
    """Clips the token count of messages to stay within a maximum limit.

    Args:
        messages (List[Dict[str, str]]): List of message dictionaries to clip.
        model (str, optional): The model to use for token counting. Defaults to "gpt-4".
        max_tokens (int, optional): Maximum number of tokens to keep. Defaults to 100000.

    Returns:
        List[Dict[str, str]]: Clipped messages list that fits within the token limit.
    """
    enc = tiktoken.encoding_for_model(model)
    total_tokens = sum([len(enc.encode(message["content"])) for message in messages])

    if total_tokens <= max_tokens:
        return messages  # No need to clip if under the limit

    # Start removing tokens from the beginning
    tokenized_messages = []
    for message in messages:
        tokenized_content = enc.encode(message["content"])
        tokenized_messages.append(
            {"role": message["role"], "content": tokenized_content}
        )

    # Flatten all tokens
    all_tokens = [
        token for message in tokenized_messages for token in message["content"]
    ]

    # Remove tokens from the beginning
    clipped_tokens = all_tokens[total_tokens - max_tokens :]

    # Rebuild the clipped messages
    clipped_messages = []
    current_idx = 0
    for message in tokenized_messages:
        message_token_count = len(message["content"])
        if current_idx + message_token_count > len(clipped_tokens):
            clipped_message_content = clipped_tokens[current_idx:]
            clipped_message = enc.decode(clipped_message_content)
            clipped_messages.append(
                {"role": message["role"], "content": clipped_message}
            )
            break
        else:
            clipped_message_content = clipped_tokens[
                current_idx : current_idx + message_token_count
            ]
            clipped_message = enc.decode(clipped_message_content)
            clipped_messages.append(
                {"role": message["role"], "content": clipped_message}
            )
            current_idx += message_token_count
    return clipped_messages


def extract_prompt(text: str, word: str) -> str:
    """Extracts code blocks from text that are marked with a specific language identifier.

    Args:
        text (str): The text to search for code blocks.
        word (str): The language identifier to look for in code block markers.

    Returns:
        str: Concatenated string of all matching code blocks, stripped of leading/trailing whitespace.
    """
    code_block_pattern = rf"```{word}(.*?)```"
    code_blocks = re.findall(code_block_pattern, text, re.DOTALL)
    extracted_code = "\n".join(code_blocks).strip()
    return extracted_code
