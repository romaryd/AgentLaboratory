
import io
import sys
import traceback
import concurrent.futures
import matplotlib

def execute_code(code_str, timeout=60, MAX_LEN=1000):
    # prevent plotting errors
    matplotlib.use('Agg')  # Use the non-interactive Agg backend
    import matplotlib.pyplot as plt

    # Preventing execution of certain resource-intensive datasets
    if "load_dataset('pubmed" in code_str:
        return "[CODE EXECUTION ERROR] pubmed Download took way too long. Program terminated"
    if "exit(" in code_str:
        return "[CODE EXECUTION ERROR] The exit() command is not allowed you must remove this."

    # Capturing the output
    output_capture = io.StringIO()
    sys.stdout = output_capture

    # Create a new global context for exec
    exec_globals = globals()

    def run_code():
        try:
            # Executing the code in the global namespace
            exec(code_str, exec_globals)
        except Exception as e:
            output_capture.write(f"[CODE EXECUTION ERROR]: {str(e)}\n")
            traceback.print_exc(file=output_capture)

    try:
        # Running code in a separate thread with a timeout
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_code)
            future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        return f"[CODE EXECUTION ERROR]: Code execution exceeded the timeout limit of {timeout} seconds. You must reduce the time complexity of your code."
    except Exception as e:
        return f"[CODE EXECUTION ERROR]: {str(e)}"
    finally:
        # Restoring standard output
        sys.stdout = sys.__stdout__

    # Returning the captured output
    return output_capture.getvalue()[:MAX_LEN]