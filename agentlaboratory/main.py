import os
import argparse
import pickle
from .laboratory_workflow import LaboratoryWorkflow


def parse_arguments():
    parser = argparse.ArgumentParser(description="AgentLaboratory Research Workflow")

    parser.add_argument(
        "--copilot-mode",
        type=str,
        default="False",
        help="Enable human interaction mode.",
    )

    parser.add_argument(
        "--deepseek-api-key", type=str, help="Provide the DeepSeek API key."
    )

    parser.add_argument(
        "--load-existing",
        type=str,
        default="False",
        help="Do not load existing state; start a new workflow.",
    )

    parser.add_argument(
        "--load-existing-path",
        type=str,
        help="Path to load existing state; start a new workflow, e.g. state_saves/results_interpretation.pkl",
    )

    parser.add_argument(
        "--research-topic", type=str, help="Specify the research topic."
    )

    parser.add_argument("--api-key", type=str, help="Provide the OpenAI API key.")

    parser.add_argument(
        "--compile-latex",
        type=str,
        default="True",
        help="Compile latex into pdf during paper writing phase. Disable if you can not install pdflatex.",
    )

    parser.add_argument(
        "--llm-backend",
        type=str,
        default="o1-mini",
        help="Backend LLM to use for agents in Agent Laboratory.",
    )

    parser.add_argument(
        "--language",
        type=str,
        default="English",
        help="Language to operate Agent Laboratory in.",
    )

    parser.add_argument(
        "--num-papers-lit-review",
        type=str,
        default="5",
        help="Total number of papers to summarize in literature review stage",
    )

    parser.add_argument(
        "--mlesolver-max-steps",
        type=str,
        default="3",
        help="Total number of mle-solver steps",
    )

    parser.add_argument(
        "--papersolver-max-steps",
        type=str,
        default="5",
        help="Total number of paper-solver steps",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    llm_backend = args.llm_backend
    human_mode = args.copilot_mode.lower() == "true"
    compile_pdf = args.compile_latex.lower() == "true"
    load_existing = args.load_existing.lower() == "true"
    try:
        num_papers_lit_review = int(args.num_papers_lit_review.lower())
    except Exception:
        raise Exception("args.num_papers_lit_review must be a valid integer!")
    try:
        papersolver_max_steps = int(args.papersolver_max_steps.lower())
    except Exception:
        raise Exception("args.papersolver_max_steps must be a valid integer!")
    try:
        mlesolver_max_steps = int(args.mlesolver_max_steps.lower())
    except Exception:
        raise Exception("args.papersolver_max_steps must be a valid integer!")

    api_key = os.getenv("OPENAI_API_KEY") or args.api_key
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY") or args.deepseek_api_key
    if args.api_key is not None and os.getenv("OPENAI_API_KEY") is None:
        os.environ["OPENAI_API_KEY"] = args.api_key
    if args.deepseek_api_key is not None and os.getenv("DEEPSEEK_API_KEY") is None:
        os.environ["DEEPSEEK_API_KEY"] = args.deepseek_api_key

    if not api_key and not deepseek_api_key:
        raise ValueError(
            "API key must be provided via --api-key / -deepseek-api-key or the OPENAI_API_KEY / DEEPSEEK_API_KEY environment variable."
        )

    ##########################################################
    # Research question that the agents are going to explore #
    ##########################################################
    if human_mode or args.research_topic is None:
        research_topic = input(
            "Please name an experiment idea for AgentLaboratory to perform: "
        )
    else:
        research_topic = args.research_topic

    task_notes_LLM = [
        {
            "phases": ["plan formulation"],
            "note": f"You should come up with a plan for TWO experiments.",
        },
        {
            "phases": ["plan formulation", "data preparation", "running experiments"],
            "note": "Please use gpt-4o-mini for your experiments.",
        },
        {
            "phases": ["running experiments"],
            "note": f'Use the following code to inference gpt-4o-mini: \nfrom openai import OpenAI\nos.environ["OPENAI_API_KEY"] = "{api_key}"\nclient = OpenAI()\ncompletion = client.chat.completions.create(\nmodel="gpt-4o-mini-2024-07-18", messages=messages)\nanswer = completion.choices[0].message.content\n',
        },
        {
            "phases": ["running experiments"],
            "note": f"You have access to only gpt-4o-mini using the OpenAI API, please use the following key {api_key} but do not use too many inferences. Do not use openai.ChatCompletion.create or any openai==0.28 commands. Instead use the provided inference code.",
        },
        {
            "phases": ["running experiments"],
            "note": "I would recommend using a small dataset (approximately only 100 data points) to run experiments in order to save time. Do not use much more than this unless you have to or are running the final tests.",
        },
        {
            "phases": ["data preparation", "running experiments"],
            "note": "You are running on a MacBook laptop. You can use 'mps' with PyTorch",
        },
        {
            "phases": ["data preparation", "running experiments"],
            "note": "Generate figures with very colorful and artistic design.",
        },
    ]

    task_notes_LLM.append(
        {
            "phases": [
                "literature review",
                "plan formulation",
                "data preparation",
                "running experiments",
                "results interpretation",
                "report writing",
                "report refinement",
            ],
            "note": f"You should always write in the following language to converse and to write the report {args.language}",
        },
    )

    ####################################################
    ###  Stages where human input will be requested  ###
    ####################################################
    human_in_loop = {
        "literature review": human_mode,
        "plan formulation": human_mode,
        "data preparation": human_mode,
        "running experiments": human_mode,
        "results interpretation": human_mode,
        "report writing": human_mode,
        "report refinement": human_mode,
    }

    ###################################################
    ###  LLM Backend used for the different phases  ###
    ###################################################
    agent_models = {
        "literature review": llm_backend,
        "plan formulation": llm_backend,
        "data preparation": llm_backend,
        "running experiments": llm_backend,
        "report writing": llm_backend,
        "results interpretation": llm_backend,
        "paper refinement": llm_backend,
    }

    if load_existing:
        load_path = args.load_existing_path
        if load_path is None:
            raise ValueError("Please provide path to load existing state.")
        with open(load_path, "rb") as f:
            lab = pickle.load(f)
    else:
        lab = LaboratoryWorkflow(
            research_topic=research_topic,
            notes=task_notes_LLM,
            agent_model_backbone=agent_models,
            human_in_loop_flag=human_in_loop,
            openai_api_key=api_key,
            compile_pdf=compile_pdf,
            num_papers_lit_review=num_papers_lit_review,
            papersolver_max_steps=papersolver_max_steps,
            mlesolver_max_steps=mlesolver_max_steps,
        )

    lab.perform_research()


if __name__ == "__main__":
    main()
