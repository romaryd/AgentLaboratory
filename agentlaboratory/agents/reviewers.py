import json
import re
from agentlaboratory.inference import query_model


def extract_json_between_markers(llm_output):
    # Regular expression pattern to find JSON content between ```json and ```
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)

    if not matches:
        # Fallback: Try to find any JSON-like content in the output
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError:
            # Attempt to fix common JSON issues
            try:
                # Remove invalid control characters
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                parsed_json = json.loads(json_string_clean)
                return parsed_json
            except json.JSONDecodeError:
                continue  # Try next match

    return None  # No valid JSON found


def get_score(
    outlined_plan,
    latex,
    reward_model_llm,
    reviewer_type=None,
    attempts=3,
    openai_api_key=None,
):
    e = str()
    for _attempt in range(attempts):
        try:
            # todo: have a reward function here
            # template inherited from the AI Scientist (good work on this prompt Sakana AI team :D)
            template_instructions = """
            Respond in the following format:

            THOUGHT:
            <THOUGHT>

            REVIEW JSON:
            ```json
            <JSON>
            ```

            In <THOUGHT>, first briefly discuss your intuitions and reasoning for the evaluation.
            Detail your high-level arguments, necessary choices and desired outcomes of the review.
            Do not make generic comments here, but be specific to your current paper.
            Treat this as the note-taking phase of your review.

            In <JSON>, provide the review in JSON format with the following fields in the order:
            - "Summary": A summary of the paper content and its contributions.
            - "Strengths": A list of strengths of the paper.
            - "Weaknesses": A list of weaknesses of the paper.
            - "Originality": A rating from 1 to 4 (low, medium, high, very high).
            - "Quality": A rating from 1 to 4 (low, medium, high, very high).
            - "Clarity": A rating from 1 to 4 (low, medium, high, very high).
            - "Significance": A rating from 1 to 4 (low, medium, high, very high).
            - "Questions": A set of clarifying questions to be answered by the paper authors.
            - "Limitations": A set of limitations and potential negative societal impacts of the work.
            - "Ethical Concerns": A boolean value indicating whether there are ethical concerns.
            - "Soundness": A rating from 1 to 4 (poor, fair, good, excellent).
            - "Presentation": A rating from 1 to 4 (poor, fair, good, excellent).
            - "Contribution": A rating from 1 to 4 (poor, fair, good, excellent).
            - "Overall": A rating from 1 to 10 (very strong reject to award quality).
            - "Confidence": A rating from 1 to 5 (low, medium, high, very high, absolute).
            - "Decision": A decision that has to be one of the following: Accept, Reject.

            For the "Decision" field, don't use Weak Accept, Borderline Accept, Borderline Reject, or Strong Reject. Instead, only use Accept or Reject.
            This JSON will be automatically parsed, so ensure the format is precise.
            """
            neurips_form = (
                """
                ## Review Form
                Below is a description of the questions you will be asked on the review form for each paper and some guidelines on what to consider when answering these questions.
                When writing your review, please keep in mind that after decisions have been made, reviews and meta-reviews of accepted papers and opted-in rejected papers will be made public. 

                1. Summary: Briefly summarize the paper and its contributions. This is not the place to critique the paper; the authors should generally agree with a well-written summary.
                  - Strengths and Weaknesses: Please provide a thorough assessment of the strengths and weaknesses of the paper, touching on each of the following dimensions:
                  - Originality: Are the tasks or methods new? Is the work a novel combination of well-known techniques? (This can be valuable!) Is it clear how this work differs from previous contributions? Is related work adequately cited
                  - Quality: Is the submission technically sound? Are claims well supported (e.g., by theoretical analysis or experimental results)? Are the methods used appropriate? Is this a complete piece of work or work in progress? Are the authors careful and honest about evaluating both the strengths and weaknesses of their work
                  - Clarity: Is the submission clearly written? Is it well organized? (If not, please make constructive suggestions for improving its clarity.) Does it adequately inform the reader? (Note that a superbly written paper provides enough information for an expert reader to reproduce its results.)
                  - Significance: Are the results important? Are others (researchers or practitioners) likely to use the ideas or build on them? Does the submission address a difficult task in a better way than previous work? Does it advance the state of the art in a demonstrable way? Does it provide unique data, unique conclusions about existing data, or a unique theoretical or experimental approach?

                2. Questions: Please list up and carefully describe any questions and suggestions for the authors. Think of the things where a response from the author can change your opinion, clarify a confusion or address a limitation. This can be very important for a productive rebuttal and discussion phase with the authors.  

                3. Limitations: Have the authors adequately addressed the limitations and potential negative societal impact of their work? If not, please include constructive suggestions for improvement.
                In general, authors should be rewarded rather than punished for being up front about the limitations of their work and any potential negative societal impact. You are encouraged to think through whether any critical points are missing and provide these as feedback for the authors.

                4. Ethical concerns: If there are ethical issues with this paper, please flag the paper for an ethics review. For guidance on when this is appropriate, please review the NeurIPS ethics guidelines.

                5. Soundness: Please assign the paper a numerical rating on the following scale to indicate the soundness of the technical claims, experimental and research methodology and on whether the central claims of the paper are adequately supported with evidence.
                  4: excellent
                  3: good
                  2: fair
                  1: poor

                6. Presentation: Please assign the paper a numerical rating on the following scale to indicate the quality of the presentation. This should take into account the writing style and clarity, as well as contextualization relative to prior work.
                  4: excellent
                  3: good
                  2: fair
                  1: poor

                7. Contribution: Please assign the paper a numerical rating on the following scale to indicate the quality of the overall contribution this paper makes to the research area being studied. Are the questions being asked important? Does the paper bring a significant originality of ideas and/or execution? Are the results valuable to share with the broader NeurIPS community.
                  4: excellent
                  3: good
                  2: fair
                  1: poor

                8. Overall: Please provide an "overall score" for this submission. Choices: 
                  10: Award quality: Technically flawless paper with groundbreaking impact on one or more areas of AI, with exceptionally strong evaluation, reproducibility, and resources, and no unaddressed ethical considerations.
                  9: Very Strong Accept: Technically flawless paper with groundbreaking impact on at least one area of AI and excellent impact on multiple areas of AI, with flawless evaluation, resources, and reproducibility, and no unaddressed ethical considerations.
                  8: Strong Accept: Technically strong paper with, with novel ideas, excellent impact on at least one area of AI or high-to-excellent impact on multiple areas of AI, with excellent evaluation, resources, and reproducibility, and no unaddressed ethical considerations.
                  7: Accept: Technically solid paper, with high impact on at least one sub-area of AI or moderate-to-high impact on more than one area of AI, with good-to-excellent evaluation, resources, reproducibility, and no unaddressed ethical considerations.
                  6: Weak Accept: Technically solid, moderate-to-high impact paper, with no major concerns with respect to evaluation, resources, reproducibility, ethical considerations.
                  5: Borderline accept: Technically solid paper where reasons to accept outweigh reasons to reject, e.g., limited evaluation. Please use sparingly.
                  4: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.
                  3: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility and incompletely addressed ethical considerations.
                  2: Strong Reject: For instance, a paper with major technical flaws, and/or poor evaluation, limited impact, poor reproducibility and mostly unaddressed ethical considerations.
                  1: Very Strong Reject: For instance, a paper with trivial results or unaddressed ethical considerations

                9. Confidence:  Please provide a "confidence score" for your assessment of this submission to indicate how confident you are in your evaluation. Choices:
                  5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.
                  4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
                  3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
                  2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
                  1: Your assessment is an educated guess. The submission is not in your area or the submission was difficult to understand. Math/other details were not carefully checked.

                  You must make sure that all sections are properly created: abstract, introduction, methods, results, and discussion. Points must be reduced from your scores if any of these are missing.
                """
                + template_instructions
            )
            if reviewer_type is None:
                reviewer_type = ""
            sys = (
                "You are an AI researcher who is reviewing a paper that was submitted to a prestigious ML venue. "
                f"Be critical and cautious in your decision. {reviewer_type}\n"
            ) + neurips_form
            scoring = query_model(
                model_str=f"{reward_model_llm}",
                system_prompt=sys,
                openai_api_key=openai_api_key,
                prompt=(
                    f"Outlined in the following text is the research plan that the machine learning engineer was tasked with building: {outlined_plan}\n\n"
                    f"The following text is the research latex that the model produced: \n{latex}\n\n"
                ),
                temp=0.0,
            )
            review_json = extract_json_between_markers(scoring)

            overall = int(review_json["Overall"]) / 10
            soundness = int(review_json["Soundness"]) / 4
            confidence = int(review_json["Confidence"]) / 5
            contribution = int(review_json["Contribution"]) / 4
            presentation = int(review_json["Presentation"]) / 4
            clarity = int(review_json["Clarity"]) / 4
            originality = int(review_json["Originality"]) / 4
            quality = int(review_json["Quality"]) / 4
            significance = int(review_json["Significance"]) / 4

            clarity_weight = 0.1
            quality_weight = 0.1
            overall_weight = 1.0
            soundness_weight = 0.1
            confidence_weight = 0.1
            originality_weight = 0.1
            significance_weight = 0.1
            contribution_weight = 0.4
            presentation_weight = 0.2

            # max possible
            max_score = (
                clarity_weight
                + quality_weight
                + overall_weight
                + soundness_weight
                + confidence_weight
                + originality_weight
                + significance_weight
                + contribution_weight
                + presentation_weight
            )

            performance = (
                (
                    soundness_weight * soundness
                    + presentation_weight * presentation
                    + confidence_weight * confidence
                    + contribution_weight * contribution
                    + overall_weight * overall
                    + originality_weight * originality
                    + significance * significance_weight
                    + clarity_weight * clarity
                    + quality_weight * quality
                )
                / max_score
            ) * 10
            return (
                performance,
                f"The performance of your submission is: {performance}" + scoring,
                True,
            )
        except Exception as e:
            print(e)
            return None, str(e), False
    return 0, e


class ReviewersAgent:
    def __init__(self, model="gpt-4o-mini", notes=None, openai_api_key=None):
        if notes is None:
            self.notes = []
        else:
            self.notes = notes
            self.model = model
            self.openai_api_key = openai_api_key

    def inference(self, plan, report):
        reviewer_1 = "You are a harsh but fair reviewer and expect good experiments that lead to insights for the research topic."
        review_1 = get_score(
            outlined_plan=plan,
            latex=report,
            reward_model_llm=self.model,
            reviewer_type=reviewer_1,
            openai_api_key=self.openai_api_key,
        )

        reviewer_2 = "You are a harsh and critical but fair fair reviewer who is looking for idea that would be impactful in the field."
        review_2 = get_score(
            outlined_plan=plan,
            latex=report,
            reward_model_llm=self.model,
            reviewer_type=reviewer_2,
            openai_api_key=self.openai_api_key,
        )

        reviewer_3 = "You are a harsh but fair open-minded reviewer that is looking for novel ideas that have not been proposed before."
        review_3 = get_score(
            outlined_plan=plan,
            latex=report,
            reward_model_llm=self.model,
            reviewer_type=reviewer_3,
            openai_api_key=self.openai_api_key,
        )

        return f"Reviewer #1:\n{review_1}, \nReviewer #2:\n{review_2}, \nReviewer #3:\n{review_3}"
