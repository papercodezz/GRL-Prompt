def get_question_text(problem, option_inds):
    question = problem['question']
    context = problem['context']
    the_question=""
    if len(context) > 0:
        the_question = the_question = f"Question:\n{question}Context:\n{context}"
    else:
        the_question = f"Question:\n{question}"
    return the_question



def get_answer(problem):
    return problem['answer']


def get_solution_text(problem):
    # \\n: GPT-3 can generate the solution with more tokens
    solution = problem['solution'].replace("\n", "\\n")
    return solution


def create_one_example(format, table, question, answer, solution, test_example=True):

    input_format, output_format = format.split("-")  # e.g., "Q-A"

    elements = {
        "Q": f"{question}",
        "A": f"Answer:\n{answer}",
    }

    # Input
    input = "\n".join(elements[label] for label in input_format)

    # Output
    if test_example:
        output = "Answer:\n"
    else:
        output = elements[output_format]

    # Prompt text
    text = input + "\n" + output
    # text = text.replace("  ", " ").strip()

    return text


def build_prompt(problems, shot_pids, test_pid, args):

    examples = []
    pids = shot_pids + [test_pid]

    # n-shot training examples
    for pid in pids:
        problem = problems[pid]
        table = ""
        question = get_question_text(problem, args.option_inds)
        answer = get_answer(problem)
        solution = ""

        if pid == test_pid:
            assert pid not in shot_pids
            example = create_one_example(args.prompt_format, table, question, answer, solution, test_example=True)
        else:
            example = create_one_example(args.prompt_format, table, question, answer, solution, test_example=False)

        examples.append(example)

    # create the prompt input
    prompt_input = '\n'.join(examples)

    return "Please read the following text carefully, learn from the examples provided, and answer the last question."+prompt_input


def create_example_from_pid(pid, problems, args, test=False):
    problem = problems[pid]
    table = ""
    question = get_question_text(problem, args.option_inds)
    answer = get_answer(problem)
    solution = ""

    if test:
        example = create_one_example(args.prompt_format, table, question, answer, solution, test_example=True)
    else:
        example = create_one_example(args.prompt_format, table, question, answer, solution, test_example=False)

    return example
