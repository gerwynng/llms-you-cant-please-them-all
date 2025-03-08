# Libraries

[**Lingua**](https://github.com/pemistahl/lingua-py)

- Lingua makes use of n-grams of sizes 1 up to 5 which results in much more accurate prediction of the correct language.
- Lingua does not only use such a statistical model, but also a rule-based engine. This engine first determines the alphabet of the input text and searches for characters which are unique in one or more languages. If exactly one language can be reliably chosen this way, the statistical model is not necessary anymore. In any case, the rule-based engine filters out languages that do not satisfy the conditions of the input text. Only then, in a second step, the probabilistic n-gram model is taken into consideration. This makes sense because loading less language models means less memory consumption and better runtime performance.

[**difflib.SequenceMatcher.ratio**](https://docs.python.org/3/library/difflib.html#difflib.SequenceMatcher.ratio)
> Return a measure of the sequences’ similarity as a float in the range [0, 1]. Where T is the total number of elements in both sequences, and M is the number of matches, this is 2.0*M / T. Note that this is 1.0 if the sequences are identical, and 0.0 if they have nothing in common.


# Papers
## [Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models](https://arxiv.org/pdf/2404.18796)

- One of the largest issues with relying on a single model J, such as GPT-4, is that it introduces intra-model bias.
- Similar pooling techniques are used to reduce variance in human annotations by normalizing out both natural variation in human judgements caused by their own subjective biases as well as human error
- In prior relevant work, Li et al. (2023) proposed an evaluation method with multiple judges to reduce bias though only looked at large models on pair-wise evaluations.
- To calculate the PoLL score, each evaluator model independently scores a given model output just as they would in any of the scenarios outlined above. Those individual scores are then pooled together through a voting function


## [Argument Quality Assessment in the Age of Instruction-Following Large Language Models](https://aclanthology.org/2024.lrec-main.135.pdf)

- Instruction-following LLMs might not entirely resolve the issue behind; while they provide new means for a reliable evaluation (e.g., handling of context), their generative nature may also complicate the validation against some ground-truth.
- Rather, we propose that an evaluation procedure ideally takes into account the main decisive factors of quality, as we exemplified for the Huckleberry Finn claim above: What quality dimension is of interest, who is the audience of the argument, and similar. The evaluation may happen on a careful selection of existing datasets, but new benchmarks that account for the factors may be needed, too.
- Many argument quality dimensions imply some hard constraints, which speaks for an absolute part (e.g., are the argument’s premise acceptable?). However, there may not be a clear best/worst quality for an argument, which speaks for a relative part wherever other arguments are accessible (e.g., are the premises more acceptable than those of other arguments?).

## [Large language models and automated essay scoring of English language learner writing: Insights into validity and reliability](https://www.researchgate.net/publication/380497027_Large_language_models_and_automated_essay_scoring_of_English_language_learner_writing_Insights_into_validity_and_reliability)


```plaintext
You will be a professional language teacher who is an expert on language assessment and English language placement exams. You will use a rubric to give score to students' English language learners writing. Each sample of writing will receive a holistic score from 1-6 [characteristics of each score on the rubric inserted here (see Appendix A)]. Here is the writing prompt given to the student: "[redacted for test security]". Now apply the rubric given above to give a holistic score between 1-6 to the following student's writing: [student writing copied here in plain text]
```

**Judgement Variance by Prompt Changes**

- We hypothesize that may be because GPT-4 is over-reasoning and injecting too much background knowledge into determining the correctness of an answer rather than simply aligning the gold reference with the generation.
- The most effective strategy is an explicit instruction to the model not to ’overthink’ and not to concern itself with the wider factuality of the answers with respect to the outside world.


**GPT4 Judge Prompt Ablation**

- No instruction Line: Here we remove the natural language instruction from the Few-shot standard prompt, on the hypothesis the instruction is confusing the model. This hypothesis turns out to be false as agreement actually drops. (-0.03 ∆κ)
- Move instruction Line: Here we modify the standard fewshot prompt by moving the instruction line into a separate system call. This results in a small improvement (+0.01 ∆κ)
- Chat-formatted shots: Here we modify the standard fewshot prompt by formatting each fewshot example as a conversational turn between the user and the assistant. This reduces performance (-0.07 ∆κ)
- ‘don’t overthink’: Here, we replace the instruction line from the standard prompt’s

```plaintext
You will be given a Question and a Provided Answer. Judge whether the Provided Answer is correct by comparing it to the Reference Answer. Differently formatted dates, people with missing middle names, and alternative spellings should all be considered the same. If the Provided Answer is correct say exactly "True", otherwise say "False".
```


... to a wording which is intended to encourage the model to perform a simpler function and not incorporate external knowledge: 

```plaintext
You are judging whether a model has generated a correct answer to a question. Study the examples the user gives you as they will be very informative for how to do the task. The Reference Answers you get will be short. An model’s answer will be longer, and can be considered correct if it contains the semantic content of short reference answer somewhere within it. Don’t worry about factuality with respect to the real world, just judge the example based on what you see. No need to overthink this task, it really comes down to just soft matching. 
```

This improves results for GPT-4 by +0.07 ∆κ. Additional small surface level changes and moving the instruction to a system call lead to an additional +0.03 ∆κ. The final optimized prompt for GPT-4 can be found in table 14.

## [Cheating Automatic LLM Benchmarks: Null Models Achieve High Win Rates](https://arxiv.org/pdf/2410.07137)

- Structured cheating responses: 
    1. It overrides the original instruction-output triplet with a fabricated one;
    2. When positioned by default, it exploits the annotator’s general preference for the last output, guiding it to predict “M”; 
    3. When swapped, it takes advantage of overwriting the output from model “M”, causing the annotator to predict “m”.

- Crafting adversarial prefix by random search (RS):  Therefore, we develop a transferable prefix, crafted using a publicly available instruction set. Our approach optimizes a single adversarial prefix by aggregating the losses over various instructions, ensuring that the prefix’s impact is universal across different input instructions and positions.


## OpenAI Prompting Guides

- https://platform.openai.com/docs/guides/prompt-engineering
- https://cookbook.openai.com/articles/related_resources
