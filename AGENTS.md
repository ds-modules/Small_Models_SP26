# AGENTS.md — Small Models Demo Class

## Project Purpose
This repository contains Jupyter notebooks for a freshman-level college course on 
small language models. All notebooks are teaching materials, not production code.

## Primary Rule
Always optimize for student comprehension, never for code elegance or efficiency.
These two goals are often in direct conflict. Always choose comprehension. A student should be able to read the notebook and understand each chunk of code.


## Mandatory Code Constraints
- All code must be written in the global scope, executed sequentially cell by cell.
- Prefer lines of code to functions unless the notebook lesson is specifically teaching functions.
- Never use list comprehensions. Always use explicit for-loops.
- Never use lambda expressions.
- Never abstract repeated code into helpers. Repetition is acceptable and often 
  pedagogically valuable — students benefit from seeing the same pattern twice.
- Imports go in the first cell only, one per line, with a comment explaining 
  what each library does.
- When necessary use try;except blocks to import libraries that may not be installed, but do not add any error handling beyond printing an error message.

## Notebook Structure the Agent Must Follow
1. Start with a markdown cell stating the learning objective in one sentence.
2. Alternate markdown and code cells throughout. Never put two code cells in a row 
   without a markdown explanation between them.
3. End with a markdown cell summarizing what the student just did.

## Language and Explanation Level
- Explain as if the reader has never seen a concept before.
- Define every technical term the first time it appears.
- Never assume context. Never say "as you know" or "recall that."
- Preferred sentence length: short. Preferred vocabulary: common English words.

## What the Agent Should Never Do
- Never wrap the main demo logic in a `main()` function or any equivalent.
- Never add error handling (try/except) unless the lesson is about error handling.
- Never generate a README or extra files unless explicitly asked.
- Never refactor working code to be "cleaner" — clarity of the original steps 
  matters more than clean code.
- Don't name dataframes `df`. Use descriptive names like `sales_data` or `customer_info`.
- Don't use generic variable names like `x` or `temp`. Use descriptive names that indicate their purpose, such as `total_sales` or `average_age`.
