## Guidelines
Follow these code guidelines, unless the prompt specifically prompts you to implement something.
* We're using python 3.12, use all standards accordingly
* Do **not** define `__all__` in any file or module.
* Do **not** add CLIs (`argparse`, `click`, `typer`, etc.), create methods unless otherwise instructed.
* The folder jobs is for scripts. Never import from the jobs folder and **do not add main guards** when writing scripts in that folder.
* Do **not** define a __main__ clause only to raise an exception
* Do **not** define ad hoc types if a pytorch class fits
* Do **not** do checks if a variable is negative that's supposed to be positive or non-negative