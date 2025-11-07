## Guidelines
Follow these code guidelines, unless the prompt specifically prompts you to implement something.
* Do **not** define `__all__` in any file or module.
* Do **not** add CLIs (`argparse`, `click`, `typer`, etc.), create methods unless otherwise instructed.
* Do **not** define a __main__ clause unless prompted.
* Do **not** defina a __main__ clause only to raise an exception
* Do **not** define ad hoc types if a pytorch class fits
* Do **not** do checks if a variable is negative that's supposed to be positive or non-negative