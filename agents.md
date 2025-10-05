## Rules

### No `__all__`

* Do **not** define `__all__` in any file or module.
* Public surface is whatever is defined in the file; consumers must explicitly import what they use.

### No command-line interfaces

* Do **not** add CLIs (`argparse`, `click`, `typer`, etc.), create methods unless otherwise instructed.
