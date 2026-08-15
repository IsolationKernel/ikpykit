# Frequently Asked Questions

This page collects common questions about IKPyKit usage, installation, and model behavior.

## 1. Which Python versions are supported?

IKPyKit supports Python 3.11 and above.

## 2. How do I install IKPyKit?

Recommended installation (with `uv`):

Install `uv` first: <https://docs.astral.sh/uv/getting-started/>

```bash
uv pip install ikpykit
```

If you prefer classic pip:

```bash
pip install ikpykit
```

For more options, see [How to install](../quick-start/how-to-install.md).


## 3. Why do I sometimes see `-1` labels in clustering outputs?

Some clustering methods can leave points unassigned and mark them as `-1` (outliers).
For example, IDKC may stop before every point is assigned because of threshold and
early-stop criteria.

## 4. How can I report a bug or request a feature?

Open an issue on GitHub:

- https://github.com/IsolationKernel/ikpykit/issues

When reporting bugs, include:
- IKPyKit version
- Python version
- Minimal reproducible code snippet
- Full traceback or error output

## 5. Where can I find runnable examples?

See [Examples and tutorials](../examples/examples_english.md) and
[User Guides](../user_guides/table-of-contents.md).
