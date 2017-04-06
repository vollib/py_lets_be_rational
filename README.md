## troubleshooting

### There is no llvm-config

`py_lets_be_rational` depends on `numba` which in turn depends on `llvm-lite`. `llvm-lite` wants LLVM 3.9 being installed.

```
brew install llvm@3.9
LLVM_CONFIG=/usr/local/opt/llvm@3.9/bin/llvm-config pip install numba
```

## development

Fork our repository, and make the changes on your local copy:

```
git clone git@github.com/your_username/py_lets_be_rational
cd py_lets_be_rational
pip install -e .
pip install -r dev-requirements.txt
```

When you are done, push the changes, and create a pull request on your repo.
