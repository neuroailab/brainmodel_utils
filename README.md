# brainmodel_utils
Basic utilities for comparing models to neural & behavioral data, along with packaging these data in Python (from Matlab).

# Installation
To install run:
```
git clone https://github.com/anayebi/brainmodel_utils
cd brainmodel_utils/
pip install -e .
```

# Code Formatting:
Put this in `.git/hooks/pre-commit`, and run `sudo chmod +x .git/hooks/pre-commit`.

```
#!/usr/bin/env bash
  
echo "# Running pre-commit hook"
echo "#########################"

echo "Checking formatting"

format_occurred=false
declare -a black_dirs=("brainmodel_utils/" "setup.py")
for black_dir in "${black_dirs[@]}"; do
    echo ">>> Checking $black_dir"
    black --check "$black_dir"

    if [ $? -ne 0 ]; then
        echo ">>> Reformatting now!"
        black "$black_dir"
        format_occurred=true
    fi
done

if [ "$format_occurred" = true ]; then
    exit 1
fi
```
# License
MIT

# Contact
If you have any questions or encounter issues, either submit a Github issue here (preferred) or [email me](https://anayebi.github.io/contact/).
