#!/usr/bin/env bash
# Enable tab completion for: python run.py --CON<TAB>
# Run once (or after a new machine / shell setup):
#   source scripts/enable_run_completion.sh
#
# Uses argcomplete's global Python hook so no alias or wrapper is needed.

set -euo pipefail

if ! command -v activate-global-python-argcomplete >/dev/null 2>&1; then
    echo "argcomplete is not installed. Install with: pip install argcomplete" >&2
    return 1 2>/dev/null || exit 1
fi

if ! python -c "import argcomplete" >/dev/null 2>&1; then
    echo "Installing argcomplete into the active Python environment..."
    python -m pip install argcomplete
fi

activate-global-python-argcomplete --user

BASHRC="${HOME}/.bashrc"
SOURCE_LINE='[ -f ~/.bash_completion ] && . ~/.bash_completion'
if [[ -f "${BASHRC}" ]] && ! grep -Fq '~/.bash_completion' "${BASHRC}"; then
    printf '\n# argcomplete (python run.py tab completion)\n%s\n' "${SOURCE_LINE}" >> "${BASHRC}"
    echo "Added bash completion hook to ${BASHRC}"
fi

# Load in the current shell when sourced (not when executed).
if [[ -f "${HOME}/.bash_completion" ]]; then
    # shellcheck disable=SC1091
    . "${HOME}/.bash_completion"
fi

echo "Tab completion enabled for: python run.py --CON<TAB>"
echo "If TAB still does nothing, open a new terminal or run: source ~/.bash_completion"
