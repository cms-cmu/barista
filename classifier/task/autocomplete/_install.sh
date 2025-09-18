#!/bin/bash
_pyml_python_bind() {
    local cmds=(python -m classifier.task.autocomplete._bind)
    if [ "$1" == "wait" ]; then
        cmds+=("wait")
    fi
    if [ "$2" == "mid" ]; then
        opts=$("${cmds[@]}" "${COMP_WORDS[@]:0:COMP_CWORD}" "" 2>&1)
    else
        opts=$("${cmds[@]}" "${COMP_WORDS[@]:0:COMP_CWORD+1}" 2>&1)
    fi
    code=$?
}

_pyml_task_autocomplete() {
    local opts code
    local cur="${COMP_WORDS[COMP_CWORD]}"
    local mid=""
    if [ $((${COMP_CWORD}+1)) -ne ${#COMP_WORDS[@]} ] && [ "${COMP_LINE:COMP_POINT-1:1}" == " " ]; then
        cur=""
        mid="mid"
    fi
    _pyml_python_bind "" ${mid}
    while [ $code -eq 254 ]; do
        (python -m classifier.task.autocomplete._core &)
        _pyml_python_bind wait ${mid}
    done
    if [ $code -eq 0 ]; then
        mapfile -t COMPREPLY < <( compgen -W "${opts}" -- "${cur}")
        return 0
    elif [ $code -eq 1 ]; then
        echo "${opts}"
    elif [ $code -eq 255 ]; then
        if command -v _filedir &>/dev/null; then
            _filedir
            return 0
        fi
    fi
    COMPREPLY=()
    return 0
}

complete -F _pyml_task_autocomplete "./pyml.py"