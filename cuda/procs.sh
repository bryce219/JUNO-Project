# Finds the processes that belong to this copy of JUNO, by the folder they're running in. That way a second copy on
# the same computer doesn't get mixed up with this one. PROJECT has to be set to the top folder first
# eg: ours -x scan, or ours -f '^bash .*supervise\.sh'
ours() {
    for pid in $(pgrep "$@"); do
        case "$(readlink "/proc/$pid/cwd" 2>/dev/null)" in
            "$PROJECT" | "$PROJECT"/*) echo "$pid" ;;
        esac
    done
}
