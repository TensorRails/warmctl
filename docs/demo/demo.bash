# Shell setup of the terminal recording (docs/demo/up.tape): a prompt, and
# `warmctl` replaying the real restore log with the markers highlighted.
PS1='\[\e[38;2;102;102;102m\]8×H100 \[\e[38;2;122;167;217m\]❯ \[\e[0m\]'
warmctl() {
  docs/demo/replay.sh docs/demo/glm53-flash-restore.log |
    sed -u -E $'s/^(serving started .*)$/\e[1;38;2;109;187;154m\\1\e[0m/; s/^(\\[ *[0-9.]+s\\])/\e[38;2;102;102;102m\\1\e[0m/'
}
