# The README's terminal recording: `make demo` (needs Docker).
#
# A GLM-5.3-Flash restore on 8×H100 SXM, replayed in real time from its log
# (docs/demo/glm53-flash-restore.log, 2026-10-10): `warmctl` is a
# shell function printing that log at its own timestamps. `up` is the
# restore command of that run, renamed since.
Output docs/assets/up.gif
Set Shell "bash"
Set FontFamily "Fira Code"
Set FontSize 18
Set Width 1240
Set Height 500
Set Padding 30
Set LineHeight 1.4
Set TypingSpeed 40ms
Set Theme { "name": "warmctl", "background": "#000000", "foreground": "#ededed", "cursor": "#ededed", "selection": "#3d3d3d", "black": "#0a0a0a", "brightBlack": "#666666", "red": "#ff6b6b", "brightRed": "#ff6b6b", "green": "#6dbb9a", "brightGreen": "#6dbb9a", "yellow": "#d9c27a", "brightYellow": "#d9c27a", "blue": "#7aa7d9", "brightBlue": "#7aa7d9", "magenta": "#a08fd6", "brightMagenta": "#a08fd6", "cyan": "#5f9fae", "brightCyan": "#5f9fae", "white": "#a1a1a1", "brightWhite": "#ededed" }

Hide
Type `source docs/demo/demo.bash; clear` Enter
Sleep 500ms
Show

Type "# GLM-5.3-Flash, FP8 + MTP, TP8 on 8x H100. A cold start takes 650 s." Enter
Sleep 800ms
Type "warmctl up /snaps/glm53-flash --gpus 0-7" Sleep 500ms Enter
Sleep 39s
Sleep 4s
