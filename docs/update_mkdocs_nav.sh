#!/bin/bash
# Update mkdocs.yml nav section to include all .md files in docs/, grouped by subfolder as currently organized

set -e

MKDOCS_YML="$(dirname "$0")/mkdocs.yml"
DOCS_ROOT="$(dirname "$0")"
TMP_NAV="$(mktemp)"

# Helper: output nav for a folder
output_nav_for_folder() {
    local folder="$1"
    local indent="$2"
    local title="$3"
    echo "${indent}- ${title}:" >> "$TMP_NAV"
    for f in "$DOCS_ROOT/$folder"/*.md; do
        [ -f "$f" ] || continue
        fname=$(basename "$f")
        # Remove .md extension and capitalize words for display
        display_name=$(echo "$fname" | sed 's/.md$//' | sed 's/_/ /g' | sed 's/\b\([a-z]\)/\u\1/g')
        echo "${indent}    - ${display_name}: ${folder}/${fname}" >> "$TMP_NAV"
    done
}

# Start nav output
cat <<EOF > "$TMP_NAV"
    - Home: index.md
EOF

output_nav_for_folder "src" "    " "Source Code"
output_nav_for_folder "software" "    " "Software"
output_nav_for_folder "classifier" "    " "Classifier"
output_nav_for_folder "bbbb" "    " "HH4b Analysis"
output_nav_for_folder "bbWW" "    " "bbWW Analysis"

cat <<EOF >> "$TMP_NAV"
    - Documentation: readme.md
EOF

# Replace nav section in mkdocs.yml
python3 -c '
import sys
yml_path = sys.argv[1]
nav_path = sys.argv[2]
with open(yml_path, "r") as f:
    lines = f.readlines()
new_lines = []
in_nav = False
for line in lines:
    if line.startswith("nav:"):
        new_lines.append(line)
        in_nav = True
        with open(nav_path, "r") as nf:
            new_lines.extend(nf.readlines())
        continue
    if in_nav:
        stripped = line.strip()
        if stripped.startswith("-") or not stripped:
            continue
        else:
            in_nav = False
    new_lines.append(line)
with open(yml_path, "w") as f:
    f.writelines(new_lines)
' "$MKDOCS_YML" "$TMP_NAV"

rm "$TMP_NAV"

echo "mkdocs.yml navigation updated!"
