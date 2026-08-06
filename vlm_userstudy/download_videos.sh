#!/usr/bin/env bash
# Download the four study videos with yt-dlp (pip install yt-dlp).
set -e
cd "$(dirname "$0")"
mkdir -p videos
cd videos
yt-dlp -f "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]" -o "v1_blobs.mp4"       "https://youtu.be/HRO9I9SAnPE"
yt-dlp -f "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]" -o "v2_cross.mp4"       "https://youtu.be/XJE1sP6E7BE"
yt-dlp -f "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]" -o "v3_aggregation.mp4" "https://youtu.be/5w4qfmG87q8"
yt-dlp -f "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]" -o "v4_hospital.mp4"    "https://youtu.be/joD1h7QhaNU"
echo "done."
