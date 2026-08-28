# push_to_github.ps1
# Run these one at a time, after git_init_steps.ps1 is done and your
# first commit is made.

# 1. Create a NEW, EMPTY repo on github.com first:
#    - Go to https://github.com/new
#    - Name it (e.g. door-monitoring-cv)
#    - Do NOT check "Add a README" or "Add .gitignore" -- you already
#      have both locally. Checking these creates conflicting history.
#    - Click "Create repository" and copy the URL it gives you,
#      e.g. https://github.com/<your-username>/door-monitoring-cv.git

# 2. Point your local repo at it (paste your actual URL below)
git remote add origin https://github.com/<your-username>/door-monitoring-cv.git

# 3. Rename branch to main (GitHub's default) if not already
git branch -M main

# 4. Push
git push -u origin main

# --- If push is rejected for being too large ---
# GitHub hard-blocks any single file over 100MB and warns above 50MB.
# If this happens:
git count-objects -v
#   Check "size" -- if it's unexpectedly large, something gitignored
#   wasn't actually ignored. Find it with:
git rev-list --objects --all | Select-String -Pattern "\.pt$|\.mp4$|\.avi$"
#   If it finds tracked large files, you'll need to remove them from
#   history (not just delete + commit) before pushing -- come back and
#   ask for help with this specifically rather than force-pushing blind.

# 5. After a successful push, verify on github.com:
#    - Confirm venv/, __pycache__/, models/best.pt, and the video files
#      under uploads/ do NOT appear in the repo (only .gitkeep should)
#    - Confirm docs/, diagnostics/, tests/, modules/ all look right
#    - Copy the repo URL to submit
