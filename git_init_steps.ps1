# git_init_steps.ps1
# Run these commands one at a time (not as a script) so you can check
# output between steps. From inside the repo folder:

# 1. Confirm you're in the right place and .gitignore is present
Get-Location
Get-ChildItem .gitignore

# 2. Initialize git (skip if already a repo)
git init

# 3. Check what git sees BEFORE adding anything -- this is your chance to
#    catch venv/, __pycache__, models/best.pt etc. being picked up if
#    .gitignore isn't working as expected
git status

# 4. Stage everything
git add .

# 5. Double check what's actually staged -- look for anything that
#    shouldn't be there (large files, venv, __pycache__)
git status
git ls-files | Select-String "venv|__pycache__|\.pt$|\.mp4$|\.avi$"
#    ^ this should print NOTHING. If it prints file paths, STOP --
#      something is being tracked that shouldn't be. Fix .gitignore,
#      run `git rm --cached <file>` for anything wrongly staged, and
#      re-check before committing.

# 6. First commit
git commit -m "Initial commit: Week 8 tracking QA, F6/F11 diagnosis, reconciler, docs"

# 7. Check commit size sanity (should be small -- no video/model weights)
git count-objects -v
