# merge_deliverables.ps1
# Run this ONCE, before git init, to bring the Week 8 Team C docs/scripts
# into the actual project repo (week6_dashboard).
#
# EDIT THESE TWO PATHS to match your machine:
$DocsFolder = "G:\dataset for internship\Week 6\Week 8 Team C"
$RepoFolder = "G:\dataset for internship\Week 6\week6_dashboard\week6_dashboard"

cd $RepoFolder

# --- Copy corrected/created files from the docs folder into the repo ---
$filesToCopy = @(
    "bug_fixes.md",
    "failure_log.md",
    "test_plan.md",
    "debug_f6_trace.py",
    "README.md",
    "Week7_TrackingQA_Report_CORRECTED.docx",
    "Week8_Pipeline_Handover_TeamC.docx",
    "Individual_Learning_Summary_Week8.docx",
    "Presentation slides.pptx"
)

foreach ($f in $filesToCopy) {
    $src = Join-Path $DocsFolder $f
    if (Test-Path $src) {
        Copy-Item $src . -Force
        Write-Host "Copied: $f"
    } else {
        Write-Host "NOT FOUND (skipped): $f"
    }
}

# --- Handle .gitignore separately: don't blindly overwrite if one exists ---
$existingGitignore = Join-Path $RepoFolder ".gitignore"
$recommendedGitignore = Join-Path $DocsFolder "gitignore_recommended.txt"

if (Test-Path $existingGitignore) {
    Write-Host ""
    Write-Host "A .gitignore already exists in the repo. NOT overwritten automatically."
    Write-Host "Compare it against gitignore_recommended.txt and merge manually:"
    Write-Host "  notepad .gitignore"
    Write-Host "  notepad `"$recommendedGitignore`""
} else {
    Copy-Item $recommendedGitignore ".gitignore" -Force
    Write-Host "No existing .gitignore found -- copied gitignore_recommended.txt as .gitignore"
}

Write-Host ""
Write-Host "Done. Now run reorganize_repo.ps1 (dry run first), then git init."
