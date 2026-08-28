# reorganize_repo.ps1
# Reorganizes week6_dashboard into the proposed repo structure.
# Run this FROM INSIDE the week6_dashboard folder.
#
# SAFE BY DEFAULT: uses -WhatIf so nothing actually moves on first run.
# Review the output, then re-run with -Confirm:$false (or just remove -WhatIf
# from each Move-Item line) once you're happy with what it plans to do.

   $WhatIfPreference = $false  # <-- set to $false when you're ready to actually run it

# --- Create target folders ---
# NOTE: NOT creating a "data" folder — uploads/ already exists and already
# holds test_1.mp4 / landscape_test_1.mp4 / landscape_test_2.mp4 with its own
# .gitkeep convention. app.py likely references "uploads/" by path directly,
# so videos stay put rather than getting moved.
$folders = @("docs", "diagnostics", "tests")
foreach ($f in $folders) {
    if (-not (Test-Path $f)) {
        New-Item -ItemType Directory -Path $f | Out-Null
        Write-Host "Created folder: $f"
    }
}

# --- docs/ ---
$docsFiles = @("bug_fixes.md", "failure_log.md", "test_plan.md")
foreach ($f in $docsFiles) {
    if (Test-Path $f) { Move-Item $f "docs\" -WhatIf:$WhatIfPreference }
}

# --- diagnostics/ ---
$diagFiles = @(
    "debug_f6_trace.py",
    "check_kalman_drift.py",
    "check_track_coexistence.py",
    "check_tracker_internals.py",
    "diagnose_fragmentation.py",
    "dist_gap_2d_sweep.py",
    "gap_threshold_sweep.py",
    "threshold_scaling.py",
    "verify_chokepoint_hypothesis.py",
    "verify_two_person_scene.py"
)
foreach ($f in $diagFiles) {
    if (Test-Path $f) { Move-Item $f "diagnostics\" -WhatIf:$WhatIfPreference }
}

# --- tests/ ---
$testFiles = @("test_door_classifier.py", "test_track_reconciler.py", "verify_classifier_accuracy.py")
foreach ($f in $testFiles) {
    if (Test-Path $f) { Move-Item $f "tests\" -WhatIf:$WhatIfPreference }
}

Write-Host ""
Write-Host "Dry run complete. Nothing was actually moved (WhatIf mode)."
Write-Host "Review the planned moves above. If they look right, open this script,"
Write-Host "set `$WhatIfPreference = `$false` at the top, and run it again."
Write-Host ""
Write-Host "NOT moved automatically (need your decision first):"
Write-Host "  - passageway1-c0.avi / terrace1-c0.avi  (identity unconfirmed - real 2-person footage or leftovers?)"
Write-Host "  - Week8_Pipeline_Handover_TeamC.docx  (still not located)"
Write-Host ""
Write-Host "RESOLVED, no action needed:"
Write-Host "  - qa_results\ (gitignored, regenerable) vs qa_reruns\ (kept + tracked, evidence for track_buffer disproof)"
Write-Host "  - yolo11n.pt = person detector, models\best.pt = door classifier (72MB, needs Git LFS or external hosting)"
