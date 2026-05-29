#!/usr/bin/env bash
#
# Disclosure scrub gate.
#
# This repository is an engineering artifact: code comments, docstrings, and
# docs describe ENGINEERING only. This gate fails the build if internal or
# non-engineering content slips into the tracked tree -- specifically the known
# leak class: internal validation-program taxonomy, prosecution / claim-element
# framing, and internal paths or names.
#
# Patterns are kept high-signal so the gate does not false-positive on
# legitimate engineering content or on the intended patent posture in NOTICE
# (which is allowed: it carries the patent application numbers + a one-line
# "as practiced by the SDK code as distributed" posture, and nothing here).
#
# If a future match is a genuine false positive, narrow the pattern below
# rather than weakening the gate wholesale.
set -uo pipefail

PATTERN='reduction.to.?practice'
PATTERN+='|non-provisional'
PATTERN+='|provisional drafter'
PATTERN+='|claim language'
PATTERN+='|claims [0-9]+ and [0-9]+'
PATTERN+='|distinguishing argument'
PATTERN+='|\bDI-[0-9]'
PATTERN+='|Stage [A-E]\b'
PATTERN+='|\bM1 Max\b'
PATTERN+='|MasterVision'
PATTERN+='|/Users/'
PATTERN+='|_shareable|_handoff|_patents'
PATTERN+='|Shaheen|Mandragona|Wenling'
PATTERN+='|\bcounsel\b'

# Search tracked files only; exclude this script (it necessarily names the
# patterns it is looking for).
hits=$(git grep -nEI "$PATTERN" -- . ':(exclude)scripts/check_disclosure.sh' 2>/dev/null || true)

if [ -n "$hits" ]; then
  echo "Disclosure scrub gate FAILED -- internal/non-engineering content found:"
  echo "-------------------------------------------------------------------"
  echo "$hits"
  echo "-------------------------------------------------------------------"
  echo "Public repos are engineering-only. Remove the content above. If a match"
  echo "is a genuine false positive, narrow the pattern in"
  echo "scripts/check_disclosure.sh (do not weaken the gate wholesale)."
  exit 1
fi

echo "Disclosure scrub gate passed (0 hits)."
