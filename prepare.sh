#!/bin/bash
# re-validate login information
rm -rf ./.auth
mkdir -p ./.auth
python browser_env/auto_login.py