#!/bin/bash


export CLASSIFIEDS="http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:9980"
export CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"
export SHOPPING="http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:7770"
export SHOPPING_ADMIN="http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:7780/admin"
export REDDIT="http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:9999"
export GITLAB="http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:8023"
export MAP="http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:3000"
export WIKIPEDIA="http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export HOMEPAGE="PASS"  # The home page is not currently hosted in the demo site

# re-validate login information
mkdir -p ./.auth
python browser_env/auto_login.py