


import re
action_splitter = "```"
pattern = rf"{action_splitter}((.|\n)*?){action_splitter}"

response_one_match = """Let's think step-by-step. This page list the information of HP Inkjet Fax Machine, which is the product identified in the objective. Its price is $279.49. I think I have achieved the objective. I will issue the stop action with the answer. In summary, the next action I will perform is ```stop [$279.49]```"""
match = re.search(pattern, response_one_match)
match.group(1).strip()

matchs = re.findall(pattern, response_one_match)
matchs[-1][0]




response = """Let's think step-by-step. Currently, the page shows the driving route from the input locations. In order to compare the driving route time with the walking route time, I need to change the transport mode to walking. Specifically, I need to modify the URL to change the parameter ```engine=fossgis_osrm_car``` to ```engine=fossgis_osrm_foot```. In summary, the next action I will perform is ```goto [http://openstreetmap.org/directions?engine=fossgis_osrm_foot&route=40.447%2C-79.942%3B40.456%2C-79.941#map=16/40.4515/-79.9429]```"""
matchs = re.findall(pattern, response)
matchs[-1][0].strip()


response_wrong = """Lets think."""

matchs = re.findall(pattern, response_wrong)

if matchs:
    print("Matched")