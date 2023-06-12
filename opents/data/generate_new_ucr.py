import json
import random

# load the JSON data into Python
with open("open_ucr.json", "r") as fin:
    data = json.load(fin)

# Iterate over each entry in the JSON dictionary.
for key in data:
    items = list(data[key].items())
    # Randomly shuffle the key-value pairs.
    random.shuffle(items)
    # Divide into three piles.
    n = len(items) // 3
    test_sets = [{k: v for k, v in items[i * n:(i + 1) * n]} for i in range(3)]
    train_sets = [{k: v for k, v in items if (k, v) not in test_set} for test_set in test_sets]
    
    # Generate three test sets and training sets for each entry 
    for i in range(3):
        # Write to json file
        with open(f'{key}_test_set_{i}.json', 'w') as f:
            json.dump(test_sets[i], f)
        with open(f'{key}_train_set_{i}.json', 'w') as f:
            json.dump(train_sets[i], f)
