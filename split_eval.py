
import json
import random

random.seed(42)

data = json.load(open("eval_set.json", encoding="utf-8"))
random.shuffle(data)

test = data[:8]
train = data[8:]

json.dump(train, open("eval_set.json", "w", encoding="utf-8"), indent=2)
json.dump(test, open("eval_test.json", "w", encoding="utf-8"), indent=2)

print(f"Training: {len(train)} questions")
print(f"Test: {len(test)} questions")
