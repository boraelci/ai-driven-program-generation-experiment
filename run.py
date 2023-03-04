from transformers import RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, f1_score, recall_score

def predict_codebert(input):
	tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
	model = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
	fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer)

	return fill_mask(input)

def predict_codegpt(input):
	tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeGPT-small-java-adaptedGPT2")
	model = AutoModelForCausalLM.from_pretrained("microsoft/CodeGPT-small-java-adaptedGPT2")
	text_generation = pipeline('text-generation', model=model, tokenizer=tokenizer)

	return text_generation(input)

def print_scores(model, actuals, predictions):
	if model == "codebert":
		name = "CodeBERT"
	else:
		name = "CodeGPT"

	accuracy = accuracy_score(actuals, predictions)
	print(f"{name} Accuracy: {accuracy}")

	"""
	f1 = f1_score(actuals, predictions, average="weighted")
	print(f"{name} F1-score: {f1}")

	recall = recall_score(actuals, predictions, average="weighted")
	print(f"{name} Recall: {recall}")
	"""

def run_codebert():
	model = "codebert"

	predictions = []
	actuals = []
	for i in range(1, 9):
		q = open(f"{model}/{i}q.txt", "r")
		input = q.read()
		prediction = predict_codebert(input)[0]["token_str"].strip()
		predictions.append(prediction)

		a = open(f"{model}/{i}a.txt", "r")
		actual = a.read()
		actuals.append(actual)

	print_scores(model, predictions, actuals)

def run_codegpt():
	model = "codegpt"

	predictions = []
	actuals = []
	for i in range(1, 9):
		q = open(f"{model}/{i}q.txt", "r")
		input = q.read()
		prediction = predict_codegpt(input)[0]["generated_text"].replace(input, "").strip()

		a = open(f"{model}/{i}a.txt", "r")
		actual = a.read()

		"""
		print(f"\n{i} {actual in prediction}")
		print(prediction)
		print(actual)
		"""
		if actual in prediction:
			predictions.append(1)
			actuals.append(1)
		else:
			predictions.append(0)
			actuals.append(1)

	print_scores(model, predictions, actuals)

if __name__ == "__main__":
	# run_codebert()
	run_codegpt()