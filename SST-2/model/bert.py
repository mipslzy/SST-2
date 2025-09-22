from transformers import BertForSequenceClassification

def get_model(model_name, num_labels=2):
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model
