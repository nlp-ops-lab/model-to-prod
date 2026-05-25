from prefect import flow, task

from src.services.finbert_service import predict_sentiment


@task
def preprocess_text(text: str):
    return text.strip()


@task
def run_inference(text: str):
    return predict_sentiment(text)


@flow
def sentiment_pipeline(text: str):

    clean_text = preprocess_text(text)

    result = run_inference(clean_text)

    return result