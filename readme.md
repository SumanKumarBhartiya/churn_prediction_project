# Churn Prediction

#Steps

1. python src/ingestion/fetch_data.py

2. python src/ingestion/kafka_consumer.py

3. python src/preprocessing/clean_data.py

4. python src/preprocessing/feature_engineering.py

5. python src/training/train.py

6. uvicorn src/deployment/api:app --reload

7. airflow dags trigger churn_pipeline

8. 