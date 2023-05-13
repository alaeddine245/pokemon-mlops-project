# pokemon-mlops-project

mlflow server: http://mlops.uksouth.cloudapp.azure.com:5000
launch mlflow server (hoster on azure)
```
mlflow server -h 0.0.0.0
```

Launch training 
```
cd pokemon-mlops-project-main/src/
python training_pipeline.py
```

Launch training with opentelemetry
```
opentelemetry-bootstrap -a install
cd pokemon-mlops-project-main/src/

opentelemetry-instrument --traces_exporter console --metrics_exporter console python pokemon-mlops-project-main/src/training_pipeline.py > output
```
