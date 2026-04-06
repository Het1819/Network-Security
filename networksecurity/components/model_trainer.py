import os
import sys

from networksecurity.exception.exception import NetWorkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object, load_object
from networksecurity.utils.main_utils.utils import load_numpy_array_data, evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
import mlflow
import dagshub
dagshub.init(repo_owner='sailesh5419', repo_name='Network-Security', mlflow=True)



class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetWorkSecurityException(e,sys)

    def track_mlflow(self, best_model, train_metric, test_metric):
        try:
            mlflow.end_run()
            # mlflow.set_tracking_uri("sqlite:///mlflow.db")
            
            mlflow.set_experiment("NetworkSecurityExperiment")

            print("Tracking URI:", mlflow.get_tracking_uri())

            with mlflow.start_run() as run:
                print("Run ID:", run.info.run_id)

                mlflow.log_metric("train_f1_score", train_metric.f1_score)
                mlflow.log_metric("train_precision_score", train_metric.precision_score)
                mlflow.log_metric("train_recall_score", train_metric.recall_score)

                mlflow.log_metric("test_f1_score", test_metric.f1_score)
                mlflow.log_metric("test_precision_score", test_metric.precision_score)
                mlflow.log_metric("test_recall_score", test_metric.recall_score)

                mlflow.sklearn.log_model(sk_model=best_model, name="model")

        except Exception as e:
            raise NetWorkSecurityException(e, sys)


    # def track_mlflow(self, best_model, train_metric, test_metric):
    #     try:
    #         mlflow.end_run()
    #         mlflow.set_tracking_uri("sqlite:///mlflow.db")
    #         mlflow.set_experiment("NetworkSecurityExperiment")

    #         print("Tracking URI:", mlflow.get_tracking_uri())

    #         with mlflow.start_run() as run:
    #             print("Run ID:", run.info.run_id)

    #             mlflow.log_metric("train_f1_score", train_metric.f1_score)
    #             mlflow.log_metric("train_precision_score", train_metric.precision_score)
    #             mlflow.log_metric("train_recall_score", train_metric.recall_score)

    #             mlflow.log_metric("test_f1_score", test_metric.f1_score)
    #             mlflow.log_metric("test_precision_score", test_metric.precision_score)
    #             mlflow.log_metric("test_recall_score", test_metric.recall_score)

    #             mlflow.sklearn.log_model(sk_model=best_model, name="model")

    #     except Exception as e:
    #         raise NetWorkSecurityException(e, sys)
        
    # def track_mlflow(self, best_model, classificationmetric):
    #         with mlflow.start_run():
    #             mlflow.set_tracking_uri("sqlite:///mlflow.db")
    #             mlflow.set_experiment("NetworkSecurityExperiment")

    #             print("Tracking URI:", mlflow.get_tracking_uri())

    #             f1_score = classificationmetric.f1_score
    #             precision_score = classificationmetric.precision_score
    #             recall_score = classificationmetric.recall_score

    #             mlflow.log_metric("f1_score", f1_score)
    #             mlflow.log_metric("precision_score", precision_score)
    #             mlflow.log_metric("recall_score", recall_score)
    #             mlflow.sklearn.log_model(sk_model=best_model, name="model")




    def train_model(self,x_train, y_train, x_test, y_test):
        try: 
            models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(verbose=1, max_iter=1000),
                "AdaBoost": AdaBoostClassifier(),
            }
            params = {
                "Decision Tree": {
                    'criterion': ['gini', 'entropy', 'log_loss'],
                    # 'splitter':['best','random'],
                    # 'max_features': ['sqrt','log2']
                },
                "Random Forest": {
                    #  'criterion': ['gini', 'entropy', 'log_loss'],
                    # 'max_features': ['sqrt','log2', None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting": {
                    # 'loss': ['log_loss', 'exponential'],
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'criterion': ['squared_error', 'friedman_mse'],
                    'max_features': ['sqrt', 'log2', None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Logistic Regression": {},
                "AdaBoost": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8,16,32,64,128,256]
                }

            }
            model_report: dict = evaluate_models(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                                                models=models,param=params)
            
            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            y_train_pred = best_model.predict(x_train)
            classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
            
            # Track the experiements with MLflow
            # self.track_mlflow(best_model,classification_train_metric)     
            


            y_test_pred = best_model.predict(x_test)
            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)
        
            # self.track_mlflow(best_model,classification_test_metric)     
            
            self.track_mlflow(
                best_model=best_model,
                train_metric=classification_train_metric,
                test_metric=classification_test_metric
            )

            preprocessor = load_object(file_path = self.data_transformation_artifact.transformed_object_file_path)


            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path,exist_ok=True)

            Network_Model = NetworkModel(preprocessor=preprocessor, model=best_model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=Network_Model)


            save_object("final_models/model.pkl", best_model)

            # Model Trainer Artifact
            model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path = self.model_trainer_config.trained_model_file_path,
                                train_metric_artifact = classification_train_metric,
                                test_metric_artifact=classification_test_metric)

            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise NetWorkSecurityException(e,sys)


    def initiate_model_trainer(self)-> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # Loading training array and testig array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:,:-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )
            model_trainer_artifact = self.train_model(x_train=x_train,y_train=y_train, x_test=x_test, y_test=y_test)
            return model_trainer_artifact
        
        except Exception as e:
            raise NetWorkSecurityException(e,sys)