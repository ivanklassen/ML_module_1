import numpy as np
import pandas as pd
import time
import logging
import os
import pickle
import argparse

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

print(f"Текущий рабочий каталог: {os.getcwd()}")

# Константы
MODEL_DIR = "./model/"
RESULTS_FILE = "./data/results.csv"
LOG_FILE = "./data/log_file.log"
FOLDS = 5
RANDOM_STATE = 0

# # Настройка логирования
# logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(message)s')


class My_Classifier_Model:
    def __init__(self):

        self.classifiers = {
            "LGBM": LGBMClassifier(
                learning_rate=0.00974924788169623, max_depth=9, n_estimators=474, num_leaves=19, random_state=RANDOM_STATE
            ),
        }

        self.model = None  
        self.preprocessor = None 

    def train(self, dataset_name):

        train_file_path = f"{dataset_name}.csv"
        #logging.info(f"Попытка загрузить train.csv из: {train_file_path}")
        print(f"Попытка загрузить train.csv из: {train_file_path}") 

        #logging.info(f"Начало обучения модели на наборе данных: {dataset_name}")
        start_time = time.time()

        try:
            train = pd.read_csv(f"{dataset_name}.csv")
            train_y = train[["Transported"]].astype(int)

            train['Group'] = train['PassengerId'].apply(lambda x: x.split('_')[0])
            train['GroupSize'] = train.groupby('Group')['PassengerId'].transform('count')
            train.drop('Group', axis=1, inplace=True)

            train['CabinSide'] = train['Cabin'].apply(self._get_cabin_side)
            train['CabinDeck'] = train['Cabin'].apply(self._get_cabin_deck)

            train['VIP'] = train['VIP'].fillna(0)
            train['CryoSleep'] = train['CryoSleep'].fillna(0)

            train["VIP"] = train["VIP"].astype(int)
            train["CryoSleep"] = train["CryoSleep"].astype(int)

            train = train.drop(columns=["Name", "PassengerId", "Transported", "Cabin"])

            train[["RoomService_log"]] = np.log10(1 + train[["RoomService"]])
            train[["FoodCourt_log"]] = np.log10(1 + train[["FoodCourt"]])
            train[["ShoppingMall_log"]] = np.log10(1 + train[["ShoppingMall"]])
            train[["Spa_log"]] = np.log10(1 + train[["Spa"]])
            train[["VRDeck_log"]] = np.log10(1 + train[["VRDeck"]])

            train = train.drop(["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"], axis=1)

            num_cols = [cname for cname in train.columns if train[cname].dtype in ["int64", "float64"]]
            cat_cols = [cname for cname in train.columns if train[cname].dtype == "object"]

            num_transformer = Pipeline(steps=[("scaler", StandardScaler())])
            cat_transformer = Pipeline(
                steps=[
                    (
                        "onehot",
                        OneHotEncoder(drop='if_binary', handle_unknown="ignore")
                    )
                ]
            )

            self.preprocessor = ColumnTransformer(
                transformers = [
                    ("num", num_transformer, num_cols),
                    ("cat", cat_transformer, cat_cols),
                ],
                remainder="passthrough"
            )

            train = self.preprocessor.fit_transform(train)

            FOLDS = 5
            y = train_y.values
            preds_train = np.zeros(train.shape[0])

            cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=0)
            score = 0

            for fold, (train_idx, val_idx) in enumerate(cv.split(train, y)):
                X_train, X_valid = train[train_idx], train[val_idx]
                y_train, y_valid = y[train_idx], y[val_idx]

                clf = self.classifiers["LGBM"]
                clf.fit(X_train, y_train)

                preds_train += clf.predict_proba(train)[:, 1]
                score += clf.score(X_valid, y_valid)

                #logging.info(f"Fold {fold + 1} Accuracy: {clf.score(X_valid, y_valid):.4f}") #Логируем точность
                print(f"Fold {fold + 1} Accuracy: {clf.score(X_valid, y_valid):.4f}")
                
            score = score / FOLDS
            #logging.info(f"Средняя точность по кросс-валидации: {score:.4f}")
            print(f"Средняя точность по кросс-валидации: {score:.4f}")
          

            self.model = self.classifiers["LGBM"]  
            self.save_model()
            #logging.info("Модель успешно обучена и сохранена.")

            end_time = time.time()
            training_time = end_time - start_time
            #logging.info(f"Время обучения: {training_time:.2f} секунд")

        except FileNotFoundError:
            #logging.error(f"Файл {dataset_name}.csv не найден.")
            print(f"Ошибка: Файл {dataset_name}.csv не найден.")
        except Exception as e:
            #logging.exception("Произошла ошибка во время обучения:")
            print(f"Произошла ошибка: {e}")



    def predict(self, dataset_name):
        #logging.info(f"Начало предсказания на наборе данных: {dataset_name}")

        try:
            self.load_model() 

            test = pd.read_csv(f"{dataset_name}.csv")
            test_y = test[["PassengerId"]]

            test['Group'] = test['PassengerId'].apply(lambda x: x.split('_')[0])
            test['GroupSize'] = test.groupby('Group')['PassengerId'].transform('count')
            test = test.drop('Group', axis=1)

            test['CabinSide'] = test['Cabin'].apply(self._get_cabin_side)
            test['CabinDeck'] = test['Cabin'].apply(self._get_cabin_deck)

            test['VIP'] = test['VIP'].fillna(0)
            test['CryoSleep'] = test['CryoSleep'].fillna(0)

            test["VIP"] = test["VIP"].astype(int)
            test["CryoSleep"] = test["CryoSleep"].astype(int)

            test = test.drop(columns=["Name", "PassengerId", "Cabin"])

            test[["RoomService_log"]] = np.log10(1 + test[["RoomService"]])
            test[["FoodCourt_log"]] = np.log10(1 + test[["FoodCourt"]])
            test[["ShoppingMall_log"]] = np.log10(1 + test[["ShoppingMall"]])
            test[["Spa_log"]] = np.log10(1 + test[["Spa"]])
            test[["VRDeck_log"]] = np.log10(1 + test[["VRDeck"]])

            test = test.drop(["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"], axis=1)

            test = self.preprocessor.transform(test)
            probabilities = self.model.predict_proba(test)[:, 1]

            predictions = (probabilities > 0.5).astype(bool)

            submission = pd.DataFrame({'PassengerId': test_y['PassengerId'], 'Transported': predictions})
            submission.to_csv(RESULTS_FILE, index=False)

            #logging.info(f"Предсказания сохранены в файл: {RESULTS_FILE}")
            print(f"Предсказания сохранены в файл: {RESULTS_FILE}")

        except FileNotFoundError:
            #logging.error(f"Файл {dataset_name}.csv или файлы модели не найдены.")
            print(f"Ошибка: Файл {dataset_name}.csv или файлы модели не найдены.")
        except Exception as e:
            #logging.exception("Произошла ошибка во время предсказания:")
            print(f"Произошла ошибка: {e}")

    def save_model(self):
        os.makedirs(MODEL_DIR, exist_ok=True)

        with open(os.path.join(MODEL_DIR, 'model.pkl'), 'wb') as file:
            pickle.dump(self.model, file)
        #logging.info(f"Модель сохранена в {MODEL_DIR}model.pkl")

        with open(os.path.join(MODEL_DIR, 'preprocessor.pkl'), 'wb') as file:
            pickle.dump(self.preprocessor, file)
        #logging.info(f"Препроцессор сохранен в {MODEL_DIR}preprocessor.pkl")


    def load_model(self):
        try:
            with open(os.path.join(MODEL_DIR, 'model.pkl'), 'rb') as file:
                self.model = pickle.load(file)
            #logging.info(f"Модель загружена из {MODEL_DIR}model.pkl")

            with open(os.path.join(MODEL_DIR, 'preprocessor.pkl'), 'rb') as file:
                self.preprocessor = pickle.load(file)
            #logging.info(f"Препроцессор загружен из {MODEL_DIR}preprocessor.pkl")


        except FileNotFoundError:
            #logging.error("Файлы модели не найдены. Запустите обучение модели.")
            raise FileNotFoundError("Файлы модели не найдены. Запустите обучение модели.")
        except Exception as e:
             #logging.exception(f"Ошибка при загрузке модели: {e}")
             raise Exception(f"Ошибка при загрузке модели: {e}")

    def _get_cabin_side(self, cabin):

        if pd.isna(cabin):
            return None
        side = cabin.split('/')[-1]
        if side == 'P':
            return 0
        elif side == 'S':
            return 1
        else:
            return None

    def _get_cabin_deck(self, cabin):
        if pd.isna(cabin):
            return None
        try:
            parts = cabin.split('/')
            return f"{parts[0]}"
        except:
            return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate a model.')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--dataset', required=True, help='Path to the training dataset (train.csv)')

    predict_parser = subparsers.add_parser('predict', help='Make predictions with the model')
    predict_parser.add_argument('--dataset', required=True, help='Path to the evaluation dataset (test.csv)')

    args = parser.parse_args()

    model = My_Classifier_Model()

    if args.command == 'train':
        model.train(args.dataset)
    elif args.command == 'predict':
        model.predict(args.dataset)
    else:
        print("Invalid command. Use 'train' or 'predict'.")

