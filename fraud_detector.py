import time
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from snapml import LogisticRegression as SnapLogisticRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class FraudDetector:
    def __init__(self):
        # Carga directa del dataset (Requisito del PDF)
        self.df = pd.read_csv('creditcard.csv')
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Modelos (Pista 4: n_jobs=-1 para Scikit-Learn)
        self.sklearn_model = LogisticRegression(n_jobs=-1, max_iter=1000, random_state=42)
        self.snapml_model = SnapLogisticRegression(max_iter=1000, random_state=42)

    def preprocess(self):
        # Pista 2: Estandarizar la columna Amount
        scaler = StandardScaler()
        self.df['Amount'] = scaler.fit_transform(self.df[['Amount']])
        
        # Eliminamos Time (no estandarizada) y extraemos los valores (numpy arrays) para Snap ML
        X = self.df.drop(columns=['Class', 'Time']).values
        y = self.df['Class'].values
        
        # Particion 80/20. stratify=y por el alto desbalance
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    def train_and_evaluate(self):
        # 1. Entrenamiento y evaluacion: Scikit-Learn
        start_sk = time.time()
        self.sklearn_model.fit(self.X_train, self.y_train)
        time_sk = time.time() - start_sk
        
        pred_sk = self.sklearn_model.predict(self.X_test)
        f1_sk = f1_score(self.y_test, pred_sk) # Pista 1: F1-Score
        
        # 2. Entrenamiento y evaluacion: Snap ML
        start_snap = time.time()
        self.snapml_model.fit(self.X_train, self.y_train)
        time_snap = time.time() - start_snap
        
        pred_snap = self.snapml_model.predict(self.X_test)
        f1_snap = f1_score(self.y_test, pred_snap) # Pista 1: F1-Score

        # Mostrar resultados en consola
        print(f"Scikit-Learn -> Tiempo: {time_sk:.4f}s | F1-Score: {f1_sk:.4f}")
        print(f"Snap ML      -> Tiempo: {time_snap:.4f}s | F1-Score: {f1_snap:.4f}")
        
        # Generar grafica comparativa
        plt.bar(['Scikit-Learn', 'Snap ML'], [time_sk, time_snap], color=['blue', 'orange'])
        plt.title('Comparacion de Tiempos de Entrenamiento')
        plt.ylabel('Segundos')
        plt.savefig('comparacion_tiempos.png')
        print("Grafica generada y guardada como 'comparacion_tiempos.png'")

if __name__ == "__main__":
    detector = FraudDetector()
    detector.preprocess()
    detector.train_and_evaluate()
