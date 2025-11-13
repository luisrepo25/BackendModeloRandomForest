"""
Módulo para entrenar el modelo de Random Forest para predicción de ventas
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
from django.conf import settings
from django.db.models import Sum, Count, F
from datetime import datetime

from .models import NotaVenta, Detalle_Venta, Producto


class VentasPredictor:
    """
    Clase para entrenar y gestionar el modelo de predicción de ventas
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.model_path = os.path.join(settings.ML_MODELS_DIR, 'modelo_rf.pkl')
        self.scaler_path = os.path.join(settings.ML_MODELS_DIR, 'scaler.pkl')
        
    def extraer_features_ventas(self):
        """
        Extrae features de las ventas históricas desde la base de datos
        """
        from django.db.models import F
        
        # Obtener datos de ventas pagadas optimizado
        detalles = Detalle_Venta.objects.filter(
            nota_venta__estado='pagada'
        ).select_related(
            'producto', 
            'producto__categoria', 
            'producto__marca',
            'nota_venta'
        ).values(
            'producto_id',
            'producto__nombre',
            producto_precio=F('producto__precio'),
            producto_categoria_id=F('producto__categoria_id'),
            producto_marca_id=F('producto__marca_id'),
            fecha_venta=F('nota_venta__created_at'),
            cantidad_vendida=F('cantidad'),
            subtotal_venta=F('subtotal')
        )
        
        # Convertir QuerySet a lista
        ventas_data = list(detalles)
        
        # Convertir a DataFrame
        df = pd.DataFrame(ventas_data)
        
        if df.empty:
            raise ValueError("No hay datos de ventas disponibles para entrenar el modelo")
        
        # Rellenar valores nulos
        df['producto_categoria_id'] = df['producto_categoria_id'].fillna(0)
        df['producto_marca_id'] = df['producto_marca_id'].fillna(0)
        
        # Extraer features temporales
        df['fecha'] = pd.to_datetime(df['fecha_venta'])
        df['mes'] = df['fecha'].dt.month
        df['anio'] = df['fecha'].dt.year
        df['dia_semana'] = df['fecha'].dt.dayofweek
        df['dia_mes'] = df['fecha'].dt.day
        df['trimestre'] = df['fecha'].dt.quarter
        
        # Función auxiliar para obtener la moda de forma segura
        def safe_mode(x):
            if len(x) == 0:
                return 0
            mode_result = x.mode()
            return mode_result.iloc[0] if len(mode_result) > 0 else x.iloc[0]
        
        # Agregar por producto y periodo
        features = df.groupby(['producto_id', 'mes', 'anio']).agg({
            'cantidad_vendida': 'sum',
            'subtotal_venta': 'sum',
            'producto_precio': 'first',
            'producto_categoria_id': 'first',
            'producto_marca_id': 'first',
            'trimestre': 'first',
            'dia_semana': safe_mode
        }).reset_index()
        
        return features
    
    def preparar_datos(self, df):
        """
        Prepara los datos para entrenamiento
        """
        # Features de entrada
        self.feature_names = [
            'producto_id', 'mes', 'anio', 'trimestre', 
            'producto_precio', 'producto_categoria_id', 
            'producto_marca_id', 'dia_semana'
        ]
        
        X = df[self.feature_names].copy()
        y = df['cantidad_vendida'].values
        
        # Manejar valores nulos
        X = X.fillna(0)
        
        return X, y
    
    def entrenar_modelo(self, test_size=0.2, random_state=42):
        """
        Entrena el modelo Random Forest
        """
        print("🔄 Extrayendo datos de ventas...")
        df = self.extraer_features_ventas()
        
        print(f"📊 Total de registros: {len(df)}")
        
        # Preparar datos
        X, y = self.preparar_datos(df)
        
        # Dividir en train y test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Escalar features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("🤖 Entrenando modelo Random Forest...")
        
        # Configurar y entrenar Random Forest
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluar modelo
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        metricas = {
            'r2_train': r2_score(y_train, y_pred_train),
            'r2_test': r2_score(y_test, y_pred_test),
            'rmse_train': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'rmse_test': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'mae_train': mean_absolute_error(y_train, y_pred_train),
            'mae_test': mean_absolute_error(y_test, y_pred_test),
        }
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, 
            cv=5, scoring='r2', n_jobs=-1
        )
        metricas['cv_r2_mean'] = cv_scores.mean()
        metricas['cv_r2_std'] = cv_scores.std()
        
        print(f"✅ Modelo entrenado!")
        print(f"   R² Test: {metricas['r2_test']:.4f}")
        print(f"   RMSE Test: {metricas['rmse_test']:.4f}")
        print(f"   MAE Test: {metricas['mae_test']:.4f}")
        
        return metricas
    
    def guardar_modelo(self):
        """
        Guarda el modelo entrenado y el scaler
        """
        if self.model is None:
            raise ValueError("Primero debes entrenar el modelo")
        
        os.makedirs(settings.ML_MODELS_DIR, exist_ok=True)
        
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        joblib.dump(self.feature_names, os.path.join(settings.ML_MODELS_DIR, 'feature_names.pkl'))
        
        print(f"💾 Modelo guardado en: {self.model_path}")
    
    def cargar_modelo(self):
        """
        Carga el modelo entrenado desde disco
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"No se encontró el modelo en {self.model_path}")
        
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.feature_names = joblib.load(os.path.join(settings.ML_MODELS_DIR, 'feature_names.pkl'))
        
        return True
    
    def obtener_importancia_features(self):
        """
        Obtiene la importancia de cada feature
        """
        if self.model is None:
            raise ValueError("Primero debes entrenar o cargar el modelo")
        
        importancia = dict(zip(self.feature_names, self.model.feature_importances_))
        return dict(sorted(importancia.items(), key=lambda x: x[1], reverse=True))


def entrenar_y_guardar_modelo():
    """
    Función auxiliar para entrenar y guardar el modelo
    """
    predictor = VentasPredictor()
    metricas = predictor.entrenar_modelo()
    predictor.guardar_modelo()
    
    return {
        'metricas': metricas,
        'fecha_entrenamiento': datetime.now(),
        'feature_importance': predictor.obtener_importancia_features()
    }
