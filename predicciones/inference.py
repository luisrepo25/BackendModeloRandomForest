"""
Módulo para realizar predicciones con el modelo entrenado
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from django.core.cache import cache
import os

from .ml_model import VentasPredictor
from .models import Producto


class PrediccionVentas:
    """
    Clase para realizar predicciones de ventas
    """
    
    def __init__(self):
        self.predictor = VentasPredictor()
        self._cargar_modelo_si_existe()
    
    def _cargar_modelo_si_existe(self):
        """
        Intenta cargar el modelo si existe
        """
        try:
            self.predictor.cargar_modelo()
            self.modelo_cargado = True
        except FileNotFoundError:
            self.modelo_cargado = False
            print("⚠️ Modelo no encontrado. Necesitas entrenar el modelo primero.")
    
    def predecir_ventas_producto(self, producto_id, mes=None, anio=None, dias_futuro=30, usar_cache=True):
        """
        Predice las ventas de un producto específico
        
        Args:
            producto_id: ID del producto
            mes: Mes para la predicción (1-12). Si es None, usa el mes actual
            anio: Año para la predicción. Si es None, usa el año actual
            dias_futuro: Días hacia el futuro para predicción
            usar_cache: Si True, usa caché para resultados (5 min TTL)
        
        Returns:
            dict con la predicción y metadatos
        """
        if not self.modelo_cargado:
            raise ValueError("El modelo no está cargado. Entrena el modelo primero.")
        
        # Calcular fecha
        fecha_actual = datetime.now()
        if mes is None:
            mes = fecha_actual.month
        if anio is None:
            anio = fecha_actual.year
        
        # Intentar obtener de caché
        if usar_cache:
            from django.core.cache import cache
            cache_key = f"pred_{producto_id}_{mes}_{anio}"
            cached = cache.get(cache_key)
            if cached:
                return cached
        
        # Obtener información del producto
        try:
            producto = Producto.objects.select_related('categoria', 'marca').get(id=producto_id)
        except Producto.DoesNotExist:
            raise ValueError(f"Producto con ID {producto_id} no existe")
        
        # Calcular fecha de predicción
        fecha_actual = datetime.now()
        if mes is None:
            mes = fecha_actual.month
        if anio is None:
            anio = fecha_actual.year
        
        fecha_prediccion = datetime(anio, mes, 1)
        
        # Preparar features
        features = self._preparar_features_prediccion(
            producto=producto,
            mes=mes,
            anio=anio,
            fecha_prediccion=fecha_prediccion
        )
        
        # Realizar predicción
        X = pd.DataFrame([features])[self.predictor.feature_names]
        X_scaled = self.predictor.scaler.transform(X)
        
        # Predicción puntual
        prediccion = self.predictor.model.predict(X_scaled)[0]
        
        # Calcular intervalo de confianza usando predicciones de árboles individuales
        predicciones_arboles = np.array([
            arbol.predict(X_scaled)[0] 
            for arbol in self.predictor.model.estimators_
        ])
        
        intervalo_confianza = {
            'inferior': float(np.percentile(predicciones_arboles, 5)),
            'superior': float(np.percentile(predicciones_arboles, 95)),
            'std': float(np.std(predicciones_arboles))
        }
        
        resultado = {
            'prediccion': float(max(0, prediccion)),  # No puede ser negativo
            'intervalo_confianza': intervalo_confianza,
            'features_utilizados': features,
            'fecha_prediccion': fecha_prediccion,
            'producto': {
                'id': producto.id,
                'nombre': producto.nombre,
                'precio': float(producto.precio),
                'categoria': producto.categoria.nombre if producto.categoria else None,
                'marca': producto.marca.nombre if producto.marca else None
            },
            'dias_futuro': dias_futuro
        }
        
        # Guardar en caché (5 minutos)
        if usar_cache:
            from django.core.cache import cache
            cache_key = f"pred_{producto_id}_{mes}_{anio}"
            cache.set(cache_key, resultado, 300)
        
        return resultado
    
    def predecir_multiples_productos(self, productos_ids, mes=None, anio=None):
        """
        Predice ventas para múltiples productos
        """
        predicciones = []
        
        for producto_id in productos_ids:
            try:
                pred = self.predecir_ventas_producto(producto_id, mes, anio)
                predicciones.append(pred)
            except Exception as e:
                print(f"Error prediciendo producto {producto_id}: {str(e)}")
                continue
        
        return predicciones
    
    def predecir_tendencia_producto(self, producto_id, meses_futuro=6):
        """
        Predice la tendencia de ventas para los próximos N meses
        """
        from django.core.cache import cache
        
        # Intentar caché
        cache_key = f"tendencia_{producto_id}_{meses_futuro}"
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        fecha_actual = datetime.now()
        predicciones_tendencia = []
        
        for i in range(meses_futuro):
            fecha_pred = fecha_actual + timedelta(days=30*i)
            try:
                pred = self.predecir_ventas_producto(
                    producto_id=producto_id,
                    mes=fecha_pred.month,
                    anio=fecha_pred.year,
                    usar_cache=True
                )
                predicciones_tendencia.append({
                    'mes': fecha_pred.month,
                    'anio': fecha_pred.year,
                    'prediccion': pred['prediccion'],
                    'intervalo_confianza': pred['intervalo_confianza']
                })
            except Exception as e:
                print(f"Error en tendencia mes {i}: {str(e)}")
                continue
        
        resultado = {
            'producto_id': producto_id,
            'tendencia': predicciones_tendencia,
            'fecha_inicio': fecha_actual,
            'meses_proyectados': meses_futuro
        }
        
        # Guardar en caché (10 minutos)
        from django.core.cache import cache
        cache_key = f"tendencia_{producto_id}_{meses_futuro}"
        cache.set(cache_key, resultado, 600)
        
        return resultado
    
    def _preparar_features_prediccion(self, producto, mes, anio, fecha_prediccion):
        """
        Prepara los features para predicción
        """
        trimestre = (mes - 1) // 3 + 1
        dia_semana = fecha_prediccion.weekday()
        
        features = {
            'producto_id': producto.id,
            'mes': mes,
            'anio': anio,
            'trimestre': trimestre,
            'producto_precio': float(producto.precio),
            'producto_categoria_id': producto.categoria_id if producto.categoria_id else 0,
            'producto_marca_id': producto.marca_id if producto.marca_id else 0,
            'dia_semana': dia_semana
        }
        
        return features
    
    def obtener_productos_top_prediccion(self, top_n=10, mes=None, anio=None):
        """
        Obtiene los productos con mayor predicción de ventas
        """
        from django.core.cache import cache
        
        # Intentar caché
        cache_key = f"top_prod_{top_n}_{mes}_{anio}"
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        # Limitar a productos con stock (máximo 100 para no saturar)
        productos = Producto.objects.filter(stock__gt=0).select_related(
            'categoria', 'marca'
        ).values('id', 'nombre', 'precio')[:100]
        
        predicciones = []
        for producto in productos:
            try:
                pred = self.predecir_ventas_producto(
                    producto['id'], mes, anio, usar_cache=True
                )
                predicciones.append({
                    'producto_id': producto['id'],
                    'producto_nombre': producto['nombre'],
                    'prediccion': pred['prediccion'],
                    'precio': float(producto['precio'])
                })
            except Exception:
                continue
        
        # Ordenar por predicción
        predicciones_ordenadas = sorted(
            predicciones, 
            key=lambda x: x['prediccion'], 
            reverse=True
        )[:top_n]
        
        # Guardar en caché (10 minutos)
        cache.set(cache_key, predicciones_ordenadas, 600)
        
        return predicciones_ordenadas
    
    def predecir_ventas_totales_agregadas(self, meses_futuro=12, incluir_top_productos=5):
        """
        Predice ventas totales agregadas para graficar tendencia general
        Incluye también top N productos para comparación
        """
        from collections import defaultdict
        from django.core.cache import cache
        
        # Intentar caché
        cache_key = f"pred_agregada_{meses_futuro}_{incluir_top_productos}"
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        fecha_actual = datetime.now()
        
        # Limitar productos para Render gratuito (max 10 productos para evitar OOM)
        productos = Producto.objects.filter(stock__gt=0).select_related(
            'categoria', 'marca'
        ).values('id', 'nombre', 'precio')[:10]
        
        # Almacenar predicciones por mes
        predicciones_por_mes = defaultdict(lambda: {
            'cantidad_total': 0,
            'ingresos_estimados': 0,
            'productos_detalle': []
        })
        
        for i in range(meses_futuro):
            fecha_pred = fecha_actual + timedelta(days=30*i)
            mes = fecha_pred.month
            anio = fecha_pred.year
            mes_label = f"{anio}-{mes:02d}"
            
            for producto in productos:
                try:
                    pred = self.predecir_ventas_producto(
                        producto_id=producto['id'],
                        mes=mes,
                        anio=anio,
                        usar_cache=True
                    )
                    
                    cantidad = pred['prediccion']
                    precio = float(producto['precio'])
                    ingresos = cantidad * precio
                    
                    predicciones_por_mes[mes_label]['cantidad_total'] += cantidad
                    predicciones_por_mes[mes_label]['ingresos_estimados'] += ingresos
                    predicciones_por_mes[mes_label]['productos_detalle'].append({
                        'producto_id': producto['id'],
                        'nombre': producto['nombre'],
                        'cantidad': round(cantidad, 2),
                        'ingresos': round(ingresos, 2)
                    })
                    
                except Exception:
                    continue
        
        # Formatear para gráficas
        series_temporal = []
        for mes_label in sorted(predicciones_por_mes.keys()):
            data = predicciones_por_mes[mes_label]
            
            # Top productos del mes
            top_productos = sorted(
                data['productos_detalle'],
                key=lambda x: x['cantidad'],
                reverse=True
            )[:incluir_top_productos]
            
            series_temporal.append({
                'periodo': mes_label,
                'cantidad_total': round(data['cantidad_total'], 2),
                'ingresos_estimados': round(data['ingresos_estimados'], 2),
                'top_productos': top_productos
            })
        
        resultado = {
            'serie_temporal': series_temporal,
            'resumen': {
                'cantidad_total_proyectada': sum(p['cantidad_total'] for p in series_temporal),
                'ingresos_totales_proyectados': sum(p['ingresos_estimados'] for p in series_temporal),
                'meses_proyectados': meses_futuro,
                'fecha_generacion': fecha_actual
            }
        }
        
        # Guardar en caché (15 minutos)
        cache.set(cache_key, resultado, 900)
        
        return resultado


# Instancia global para reutilizar
_prediccion_instance = None

def obtener_predictor():
    """
    Obtiene la instancia singleton del predictor
    """
    global _prediccion_instance
    if _prediccion_instance is None:
        _prediccion_instance = PrediccionVentas()
    return _prediccion_instance
