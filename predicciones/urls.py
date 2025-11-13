from django.urls import path
from .views import (
    EntrenarModeloView,
    PrediccionVentasView,
    TendenciaVentasView,
    TopProductosPrediccionView,
    EstadisticasVentasView,
    PrediccionAgregadaView,
    HealthCheckView,
    LimpiarCacheView
)

app_name = 'predicciones'

urlpatterns = [
    # Health check
    path('health/', HealthCheckView.as_view(), name='health-check'),
    
    # Entrenamiento
    path('entrenar/', EntrenarModeloView.as_view(), name='entrenar-modelo'),
    
    # Cache
    path('limpiar-cache/', LimpiarCacheView.as_view(), name='limpiar-cache'),
    
    # Predicciones
    path('predecir/', PrediccionVentasView.as_view(), name='predecir-ventas'),
    path('tendencia/<int:producto_id>/', TendenciaVentasView.as_view(), name='tendencia-ventas'),
    path('top-productos/', TopProductosPrediccionView.as_view(), name='top-productos'),
    path('agregadas/', PrediccionAgregadaView.as_view(), name='prediccion-agregada'),
    
    # Estadísticas
    path('estadisticas/', EstadisticasVentasView.as_view(), name='estadisticas'),
]
