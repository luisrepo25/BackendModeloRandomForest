from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.db.models import Sum, Count, Avg
from datetime import datetime

from .serializers import (
    PrediccionVentasInputSerializer,
    PrediccionVentasOutputSerializer,
    EntrenamientoModeloSerializer,
    EstadisticasVentasSerializer
)
from .ml_model import entrenar_y_guardar_modelo
from .inference import obtener_predictor
from .models import NotaVenta, Detalle_Venta, Producto


class EntrenarModeloView(APIView):
    """
    POST /api/predicciones/entrenar/
    Entrena el modelo de Random Forest con los datos históricos
    """
    
    def post(self, request):
        try:
            # Entrenar modelo
            resultado = entrenar_y_guardar_modelo()
            
            # Preparar respuesta
            response_data = {
                'mensaje': 'Modelo entrenado y guardado exitosamente',
                'metricas': {
                    'r2_test': round(resultado['metricas']['r2_test'], 4),
                    'rmse_test': round(resultado['metricas']['rmse_test'], 4),
                    'mae_test': round(resultado['metricas']['mae_test'], 4),
                    'cv_r2_mean': round(resultado['metricas']['cv_r2_mean'], 4),
                },
                'fecha_entrenamiento': resultado['fecha_entrenamiento'],
                'num_registros': Detalle_Venta.objects.filter(nota_venta__estado='pagada').count()
            }
            
            serializer = EntrenamientoModeloSerializer(data=response_data)
            if serializer.is_valid():
                return Response(serializer.data, status=status.HTTP_200_OK)
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except ValueError as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            return Response(
                {'error': f'Error al entrenar el modelo: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class PrediccionVentasView(APIView):
    """
    POST /api/predicciones/predecir/
    Realiza predicción de ventas para un producto
    
    Body:
    {
        "producto_id": 1,
        "mes": 12,  // opcional
        "anio": 2025,  // opcional
        "dias_futuro": 30  // opcional
    }
    """
    
    def post(self, request):
        # Validar entrada
        serializer = PrediccionVentasInputSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            # Obtener predictor
            predictor = obtener_predictor()
            
            # Realizar predicción
            resultado = predictor.predecir_ventas_producto(
                producto_id=serializer.validated_data.get('producto_id'),
                mes=serializer.validated_data.get('mes'),
                anio=serializer.validated_data.get('anio'),
                dias_futuro=serializer.validated_data.get('dias_futuro', 30)
            )
            
            return Response(resultado, status=status.HTTP_200_OK)
            
        except ValueError as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            return Response(
                {'error': f'Error al realizar predicción: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class TendenciaVentasView(APIView):
    """
    GET /api/predicciones/tendencia/{producto_id}/?meses=6
    Obtiene la tendencia de ventas para los próximos N meses
    """
    
    def get(self, request, producto_id):
        try:
            meses_futuro = int(request.query_params.get('meses', 6))
            
            predictor = obtener_predictor()
            resultado = predictor.predecir_tendencia_producto(
                producto_id=producto_id,
                meses_futuro=meses_futuro
            )
            
            return Response(resultado, status=status.HTTP_200_OK)
            
        except ValueError as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            return Response(
                {'error': f'Error al obtener tendencia: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class TopProductosPrediccionView(APIView):
    """
    GET /api/predicciones/top-productos/?top=10&mes=12&anio=2025
    Obtiene los productos con mayor predicción de ventas
    """
    
    def get(self, request):
        try:
            top_n = int(request.query_params.get('top', 10))
            mes = request.query_params.get('mes')
            anio = request.query_params.get('anio')
            
            if mes:
                mes = int(mes)
            if anio:
                anio = int(anio)
            
            predictor = obtener_predictor()
            resultado = predictor.obtener_productos_top_prediccion(
                top_n=top_n,
                mes=mes,
                anio=anio
            )
            
            return Response(
                {'top_productos': resultado, 'total': len(resultado)},
                status=status.HTTP_200_OK
            )
            
        except Exception as e:
            return Response(
                {'error': f'Error al obtener top productos: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class EstadisticasVentasView(APIView):
    """
    GET /api/predicciones/estadisticas/
    Obtiene estadísticas generales de ventas
    """
    
    def get(self, request):
        try:
            # Obtener ventas pagadas
            ventas_pagadas = NotaVenta.objects.filter(estado='pagada')
            detalles = Detalle_Venta.objects.filter(nota_venta__estado='pagada')
            
            # Estadísticas básicas
            total_ventas = detalles.aggregate(total=Count('id'))['total'] or 0
            total_ingresos = ventas_pagadas.aggregate(total=Sum('total'))['total'] or 0
            promedio_venta = ventas_pagadas.aggregate(avg=Avg('total'))['avg'] or 0
            
            # Producto más vendido
            producto_top = detalles.values(
                'producto__nombre'
            ).annotate(
                total_cantidad=Sum('cantidad')
            ).order_by('-total_cantidad').first()
            
            producto_mas_vendido = producto_top['producto__nombre'] if producto_top else 'N/A'
            
            # Categoría más vendida
            categoria_top = detalles.values(
                'producto__categoria__nombre'
            ).annotate(
                total_cantidad=Sum('cantidad')
            ).order_by('-total_cantidad').first()
            
            categoria_mas_vendida = categoria_top['producto__categoria__nombre'] if categoria_top else 'N/A'
            
            # Periodo de datos
            primera_venta = ventas_pagadas.order_by('created_at').first()
            ultima_venta = ventas_pagadas.order_by('-created_at').first()
            
            response_data = {
                'total_ventas': total_ventas,
                'total_ingresos': float(total_ingresos),
                'producto_mas_vendido': producto_mas_vendido,
                'categoria_mas_vendida': categoria_mas_vendida,
                'promedio_venta': float(promedio_venta),
                'periodo': {
                    'desde': primera_venta.created_at if primera_venta else None,
                    'hasta': ultima_venta.created_at if ultima_venta else None
                }
            }
            
            serializer = EstadisticasVentasSerializer(data=response_data)
            if serializer.is_valid():
                return Response(serializer.data, status=status.HTTP_200_OK)
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response(
                {'error': f'Error al obtener estadísticas: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class PrediccionAgregadaView(APIView):
    """
    GET /api/predicciones/agregadas/?meses=12&top_productos=5
    Obtiene predicción de ventas totales agregadas para gráficas
    """
    
    def get(self, request):
        try:
            meses = int(request.query_params.get('meses', 12))
            top_productos = int(request.query_params.get('top_productos', 5))
            
            predictor = obtener_predictor()
            resultado = predictor.predecir_ventas_totales_agregadas(
                meses_futuro=meses,
                incluir_top_productos=top_productos
            )
            
            return Response(resultado, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response(
                {'error': f'Error al obtener predicción agregada: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class HealthCheckView(APIView):
    """
    GET /api/predicciones/health/
    Verifica el estado del servicio y el modelo
    """
    
    def get(self, request):
        try:
            predictor = obtener_predictor()
            modelo_estado = 'cargado' if predictor.modelo_cargado else 'no entrenado'
            
            return Response({
                'status': 'ok',
                'modelo': modelo_estado,
                'timestamp': datetime.now(),
                'database': 'conectada'
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now()
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class LimpiarCacheView(APIView):
    """
    POST /api/predicciones/limpiar-cache/
    Limpia el caché de predicciones
    """
    
    def post(self, request):
        try:
            from django.core.cache import cache
            cache.clear()
            
            return Response({
                'mensaje': 'Caché limpiado exitosamente',
                'timestamp': datetime.now()
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
