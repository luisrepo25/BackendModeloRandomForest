from rest_framework import serializers
from .models import NotaVenta, Detalle_Venta, Producto


class PrediccionVentasInputSerializer(serializers.Serializer):
    """
    Serializer para recibir datos de entrada para predicción
    """
    producto_id = serializers.IntegerField(required=False)
    categoria_id = serializers.IntegerField(required=False)
    marca_id = serializers.IntegerField(required=False)
    mes = serializers.IntegerField(min_value=1, max_value=12, required=False)
    anio = serializers.IntegerField(min_value=2020, required=False)
    precio = serializers.DecimalField(max_digits=10, decimal_places=2, required=False)
    dias_futuro = serializers.IntegerField(min_value=1, max_value=365, default=30)


class PrediccionVentasOutputSerializer(serializers.Serializer):
    """
    Serializer para la respuesta de predicción
    """
    prediccion = serializers.FloatField()
    intervalo_confianza = serializers.DictField(child=serializers.FloatField())
    features_utilizados = serializers.DictField()
    fecha_prediccion = serializers.DateTimeField()


class EntrenamientoModeloSerializer(serializers.Serializer):
    """
    Serializer para respuesta del entrenamiento del modelo
    """
    mensaje = serializers.CharField()
    metricas = serializers.DictField()
    fecha_entrenamiento = serializers.DateTimeField()
    num_registros = serializers.IntegerField()


class ProductoVentasSerializer(serializers.ModelSerializer):
    """
    Serializer para datos históricos de productos
    """
    total_ventas = serializers.IntegerField(read_only=True)
    ingresos_totales = serializers.DecimalField(max_digits=10, decimal_places=2, read_only=True)
    
    class Meta:
        model = Producto
        fields = ['id', 'nombre', 'precio', 'stock', 'categoria', 'marca', 'total_ventas', 'ingresos_totales']


class EstadisticasVentasSerializer(serializers.Serializer):
    """
    Serializer para estadísticas generales de ventas
    """
    total_ventas = serializers.IntegerField()
    total_ingresos = serializers.DecimalField(max_digits=15, decimal_places=2)
    producto_mas_vendido = serializers.CharField()
    categoria_mas_vendida = serializers.CharField()
    promedio_venta = serializers.DecimalField(max_digits=10, decimal_places=2)
    periodo = serializers.DictField()
