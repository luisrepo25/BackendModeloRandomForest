"""
Ejemplo de uso del sistema de predicciones ML
Ejecutar: python ejemplo_uso.py
"""
import os
import django

# Configurar Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mlproject.settings')
django.setup()

from predicciones.ml_model import entrenar_y_guardar_modelo, VentasPredictor
from predicciones.inference import obtener_predictor
from predicciones.models import Producto, NotaVenta, Detalle_Venta
from django.db.models import Sum, Count


def mostrar_estadisticas_bd():
    """Muestra estadísticas de la base de datos"""
    print("\n" + "="*60)
    print("📊 ESTADÍSTICAS DE LA BASE DE DATOS")
    print("="*60)
    
    total_productos = Producto.objects.count()
    total_ventas = NotaVenta.objects.filter(estado='pagada').count()
    total_detalles = Detalle_Venta.objects.filter(nota_venta__estado='pagada').count()
    
    print(f"✅ Total Productos: {total_productos}")
    print(f"✅ Total Ventas Pagadas: {total_ventas}")
    print(f"✅ Total Detalles de Venta: {total_detalles}")
    
    if total_detalles > 0:
        # Producto más vendido
        producto_top = Detalle_Venta.objects.filter(
            nota_venta__estado='pagada'
        ).values(
            'producto__nombre'
        ).annotate(
            total=Sum('cantidad')
        ).order_by('-total').first()
        
        if producto_top:
            print(f"\n🏆 Producto más vendido: {producto_top['producto__nombre']}")
            print(f"   Cantidad vendida: {producto_top['total']}")
    
    return total_detalles > 0


def entrenar_modelo_ejemplo():
    """Entrena el modelo Random Forest"""
    print("\n" + "="*60)
    print("🤖 ENTRENANDO MODELO RANDOM FOREST")
    print("="*60)
    
    try:
        resultado = entrenar_y_guardar_modelo()
        
        print("\n✅ Modelo entrenado exitosamente!")
        print("\n📈 Métricas del Modelo:")
        metricas = resultado['metricas']
        print(f"   R² Test: {metricas['r2_test']:.4f}")
        print(f"   RMSE Test: {metricas['rmse_test']:.4f}")
        print(f"   MAE Test: {metricas['mae_test']:.4f}")
        print(f"   CV R² Mean: {metricas['cv_r2_mean']:.4f} ± {metricas['cv_r2_std']:.4f}")
        
        print("\n🎯 Importancia de Features:")
        for feature, importancia in list(resultado['feature_importance'].items())[:5]:
            print(f"   {feature}: {importancia:.4f}")
        
        return True
        
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        return False


def ejemplo_prediccion_simple():
    """Ejemplo de predicción simple para un producto"""
    print("\n" + "="*60)
    print("🔮 EJEMPLO DE PREDICCIÓN SIMPLE")
    print("="*60)
    
    try:
        # Obtener un producto de ejemplo
        producto = Producto.objects.first()
        if not producto:
            print("❌ No hay productos en la base de datos")
            return
        
        print(f"\n📦 Producto: {producto.nombre} (ID: {producto.id})")
        print(f"   Precio: ${producto.precio}")
        print(f"   Stock: {producto.stock}")
        
        # Obtener predictor
        predictor = obtener_predictor()
        
        # Realizar predicción
        resultado = predictor.predecir_ventas_producto(
            producto_id=producto.id,
            mes=12,  # Diciembre
            anio=2025
        )
        
        print("\n🎯 Predicción de Ventas:")
        print(f"   Unidades estimadas: {resultado['prediccion']:.2f}")
        print(f"   Intervalo de confianza: [{resultado['intervalo_confianza']['inferior']:.2f}, {resultado['intervalo_confianza']['superior']:.2f}]")
        print(f"   Desviación estándar: ±{resultado['intervalo_confianza']['std']:.2f}")
        
        # Calcular ingresos estimados
        ingresos_estimados = resultado['prediccion'] * float(producto.precio)
        print(f"\n💰 Ingresos Estimados: ${ingresos_estimados:,.2f}")
        
    except Exception as e:
        print(f"\n❌ Error en predicción: {e}")


def ejemplo_tendencia():
    """Ejemplo de predicción de tendencia"""
    print("\n" + "="*60)
    print("📈 EJEMPLO DE TENDENCIA DE VENTAS")
    print("="*60)
    
    try:
        # Obtener un producto de ejemplo
        producto = Producto.objects.first()
        if not producto:
            print("❌ No hay productos en la base de datos")
            return
        
        print(f"\n📦 Producto: {producto.nombre} (ID: {producto.id})")
        
        # Obtener predictor
        predictor = obtener_predictor()
        
        # Obtener tendencia para 6 meses
        tendencia = predictor.predecir_tendencia_producto(
            producto_id=producto.id,
            meses_futuro=6
        )
        
        print("\n📊 Tendencia de Ventas (próximos 6 meses):")
        print("\n   Mes/Año  | Predicción | Intervalo Confianza")
        print("   " + "-"*50)
        
        for pred in tendencia['tendencia']:
            mes_nombre = [
                'Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'
            ][pred['mes'] - 1]
            
            print(f"   {mes_nombre}/{pred['anio']} | {pred['prediccion']:>10.2f} | [{pred['intervalo_confianza']['inferior']:.1f}, {pred['intervalo_confianza']['superior']:.1f}]")
        
    except Exception as e:
        print(f"\n❌ Error en tendencia: {e}")


def ejemplo_top_productos():
    """Ejemplo de top productos con predicción"""
    print("\n" + "="*60)
    print("🏆 TOP 5 PRODUCTOS CON MAYOR PREDICCIÓN")
    print("="*60)
    
    try:
        predictor = obtener_predictor()
        
        # Obtener top 5
        top_productos = predictor.obtener_productos_top_prediccion(
            top_n=5,
            mes=12,
            anio=2025
        )
        
        if not top_productos:
            print("\n❌ No se pudieron obtener predicciones")
            return
        
        print("\n   Rank | Producto                    | Predicción | Precio")
        print("   " + "-"*65)
        
        for idx, prod in enumerate(top_productos, 1):
            nombre = prod['producto_nombre'][:25].ljust(25)
            print(f"   #{idx}   | {nombre} | {prod['prediccion']:>10.2f} | ${prod['precio']:>8.2f}")
        
    except Exception as e:
        print(f"\n❌ Error en top productos: {e}")


def main():
    """Función principal"""
    print("\n" + "="*60)
    print("🚀 SISTEMA DE PREDICCIONES ML - DEMO")
    print("="*60)
    
    # 1. Verificar datos
    hay_datos = mostrar_estadisticas_bd()
    
    if not hay_datos:
        print("\n⚠️ No hay datos de ventas en la base de datos.")
        print("   Necesitas datos históricos para entrenar el modelo.")
        return
    
    # 2. Entrenar modelo
    print("\n¿Desea entrenar el modelo? (s/n): ", end="")
    respuesta = input().lower()
    
    if respuesta == 's':
        modelo_entrenado = entrenar_modelo_ejemplo()
        if not modelo_entrenado:
            return
    else:
        print("⏭️ Omitiendo entrenamiento...")
    
    # 3. Ejemplos de uso
    ejemplo_prediccion_simple()
    ejemplo_tendencia()
    ejemplo_top_productos()
    
    # Final
    print("\n" + "="*60)
    print("✅ DEMO COMPLETADA")
    print("="*60)
    print("\n📖 Para más información consulta:")
    print("   • README.md - Documentación completa")
    print("   • COMANDOS.md - Comandos útiles")
    print("   • ESTRUCTURA.md - Estructura del proyecto")
    print("\n🌐 Para usar la API REST:")
    print("   1. python manage.py runserver")
    print("   2. GET http://localhost:8000/api/predicciones/health/")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
