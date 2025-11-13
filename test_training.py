"""
Script de prueba para entrenar el modelo
"""
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mlproject.settings')
django.setup()

from predicciones.models import NotaVenta, Detalle_Venta, Producto

print("\n" + "="*60)
print("🔍 VERIFICANDO DATOS EN LA BASE DE DATOS")
print("="*60)

# Verificar cantidad de datos
total_productos = Producto.objects.count()
total_ventas = NotaVenta.objects.filter(estado='pagada').count()
total_detalles = Detalle_Venta.objects.filter(nota_venta__estado='pagada').count()

print(f"\n✅ Total Productos: {total_productos}")
print(f"✅ Total Ventas Pagadas: {total_ventas}")
print(f"✅ Total Detalles de Venta: {total_detalles}")

if total_detalles == 0:
    print("\n⚠️ No hay datos de ventas para entrenar el modelo")
    print("   Necesitas tener ventas con estado='pagada' en la BD")
    exit(1)

print("\n" + "="*60)
print("🤖 INICIANDO ENTRENAMIENTO DEL MODELO")
print("="*60)

try:
    from predicciones.ml_model import entrenar_y_guardar_modelo
    
    print("\n🔄 Extrayendo y procesando datos...")
    resultado = entrenar_y_guardar_modelo()
    
    print("\n✅ ¡MODELO ENTRENADO EXITOSAMENTE!")
    print("\n📊 Métricas del Modelo:")
    metricas = resultado['metricas']
    print(f"   R² Test: {metricas['r2_test']:.4f}")
    print(f"   RMSE Test: {metricas['rmse_test']:.4f}")
    print(f"   MAE Test: {metricas['mae_test']:.4f}")
    print(f"   CV R² Mean: {metricas['cv_r2_mean']:.4f} ± {metricas['cv_r2_std']:.4f}")
    
    print("\n🎯 Importancia de Features (Top 5):")
    for i, (feature, importancia) in enumerate(list(resultado['feature_importance'].items())[:5], 1):
        print(f"   {i}. {feature}: {importancia:.4f}")
    
    print("\n💾 Modelo guardado en: predicciones/ml_models/modelo_rf.pkl")
    print("="*60 + "\n")
    
except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    import traceback
    print("\n📝 Detalle del error:")
    traceback.print_exc()
    print("="*60 + "\n")
    exit(1)
