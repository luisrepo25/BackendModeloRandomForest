"""
Script para entrenar el modelo localmente antes de subirlo al servidor
Ejecutar: python entrenar_local.py
"""
import os
import django

# Configurar Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mlproject.settings')
django.setup()

from predicciones.ml_model import entrenar_y_guardar_modelo
from predicciones.models import Detalle_Venta

def main():
    print("=" * 60)
    print("ENTRENAMIENTO LOCAL DEL MODELO RANDOM FOREST")
    print("=" * 60)
    
    # Verificar datos disponibles
    total_ventas = Detalle_Venta.objects.filter(nota_venta__estado='pagada').count()
    print(f"\nRegistros de ventas disponibles: {total_ventas}")
    
    if total_ventas < 100:
        print("\n⚠️ ADVERTENCIA: Pocos datos para entrenar (mínimo recomendado: 100)")
        respuesta = input("¿Deseas continuar? (s/n): ")
        if respuesta.lower() != 's':
            print("Entrenamiento cancelado.")
            return
    
    print("\nIniciando entrenamiento...")
    print("-" * 60)
    
    try:
        resultado = entrenar_y_guardar_modelo()
        
        print("\n" + "=" * 60)
        print("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("=" * 60)
        
        print("\nMÉTRICAS DEL MODELO:")
        print(f"  R² (Test):        {resultado['metricas']['r2_test']:.4f}")
        print(f"  RMSE (Test):      {resultado['metricas']['rmse_test']:.4f}")
        print(f"  MAE (Test):       {resultado['metricas']['mae_test']:.4f}")
        print(f"  CV R² (Mean):     {resultado['metricas']['cv_r2_mean']:.4f}")
        print(f"  CV R² (Std):      {resultado['metricas']['cv_r2_std']:.4f}")
        
        print("\nIMPORTANCIA DE FEATURES:")
        for feature, importancia in list(resultado['feature_importance'].items())[:5]:
            print(f"  {feature:25s}: {importancia:.4f}")
        
        print("\n" + "=" * 60)
        print("ARCHIVOS GENERADOS:")
        print("  - predicciones/ml_models/modelo_rf.pkl")
        print("  - predicciones/ml_models/scaler.pkl")
        print("  - predicciones/ml_models/feature_names.pkl")
        print("=" * 60)
        
        print("\nPASOS SIGUIENTES:")
        print("1. Verifica los archivos en predicciones/ml_models/")
        print("2. Agrega los archivos al repositorio:")
        print("   git add predicciones/ml_models/*.pkl")
        print("   git commit -m 'Add trained model'")
        print("   git push")
        print("3. Deploy en Render - el modelo ya estará entrenado")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ ERROR EN EL ENTRENAMIENTO")
        print("=" * 60)
        print(f"\nError: {str(e)}")
        print("\nVerifica:")
        print("  - Conexión a la base de datos")
        print("  - Datos de ventas disponibles")
        print("  - Dependencias instaladas (requirements.txt)")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
