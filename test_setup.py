"""
Script para probar la conexión a la base de datos y configuración inicial
"""
import os
import django
import sys

# Configurar Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mlproject.settings')
django.setup()

from django.db import connection
from predicciones.models import Producto, NotaVenta, Detalle_Venta


def test_database_connection():
    """Prueba la conexión a la base de datos"""
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT version();")
            version = cursor.fetchone()
            print("✅ Conexión a PostgreSQL exitosa!")
            print(f"   Versión: {version[0]}\n")
        return True
    except Exception as e:
        print(f"❌ Error al conectar con la base de datos: {e}\n")
        return False


def get_database_stats():
    """Obtiene estadísticas de la base de datos"""
    try:
        total_productos = Producto.objects.count()
        total_ventas = NotaVenta.objects.filter(estado='pagada').count()
        total_detalles = Detalle_Venta.objects.count()
        
        print("📊 Estadísticas de la Base de Datos:")
        print(f"   Total Productos: {total_productos}")
        print(f"   Total Ventas (pagadas): {total_ventas}")
        print(f"   Total Detalles de Venta: {total_detalles}\n")
        
        if total_detalles > 0:
            print("✅ Hay suficientes datos para entrenar el modelo")
        else:
            print("⚠️ No hay datos de ventas. Necesitas datos históricos para entrenar el modelo.")
        
        return True
    except Exception as e:
        print(f"❌ Error al obtener estadísticas: {e}\n")
        return False


def check_ml_directory():
    """Verifica que el directorio de modelos ML existe"""
    from django.conf import settings
    ml_dir = settings.ML_MODELS_DIR
    
    if os.path.exists(ml_dir):
        print(f"✅ Directorio ML existe: {ml_dir}")
        
        # Verificar si hay modelo entrenado
        modelo_path = os.path.join(ml_dir, 'modelo_rf.pkl')
        if os.path.exists(modelo_path):
            print(f"   ✅ Modelo entrenado encontrado: modelo_rf.pkl\n")
        else:
            print(f"   ⚠️ No hay modelo entrenado. Ejecuta POST /api/predicciones/entrenar/\n")
    else:
        print(f"⚠️ Directorio ML no existe. Creando...\n")
        os.makedirs(ml_dir, exist_ok=True)
        print(f"✅ Directorio creado: {ml_dir}\n")


def main():
    print("=" * 60)
    print("🔧 TEST DE CONFIGURACIÓN - Backend ML Predicciones")
    print("=" * 60 + "\n")
    
    # Test 1: Conexión a BD
    if not test_database_connection():
        sys.exit(1)
    
    # Test 2: Estadísticas
    get_database_stats()
    
    # Test 3: Directorio ML
    check_ml_directory()
    
    print("=" * 60)
    print("✅ Configuración completada!")
    print("=" * 60)
    print("\n📖 Próximos pasos:")
    print("   1. Ejecutar: python manage.py runserver")
    print("   2. Probar: GET http://localhost:8000/api/predicciones/health/")
    print("   3. Entrenar: POST http://localhost:8000/api/predicciones/entrenar/")
    print("\n   Consulta README.md para más información")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
