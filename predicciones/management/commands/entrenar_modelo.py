"""
Comando Django para entrenar el modelo desde manage.py
Uso: python manage.py entrenar_modelo
"""
from django.core.management.base import BaseCommand
from predicciones.ml_model import entrenar_y_guardar_modelo
from predicciones.models import Detalle_Venta


class Command(BaseCommand):
    help = 'Entrena el modelo de Random Forest con los datos de ventas'

    def handle(self, *args, **options):
        self.stdout.write("=" * 60)
        self.stdout.write("ENTRENAMIENTO DEL MODELO RANDOM FOREST")
        self.stdout.write("=" * 60)
        
        # Verificar datos
        total_ventas = Detalle_Venta.objects.filter(nota_venta__estado='pagada').count()
        self.stdout.write(f"\nRegistros de ventas: {total_ventas}")
        
        if total_ventas < 100:
            self.stdout.write(
                self.style.WARNING(
                    f"\n⚠️ Solo hay {total_ventas} registros. Mínimo recomendado: 100"
                )
            )
        
        self.stdout.write("\nIniciando entrenamiento...\n")
        
        try:
            resultado = entrenar_y_guardar_modelo()
            
            self.stdout.write(self.style.SUCCESS("\n✅ Entrenamiento completado\n"))
            self.stdout.write("Métricas:")
            self.stdout.write(f"  R² Test: {resultado['metricas']['r2_test']:.4f}")
            self.stdout.write(f"  RMSE Test: {resultado['metricas']['rmse_test']:.4f}")
            self.stdout.write(f"  MAE Test: {resultado['metricas']['mae_test']:.4f}")
            
            self.stdout.write("\nModelo guardado en predicciones/ml_models/")
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"\n❌ Error: {str(e)}"))
            raise
