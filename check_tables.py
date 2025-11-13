"""
Script para verificar las tablas en la base de datos
"""
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mlproject.settings')
django.setup()

from django.db import connection

def listar_tablas():
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT tablename 
            FROM pg_tables 
            WHERE schemaname='public' 
            ORDER BY tablename;
        """)
        tablas = cursor.fetchall()
        
        print("\n" + "="*60)
        print("📊 TABLAS EN LA BASE DE DATOS")
        print("="*60)
        for tabla in tablas:
            print(f"  • {tabla[0]}")
        print("="*60 + "\n")

if __name__ == '__main__':
    listar_tablas()
