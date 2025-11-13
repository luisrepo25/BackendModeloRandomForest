from django.db import models

# Create your models here.

class Categoria(models.Model):
    nombre = models.CharField(max_length=100)
    descripcion = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'products_categoria'

    def __str__(self):
        return self.nombre
    
class Marca(models.Model):
    nombre = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'products_marca'

    def __str__(self):
        return self.nombre
    
class Garantia(models.Model):
    # Cobertura de garantia en meses
    cobertura = models.IntegerField()
    Marca = models.ForeignKey(Marca, on_delete=models.CASCADE, related_name='garantias')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'products_garantia'

    def __str__(self):
        return f"Garantía de {self.duracion_meses} meses para {self.producto.nombre}"
    
class Producto(models.Model):
    nombre = models.CharField(max_length=200)
    descripcion = models.TextField()
    precio = models.DecimalField(max_digits=10, decimal_places=2)
    stock = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    # imagen = CloudinaryField(
    #     'image',
    #     folder='productos',  # Carpeta en Cloudinary
    #     blank=True,
    #     null=True,
    #     transformation={
    #         'quality': 'auto:good',
    #         'fetch_format': 'auto',
    #         'width': 800,
    #         'height': 800,
    #         'crop': 'limit'
    #     }
    # )

    categoria = models.ForeignKey(Categoria, on_delete=models.CASCADE, related_name='productos')
    marca = models.ForeignKey(Marca, on_delete=models.CASCADE, related_name='productos')
    garantia = models.ForeignKey(Garantia, on_delete=models.CASCADE, related_name='productos', blank=True, null=True)

    class Meta:
        db_table = 'products_producto'

    def __str__(self):
        return self.nombre

    @property
    def imagen_url(self):
        """Retorna la URL completa de la imagen"""
        if self.imagen:
            return self.imagen.url
        return None



class Usuario(models.Model):
    correo = models.EmailField(unique=True, max_length=255)
    password = models.CharField(max_length=128)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    fcm_token = models.CharField(max_length=255, blank=True, null=True)

    class Meta:
        db_table = 'users_usuario'

    def __str__(self):
        return f"{self.correo}"

class Cliente(models.Model):
    # Usa OneToOneField como PK y personaliza el nombre de columna en la BD
    usuario = models.OneToOneField(
        Usuario,
        on_delete=models.CASCADE,
        primary_key=True,
        db_column='id'   # aquí defines el nombre de la columna en la tabla users_cliente
    )
    apellidoMaterno = models.CharField(max_length=100)
    apellidoPaterno = models.CharField(max_length=100)
    nombres = models.CharField(max_length=100)
    ci = models.CharField(max_length=20)
    telefono = models.CharField(max_length=20, blank=True, null=True)

    class Meta:
        db_table = 'users_cliente'

    def __str__(self):
        return f"Cliente: {self.nombres} {self.apellidoPaterno} {self.apellidoMaterno}"


class MetodoPago(models.Model):
    nombre = models.CharField(max_length=100)
    descripcion = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    estado = models.BooleanField(default=True)  # Indica si el método de pago está activo

    class Meta:
        db_table = 'sales_metodopago'

    def __str__(self):
        return self.nombre
    

    

class NotaVenta(models.Model):
    ESTADO_CHOICES = [
        ('pendiente', 'Pendiente'),
        ('pagada', 'Pagada'),
        ('fallida', 'Fallida'),
        ('cancelada', 'Cancelada'),
        ('reembolsada', 'Reembolsada'),
    ]
    
    estado = models.CharField(max_length=20, choices=ESTADO_CHOICES, default='pendiente')
    metodo_pago = models.ForeignKey(MetodoPago, on_delete=models.PROTECT, related_name='notas_venta')
    total = models.DecimalField(max_digits=10, decimal_places=2)    
    # Relacion con usuario
    usuario = models.ForeignKey(Usuario, on_delete=models.PROTECT, related_name='notas_venta')

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Stripe
    stripe_session_id = models.CharField(max_length=255, blank=True, null=True)
    stripe_payment_intent = models.CharField(max_length=255, blank=True, null=True)
    
    class Meta:
        db_table = 'sales_notaventa'
        ordering = ['-created_at']

    def __str__(self):
        return f"NotaVenta #{self.id} - {self.usuario.correo}"

class Detalle_Venta(models.Model):
    nota_venta = models.ForeignKey(NotaVenta, on_delete=models.CASCADE, related_name='detalles')
    producto = models.ForeignKey(Producto, on_delete=models.PROTECT, related_name='detalles_venta')
    cantidad = models.PositiveIntegerField()
    precio_unitario = models.DecimalField(max_digits=10, decimal_places=2)
    subtotal = models.DecimalField(max_digits=10, decimal_places=2)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'sales_detalleventa'
    
    def save(self, *args, **kwargs):
        """Calcula automáticamente el subtotal antes de guardar"""
        self.subtotal = self.cantidad * self.precio_unitario
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.producto.nombre} x{self.cantidad} - NotaVenta #{self.nota_venta.id}"


