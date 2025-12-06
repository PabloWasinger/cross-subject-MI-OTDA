import torch
import torch.nn as nn

class EEGNet(nn.Module):
    def __init__(self, nb_classes, Chans=64, Samples=128, 
                 dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16):
        super(EEGNet, self).__init__()
        
        # --- BLOQUE 1: Filtrado Temporal y Espacial ---
        # 1. Convolución Temporal (Filtros de Frecuencia)
        # Input: (Batch, 1, Chans, Samples) -> Output: (Batch, F1, Chans, Samples)
        self.conv1 = nn.Conv2d(in_channels=1, 
                               out_channels=F1, 
                               kernel_size=(1, kernLength), 
                               padding=(0, kernLength // 2), 
                               bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        
        # 2. Convolución Depthwise (Filtros Espaciales)
        # Aquí usamos 'groups=F1' para que cada filtro temporal tenga sus propios filtros espaciales.
        # Input: (Batch, F1, Chans, Samples) -> Output: (Batch, F1*D, 1, Samples)
        self.depthwiseConv = nn.Conv2d(in_channels=F1, 
                                       out_channels=F1 * D, 
                                       kernel_size=(Chans, 1), 
                                       groups=F1, 
                                       bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu1 = nn.ELU()
        self.avgpool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(p=dropoutRate)

        # --- BLOQUE 2: Convolución Separable ---
        # La conv separable se hace en dos pasos en PyTorch:
        
        # Paso 2.a: Depthwise (Temporal)
        self.separableConv_depth = nn.Conv2d(in_channels=F1 * D, 
                                             out_channels=F1 * D, 
                                             kernel_size=(1, 16), 
                                             padding=(0, 8), 
                                             groups=F1 * D, 
                                             bias=False)
        
        # Paso 2.b: Pointwise (Mezcla de características)
        self.separableConv_point = nn.Conv2d(in_channels=F1 * D, 
                                             out_channels=F2, 
                                             kernel_size=(1, 1), 
                                             bias=False)
                                             
        self.bn3 = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()
        self.avgpool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = nn.Dropout(p=dropoutRate)
        
        # --- CLASIFICADOR ---
        self.flatten = nn.Flatten()
        
        # Calculamos el tamaño de entrada para la capa lineal dinámicamente
        # Esto depende de los pooling layers ((Samples / 4) / 8)
        self.linear_input_size = F2 * (Samples // 32)
        self.classifier = nn.Linear(self.linear_input_size, nb_classes)

    def forward(self, x):
        # x shape esperado: (Batch, 1, Chans, Samples)
        
        # Bloque 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwiseConv(x) # Aquí los canales (Chans) se reducen a 1
        x = self.bn2(x)
        x = self.elu1(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        
        # Bloque 2
        x = self.separableConv_depth(x)
        x = self.separableConv_point(x)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.avgpool2(x) # Salida reducida en tiempo
        x = self.dropout2(x)
        
        # Clasificación
        x = self.flatten(x)
        x = self.classifier(x)
        
        return x

def main():

    # Configuración basada en BCI Competition IV 2a (BNCI2014_001)
    config_recomendada = {
        "nb_classes":2,        
        "Chans":22,            # 22 electrodos EEG 
        "Samples":750,        # dim de la entrada, VER CANTIDAD DE SEGUNDOS
        "dropoutRate":0.5,     # ME DIJO CLAUDIO
        "kernLength":125,      # cant de filtros, mitad de la frec
        "F1":8,                # filtros temporales, paper recomiendoa
        "D":2,                 # filtros espaciales, paper recomienda
        "F2":16                # F1*D
    }




if __name__ == "__main__":
    main()
        

"""FLUJO PARA FINE TUNING:

import torch
import torch.optim as optim

# 1. Instancias el modelo con tu configuración
modelo = EEGNet(nb_classes=2, Chans=22, Samples=1000, kernLength=125)

# 2. IMPORTANTE: Cargar los pesos pre-entrenados (el "conocimiento previo")
# Si no haces esto, estarás congelando una capa con ruido aleatorio inservible.
# modelo.load_state_dict(torch.load('pesos_sujeto_general.pth'))
print("Pesos cargados exitosamente.")

# ---------------------------------------------------------
# 3. AQUÍ aplicas el código de congelamiento (Fine-Tuning)
# ---------------------------------------------------------

# Congelar filtros temporales (Conv1) - No se tocarán
for param in modelo.conv1.parameters():
    param.requires_grad = False

# Congelar Batch Norm asociado - No se tocará
for param in modelo.bn1.parameters():
    param.requires_grad = False

# Nota: No hace falta poner 'True' en depthwiseConv,
# porque en PyTorch requires_grad es True por defecto al crear el modelo.

print("Capas temporales congeladas para fine-tuning.")

# ---------------------------------------------------------
# 4. Defines el Optimizador
# ---------------------------------------------------------

# Es buena práctica pasarle al optimizador solo los parámetros que requieren gradiente.
# Esto hace el cómputo un poco más eficiente.
parametros_entrenables = filter(lambda p: p.requires_grad, modelo.parameters())

optimizer = optim.Adam(parametros_entrenables, lr=0.0001) # Learning rate bajo

# 5. Bucle de entrenamiento normal...
# for data, target in train_loader:
#     optimizer.zero_grad()
#     output = modelo(data)
#     ...
"""