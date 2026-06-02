import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepArchitecture(nn.Module):
    def __init__(self):
        super(DeepArchitecture, self).__init__()
        
        # 1. KODLAYICI BÖLÜMÜ (ENCODER)
        # Ham veriden özellik çıkarımı yaparak 'Yeni Temsil' oluşturur
        self.encoder_conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.encoder_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # 2. YENİ TEMSİL KATMANI (REPRESENTATION LAYER)
        # Verinin en öz ve sıkıştırılmış halinin bulunduğu aşama
        self.fc_representation = nn.Linear(64 * 7 * 7, 128)
        
        # 3. KOD ÇÖZÜCÜ / ÇIKTI BÖLÜMÜ (DECODER / OUTPUT)
        # Temsil edilen veriyi sınıflandırarak sonuca ulaştırır
        self.decoder_fc = nn.Linear(128, 10) 
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Adım 1: Kodlama süreci
        x = self.pool(F.relu(self.encoder_conv1(x)))
        x = self.pool(F.relu(self.encoder_conv2(x)))
        
        # Adım 2: Veriyi düzleştirme ve Temsil katmanına geçiş
        x = x.view(-1, 64 * 7 * 7)
        representation = F.relu(self.fc_representation(x))
        
        # Adım 3: Çıktı üretimi
        x = self.dropout(representation)
        output = self.decoder_fc(x)
        
        return output

# Modeli başlatma ve mimariyi doğrulama
model = DeepArchitecture()
print("--- Proje Model Mimarisi ---")
print(model)
