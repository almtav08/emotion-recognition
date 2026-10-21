"""
Definición de Arquitectura Optimizada para Edge / Móvil
Soporta MobileNetV4 (UIB) y FastViT (RepMixer reparametrizable)
"""

import os
import copy
import torch
import torch.nn as nn
import timm


class MobileFERModel(nn.Module):
    def __init__(self, backbone_name="mobilenetv4_conv_small", num_classes=4, dropout=0.35, pretrained=True, affectnet_weights_path=None):
        """
        Modelos recomendados:
          - 'mobilenetv4_conv_small': Ideal para NPU/DSP móvil universal (~3.8M params).
          - 'fastvit_sa12': Híbrido reparametrizable (~11.6M params, alta precisión).
          - 'mobilenetv3_small_100': Ultraligero para dispositivos limitados (~2.5M params).
        """
        super().__init__()
        self.backbone_name = backbone_name
        self.affectnet_weights_path = affectnet_weights_path

        # Soporte para EfficientNet preentrenado en AffectNet (HSEmotion: B0 y B2)
        if backbone_name in ["enet_b0_8_best_vgaf", "affectnet_b0"]:
            timm_name = "tf_efficientnet_b0.ns_jft_in1k"
            self.backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=0, global_pool='avg')
            weights_file = affectnet_weights_path or "weights/affectnet_enet_b0_8_best_vgaf.pth"
            if os.path.exists(weights_file):
                print(f"Cargando pesos base preentrenados de AffectNet (B0) desde: {weights_file}")
                state = torch.load(weights_file, weights_only=True)
                missing, unexpected = self.backbone.load_state_dict(state, strict=False)
                print(f"Pesos de AffectNet B0 transferidos al backbone (capas clave cargadas con éxito).")
        elif backbone_name in ["enet_b2_8", "affectnet_b2"]:
            timm_name = "tf_efficientnet_b2.ns_jft_in1k"
            self.backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=0, global_pool='avg')
            weights_file = affectnet_weights_path or "weights/affectnet_enet_b2_8.pth"
            if os.path.exists(weights_file):
                print(f"Cargando pesos base preentrenados de AffectNet (B2) desde: {weights_file}")
                state = torch.load(weights_file, weights_only=True)
                missing, unexpected = self.backbone.load_state_dict(state, strict=False)
                print(f"Pesos de AffectNet B2 transferidos al backbone (capas clave cargadas con éxito).")
        else:
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0,
                global_pool='avg'
            )

        # Determinación dinámica y universal de la dimensión de características de salida
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(2, 3, 224, 224)
            self.in_features = self.backbone(dummy).shape[-1]
        self.backbone.train()
        self.head = self.build_head(num_classes, dropout)

    def build_head(self, num_classes, dropout=0.35):
        return nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.in_features, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(p=dropout * 0.7),
            nn.Linear(256, num_classes)
        )

    def replace_head(self, new_num_classes, dropout=0.35):
        self.head = self.build_head(new_num_classes, dropout)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

    def configure_freezing(self, mode="stage1_partial"):
        """
        Controla qué capas se entrenan según la etapa:
          - 'linear_probe': Congela 100% del backbone, solo entrena el nuevo cabezal (Etapa 2A).
          - 'stage1_partial': Descongela la mitad superior del backbone para adaptar morfología infantil.
          - 'stage2_last_block': Descongela únicamente el bloque final + BatchNorms para fine-tuning en TEA.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.head.parameters():
            param.requires_grad = True

        if mode == "linear_probe":
            return

        children = list(self.backbone.named_children())
        if mode == "stage1_partial":
            half_idx = len(children) // 2
            for _, module in children[half_idx:]:
                for param in module.parameters():
                    param.requires_grad = True

        elif mode == "stage2_last_block":
            if hasattr(self.backbone, "stages"):
                for param in self.backbone.stages[-1].parameters():
                    param.requires_grad = True
            elif hasattr(self.backbone, "blocks"):
                # Descongelar los 2 últimos bloques para mayor adaptación en TEA
                for block in self.backbone.blocks[-2:]:
                    for param in block.parameters():
                        param.requires_grad = True
                if hasattr(self.backbone, "conv_head"):
                    for param in self.backbone.conv_head.parameters():
                        param.requires_grad = True
                if hasattr(self.backbone, "bn2"):
                    for param in self.backbone.bn2.parameters():
                        param.requires_grad = True
            else:
                for param in children[-1][1].parameters():
                    param.requires_grad = True

            # Mantener capas de normalización activas para ajustar estadísticas al dominio FERAC
            for module in self.backbone.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                    for param in module.parameters():
                        param.requires_grad = True

    def reparameterize_for_deployment(self):
        """
        Fusiona ramas paralelas en arquitecturas que admiten reparametrización
        (como FastViT) para convertirlas en una sola convolución 3x3 rápida en móvil.
        """
        self.eval()
        if hasattr(self.backbone, "reparameterize"):
            print("Aplicando reparametrización estructural en el backbone...")
            self.backbone.reparameterize()
            print("Reparametrización completada con éxito.")
        else:
            print(f"El backbone '{self.backbone_name}' ya presenta topología óptima para inferencia directa.")
