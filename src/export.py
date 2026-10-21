"""
Exportador de Modelos para Despliegue Móvil (ONNX, TFLite, CoreML)
"""

import os
import copy
import numpy as np
import torch


class MobileModelExporter:
    def __init__(self, model, input_shape=(1, 3, 224, 224)):
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.model.to("cpu")
        self.model.reparameterize_for_deployment()
        self.input_shape = input_shape
        self.dummy_input = torch.randn(*input_shape, requires_grad=False)

    def export_onnx(self, output_path="fer_mobile.onnx"):
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        print(f"\n[1/3] Exportando modelo a ONNX: {output_path}...")
        torch.onnx.export(
            self.model,
            self.dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input_face'],
            output_names=['emotion_logits'],
            dynamic_axes=None  # Dimensiones estáticas para aceleración óptima en NPU
        )
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[OK] ONNX generado: {output_path} ({size_mb:.2f} MB)")
        return output_path

    def export_tflite_with_int8(self, onnx_path, calibration_loader=None, output_dir="mobile_artifacts"):
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n[2/3] Generando artefactos TFLite en {output_dir}...")
        try:
            import tensorflow as tf
            import onnx
            from onnx_tf.backend import prepare

            onnx_model = onnx.load(onnx_path)
            tf_rep = prepare(onnx_model)
            tf_saved_path = os.path.join(output_dir, "saved_model")
            tf_rep.export_graph(tf_saved_path)

            # TFLite FP16
            converter_fp16 = tf.lite.TFLiteConverter.from_saved_model(tf_saved_path)
            converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
            converter_fp16.target_spec.supported_types = [tf.float16]
            tflite_fp16 = converter_fp16.convert()
            fp16_path = os.path.join(output_dir, "fer_model_fp16.tflite")
            with open(fp16_path, "wb") as f:
                f.write(tflite_fp16)
            print(f"[OK] TFLite FP16 guardado: {fp16_path} ({len(tflite_fp16)/(1024*1024):.2f} MB)")

            # TFLite INT8 (con calibración representativa)
            if calibration_loader is not None:
                converter_int8 = tf.lite.TFLiteConverter.from_saved_model(tf_saved_path)
                converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]

                def representative_dataset():
                    count = 0
                    for images, _ in calibration_loader:
                        for i in range(images.size(0)):
                            sample = images[i:i+1].numpy().transpose(0, 2, 3, 1)
                            yield [sample.astype(np.float32)]
                            count += 1
                            if count >= 100:
                                return

                converter_int8.representative_dataset = representative_dataset
                converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter_int8.inference_input_type = tf.float32
                converter_int8.inference_output_type = tf.float32
                tflite_int8 = converter_int8.convert()
                int8_path = os.path.join(output_dir, "fer_model_int8.tflite")
                with open(int8_path, "wb") as f:
                    f.write(tflite_int8)
                print(f"[OK] TFLite INT8 guardado: {int8_path} ({len(tflite_int8)/(1024*1024):.2f} MB)")
        except ImportError:
            print("[INFO] onnx_tf o tensorflow no están instalados. Para generar TFLite ejecutar:")
            print("  uv add onnx-tf tensorflow")

    def export_coreml(self, output_path="fer_mobile.mlpackage"):
        print(f"\n[3/3] Exportando a CoreML: {output_path}...")
        try:
            import coremltools as ct
            traced_model = torch.jit.trace(self.model, self.dummy_input)
            coreml_model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(name="input_face", shape=self.input_shape)],
                minimum_deployment_target=ct.target.iOS16
            )
            coreml_model.save(output_path)
            print(f"[OK] CoreML generado: {output_path}")
        except ImportError:
            print("[INFO] coremltools no instalado. Para exportar a iOS ejecutar: uv add coremltools")

    def verify_numerical_parity(self, onnx_path):
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            with torch.no_grad():
                pytorch_out = self.model(self.dummy_input).numpy()

            onnx_inputs = {session.get_inputs()[0].name: self.dummy_input.numpy()}
            onnx_out = session.run(None, onnx_inputs)[0]

            max_diff = float(np.max(np.abs(pytorch_out - onnx_out)))
            print(f"\n[PARIDAD NUMÉRICA] PyTorch vs ONNX:")
            print(f"  Diferencia absoluta máxima: {max_diff:.6e}")
            assert max_diff < 1e-4, "Alerta: Discrepancia numérica superior a 1e-4."
            print("  [OK] Paridad numérica confirmada con éxito.")
            return True
        except Exception as e:
            print(f"[AVISO] Error al verificar paridad: {e}")
            return False
